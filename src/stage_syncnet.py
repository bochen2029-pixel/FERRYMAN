#!/usr/bin/env python
"""
stage_syncnet.py  --  FERRYMAN lip-sync confidence oracle (LatentSync stable_syncnet).

Scores audio-visual lip-sync confidence for a talking-head clip using the on-disk
LatentSync weights (models/latentsync16/stable_syncnet.pt) and the vendored MuseTalk
architecture (vendor/MuseTalk/musetalk/models/syncnet.py, "modified from LatentSync
stable_syncnet.py").

Ground truth established from the checkpoint state-dict (NOT the vendored yaml, whose
attn_blocks are all-zero and do NOT match this checkpoint):

  audio_encoder : in=1,  block_out=[32,64,128,256,512,1024,2048],
                  downsample=[[2,1],2,2,1,2,2,[2,3]], attn after stages 3,4
  visual_encoder: in=48, block_out=[64,128,256,256,512,1024,2048,2048],
                  downsample=[[1,2],2,2,2,2,2,2,2], attn after stages 4,5
  -> both encoders emit 2048-d embeddings; score = cosine of L2-normed embeddings.

Window geometry (verified against checkpoint conv shapes + LatentSync/Wav2Lip mel):
  audio  window = 0.64 s @ 16 kHz -> Wav2Lip mel (n_fft=800, hop=200, win=800, 80 mels,
         fmin=55, fmax=7600, preemph, symmetric norm +/-4)  ==  (1, 80, 52)   [EXACT]
  visual window = 16 consecutive frames @ 25 fps, mouth crop (lower half of face box)
         resized to W=256 x H=128, RGB, stacked along channel -> (48, 128, 256)
  pixels normalized to [-1, 1]  (LatentSync stable_syncnet convention).

The checkpoint's AttentionBlock2D would require xformers (absent in every venv here); we
reconstruct a byte-identical, xformers-free AttentionBlock2D (diffusers default attention
processor) so the state-dict loads with 0 missing / 0 unexpected keys.

CLI CONTRACT (fixed):
    <venv-python> stage_syncnet.py <task.json>
    task.json = {"video": "<mp4>", "audio": "<wav|mp4-with-audio>", "windows": 24}
    on success -> writes <task.json>.result =
        {"sync_conf_mean": float, "sync_conf_min": float, "sync_conf_p10": float,
         "windows_scored": int, "venv": "<name>", "note": "<one line>"}
      and exits 0.
    on any hard failure -> prints an actionable message to stderr, exits nonzero,
      and NEVER writes a fabricated result (fail-closed).

Windows are sampled uniformly across the clip, skipping the first/last ~0.5 s.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import traceback

# --- constants (derived above; do not change without re-running the falsifier) ---
FPS = 25                      # LatentSync/MuseTalk operate at 25 fps
NUM_FRAMES = 16               # frames per visual window
AUDIO_SR = 16000
WIN_SEC = NUM_FRAMES / FPS    # 0.64 s
MEL_BINS = 80
MEL_FRAMES = 52              # (1,80,52) expected by the audio encoder
CROP_W = 256                 # mouth crop width  (visual (48,128,256): H=128, W=256)
CROP_H = 128                 # mouth crop height
EDGE_SKIP_SEC = 0.5          # skip first/last ~0.5 s

VENDOR_ROOT = r"C:/FERRYMAN/vendor/MuseTalk"
CKPT_PATH = r"C:/FERRYMAN/models/latentsync16/stable_syncnet.pt"

# checkpoint-accurate config (attention interleaved as separate down_block entries)
SYNCNET_CONFIG = {
    "audio_encoder": {
        "in_channels": 1,
        "block_out_channels": [32, 64, 128, 256, 512, 1024, 2048],
        "downsample_factors": [[2, 1], 2, 2, 1, 2, 2, [2, 3]],
        "attn_blocks": [0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
    "visual_encoder": {
        "in_channels": 48,
        "block_out_channels": [64, 128, 256, 256, 512, 1024, 2048, 2048],
        "downsample_factors": [[1, 2], 2, 2, 2, 2, 2, 2, 2],
        "attn_blocks": [0, 0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
}


def die(msg, code=2):
    sys.stderr.write("stage_syncnet FAILURE: " + msg.rstrip() + "\n")
    sys.stderr.flush()
    sys.exit(code)


def _run(cmd):
    """Run a subprocess with UTF-8 forced; return (rc, stdout, stderr)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


# ----------------------------------------------------------------------------------
# Model: reconstruct SyncNet with an xformers-free AttentionBlock2D so the checkpoint
# (which DOES contain attention weights) loads cleanly.  We import the vendored
# ResnetBlock2D / DownEncoder2D / SyncNet and only swap the attention block.
# ----------------------------------------------------------------------------------
def build_model(device):
    import torch
    import torch.nn as nn
    from einops import rearrange
    from diffusers.models.attention import Attention as CrossAttention, FeedForward

    sys.path.insert(0, VENDOR_ROOT)
    import musetalk.models.syncnet as vsync

    class AttentionBlock2D(nn.Module):
        """Same structure/keys as the vendored AttentionBlock2D but without the
        xformers hard-requirement (diffusers default attention processor)."""

        def __init__(self, query_dim, norm_num_groups=32, dropout=0.0):
            super().__init__()
            self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
            self.norm2 = nn.LayerNorm(query_dim)
            self.norm3 = nn.LayerNorm(query_dim)
            self.ff = FeedForward(query_dim, dropout=dropout, activation_fn="geglu")
            self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
            self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
            self.attn = CrossAttention(query_dim=query_dim, heads=8, dim_head=query_dim // 8,
                                       dropout=dropout, bias=True)

        def forward(self, hidden_states):
            _, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.conv_in(hidden_states)
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
            norm_hidden_states = self.norm2(hidden_states)
            hidden_states = self.attn(norm_hidden_states, attention_mask=None) + hidden_states
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
            hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height, w=width)
            hidden_states = self.conv_out(hidden_states)
            return hidden_states + residual

    vsync.AttentionBlock2D = AttentionBlock2D  # runtime swap; vendor file untouched

    model = vsync.SyncNet(SYNCNET_CONFIG)
    if not os.path.isfile(CKPT_PATH):
        die("checkpoint not found: " + CKPT_PATH)
    state = torch.load(CKPT_PATH, map_location="cpu")
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        die("checkpoint key mismatch (missing={}, unexpected={}); architecture does not "
            "match weights. First missing={}, first unexpected={}".format(
                len(missing), len(unexpected), missing[:3], unexpected[:3]))
    model.eval().to(device)
    return model


# ----------------------------------------------------------------------------------
# Extraction helpers
# ----------------------------------------------------------------------------------
def extract_frames(video, out_dir):
    """Decode the whole video to 25 fps RGB JPEGs (frame%06d.jpg, 1-indexed)."""
    pattern = os.path.join(out_dir, "frame%06d.jpg")
    rc, so, se = _run([
        "ffmpeg", "-y", "-i", video,
        "-vf", "fps={}".format(FPS),
        "-q:v", "2", "-start_number", "1", pattern,
    ])
    if rc != 0:
        die("ffmpeg frame extraction failed (rc={}):\n{}".format(rc, se[-1500:]))
    frames = sorted(f for f in os.listdir(out_dir) if f.startswith("frame") and f.endswith(".jpg"))
    if len(frames) < NUM_FRAMES + 4:
        die("too few frames decoded ({}); clip may be shorter than one window".format(len(frames)))
    return [os.path.join(out_dir, f) for f in frames]


def extract_wav(audio_src, out_wav):
    """Extract mono 16 kHz PCM wav from any container/stream."""
    rc, so, se = _run([
        "ffmpeg", "-y", "-i", audio_src,
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "wav", out_wav,
    ])
    if rc != 0 or not os.path.isfile(out_wav) or os.path.getsize(out_wav) < 1024:
        die("ffmpeg audio extraction failed for '{}' (rc={}); the file may have no audio "
            "stream:\n{}".format(audio_src, rc, se[-1500:]))
    return out_wav


# ----------------------------------------------------------------------------------
# Face / mouth localization (dependency-light: OpenCV Haar, detected once)
# ----------------------------------------------------------------------------------
def locate_mouth_box(frame_paths):
    """Detect the face box on a handful of frames; return a stable (x,y,w,h) mouth box
    = lower half of the median face box.  Returns None if no face is ever found."""
    import cv2
    import numpy as np

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        die("could not load OpenCV Haar face cascade at " + cascade_path)

    # sample up to ~12 frames spread across the clip
    n = len(frame_paths)
    probe_idx = sorted(set(int(i) for i in np.linspace(n * 0.1, n * 0.9, 12)))
    boxes = []
    for i in probe_idx:
        img = cv2.imread(frame_paths[i])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(80, 80))
        if len(faces):
            # largest face in this frame
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            boxes.append([fx, fy, fw, fh])
    if not boxes:
        return None
    fx, fy, fw, fh = np.median(np.array(boxes), axis=0).astype(int)
    # mouth region = lower half of the face box
    mx = fx
    my = fy + fh // 2
    mw = fw
    mh = fh - fh // 2
    return int(mx), int(my), int(mw), int(mh)


def load_mouth_stack(frame_paths, start, mouth_box):
    """Build a (48,128,256) float tensor-ready ndarray for NUM_FRAMES starting at `start`.
    Pixels normalized to [-1,1].  Channel order = [f0_R,f0_G,f0_B, f1_R,...]."""
    import cv2
    import numpy as np

    mx, my, mw, mh = mouth_box
    chans = []
    for k in range(NUM_FRAMES):
        img = cv2.imread(frame_paths[start + k])
        if img is None:
            return None
        H, W = img.shape[:2]
        x0 = max(0, mx); y0 = max(0, my)
        x1 = min(W, mx + mw); y1 = min(H, my + mh)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)  # (H,W,3)
        crop = crop.astype(np.float32) / 255.0
        crop = crop * 2.0 - 1.0        # [-1,1]
        chans.append(np.transpose(crop, (2, 0, 1)))  # (3,H,W)
    return np.concatenate(chans, axis=0)  # (48,H,W)


# ----------------------------------------------------------------------------------
# Audio -> mel window
# ----------------------------------------------------------------------------------
def compute_mel(wav_path):
    """Full-clip Wav2Lip/LatentSync mel: (80, T). Also return frames-per-second of mel."""
    sys.path.insert(0, VENDOR_ROOT)
    from musetalk.data import audio as A
    import numpy as np

    wav = A.load_wav(wav_path, AUDIO_SR)
    if wav is None or len(wav) < AUDIO_SR * (WIN_SEC + 2 * EDGE_SKIP_SEC):
        die("audio too short to score any window")
    mel = A.melspectrogram(wav).astype(np.float32)  # (80, T)
    if mel.shape[0] != MEL_BINS:
        die("unexpected mel bins {} (want {})".format(mel.shape[0], MEL_BINS))
    mel_fps = 1.0 / (A.get_hop_size() / float(AUDIO_SR))  # 16000/200 = 80 fps
    return mel, mel_fps


def mel_window(mel, mel_fps, t_center_sec):
    """Extract a (1,80,52) mel window centered on t_center_sec (the middle of the visual
    window).  Pad-edge if we run past the boundary (shouldn't, given edge skip)."""
    import numpy as np
    start_sec = t_center_sec - WIN_SEC / 2.0
    start = int(round(start_sec * mel_fps))
    end = start + MEL_FRAMES
    T = mel.shape[1]
    if start < 0:
        start, end = 0, MEL_FRAMES
    if end > T:
        end, start = T, T - MEL_FRAMES
    win = mel[:, start:end]
    if win.shape[1] != MEL_FRAMES:
        # last-resort pad
        pad = MEL_FRAMES - win.shape[1]
        win = np.pad(win, ((0, 0), (0, max(0, pad))), mode="edge")[:, :MEL_FRAMES]
    return win[np.newaxis, :, :]  # (1,80,52)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        die("usage: stage_syncnet.py <task.json>", code=2)
    task_path = sys.argv[1]
    if not os.path.isfile(task_path):
        die("task.json not found: " + task_path)
    try:
        with open(task_path, "r", encoding="utf-8") as fh:
            task = json.load(fh)
    except Exception as e:
        die("could not parse task.json: {}".format(e))

    video = task.get("video")
    audio = task.get("audio") or video
    windows = int(task.get("windows", 24))
    if not video or not os.path.isfile(video):
        die("task.video missing or not a file: {}".format(video))
    if not os.path.isfile(audio):
        die("task.audio missing or not a file: {}".format(audio))
    if windows < 1:
        die("windows must be >= 1")

    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    venv_name = os.path.basename(os.path.dirname(os.path.dirname(sys.executable)))

    work = tempfile.mkdtemp(prefix="syncnet_", dir=r"C:/FERRYMAN/work/syncnet_probe")
    try:
        # 1) extract frames @25fps + audio wav
        frame_paths = extract_frames(video, work)
        wav_path = extract_wav(audio, os.path.join(work, "audio16k.wav"))

        n_frames = len(frame_paths)
        clip_sec = n_frames / float(FPS)

        # 2) localize mouth box once
        mouth_box = locate_mouth_box(frame_paths)
        if mouth_box is None:
            die("no face detected anywhere in the clip; cannot form mouth crops "
                "(is this a talking-head video?)")

        # 3) mel for whole clip
        mel, mel_fps = compute_mel(wav_path)

        # 4) uniform window start frames, skipping first/last ~0.5s
        lo = int(round(EDGE_SKIP_SEC * FPS))
        hi = n_frames - NUM_FRAMES - int(round(EDGE_SKIP_SEC * FPS))
        if hi <= lo:
            die("clip too short to place any window after edge-skip "
                "(frames={}, need >= {})".format(n_frames, 2 * lo + NUM_FRAMES))
        n_win = min(windows, hi - lo + 1)
        starts = [int(round(s)) for s in np.linspace(lo, hi, n_win)]

        # 5) build model, score each window
        model = build_model(device)
        scores = []
        with torch.no_grad():
            for st in starts:
                stack = load_mouth_stack(frame_paths, st, mouth_box)
                if stack is None:
                    continue
                t_center = (st + NUM_FRAMES / 2.0) / float(FPS)
                mw = mel_window(mel, mel_fps, t_center)
                v = torch.from_numpy(stack).unsqueeze(0).to(device)           # (1,48,128,256)
                a = torch.from_numpy(mw).unsqueeze(0).to(device)              # (1,1,80,52)
                vis_e, aud_e = model(v, a)
                cos = torch.nn.functional.cosine_similarity(vis_e, aud_e).item()
                scores.append(cos)

        if not scores:
            die("no windows could be scored (all crops empty?)")

        scores = np.array(scores, dtype=np.float64)
        result = {
            "sync_conf_mean": float(np.mean(scores)),
            "sync_conf_min": float(np.min(scores)),
            "sync_conf_p10": float(np.percentile(scores, 10)),
            "windows_scored": int(len(scores)),
            "venv": venv_name,
            "note": "LatentSync stable_syncnet; cosine of 2048-d embeds; "
                    "mouth=lower-half face box @256x128, mel(1,80,52)@0.64s, {} @25fps, "
                    "device={}".format(os.path.basename(video), device),
        }
        out_path = task_path + ".result"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.exit(0)

    except SystemExit:
        raise
    except Exception:
        die("unhandled error:\n" + traceback.format_exc())
    finally:
        # keep scratch only if asked; default clean
        if os.environ.get("SYNCNET_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
