"""Generate docs/USER_MANUAL.pdf — end-user cheatsheet for the FERRYMAN video factory.
Regenerate any time:  python docs/make_user_manual.py"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = str(ROOT / "docs" / "USER_MANUAL.pdf")

# ---- fonts: try Microsoft YaHei for CJK; fall back to Helvetica (English-only)
FONT, FONT_B, CJK = "Helvetica", "Helvetica-Bold", False
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("YaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    try:
        pdfmetrics.registerFont(TTFont("YaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
        FONT_B = "YaHei-Bold"
    except Exception:
        FONT_B = "YaHei"
    FONT, CJK = "YaHei", True
except Exception:
    pass

zh = (lambda s: s) if CJK else (lambda s: "")

INK = colors.HexColor("#1a2433")
ACCENT = colors.HexColor("#0e6ba8")
MUTED = colors.HexColor("#5a6675")
BG = colors.HexColor("#eef3f8")

S = dict(
    h1=ParagraphStyle("h1", fontName=FONT_B, fontSize=22, leading=26, textColor=INK),
    sub=ParagraphStyle("sub", fontName=FONT, fontSize=10.5, leading=14, textColor=MUTED),
    h2=ParagraphStyle("h2", fontName=FONT_B, fontSize=13.5, leading=17, textColor=ACCENT,
                      spaceBefore=14, spaceAfter=4),
    body=ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=15, textColor=INK),
    step=ParagraphStyle("step", fontName=FONT, fontSize=11, leading=16.5, textColor=INK,
                        leftIndent=14),
    mono=ParagraphStyle("mono", fontName="Courier", fontSize=8.8, leading=12,
                        textColor=INK, backColor=BG, borderPadding=6),
    small=ParagraphStyle("small", fontName=FONT, fontSize=9, leading=12.5, textColor=MUTED),
)


def P(text, style="body"):
    return Paragraph(text, S[style])


def rule():
    return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#c9d4e0"),
                      spaceBefore=2, spaceAfter=2)


doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=0.55 * inch,
                        bottomMargin=0.55 * inch, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, title="FERRYMAN User Manual",
                        author="FERRYMAN")
story = []

story += [P("FERRYMAN — User Manual", "h1"),
          P("The local video factory: a text script becomes a finished video of the speaker "
            "saying it, in their own cloned voice. Everything runs on this PC.", "sub"),
          Spacer(1, 6), rule()]

story += [P("1 · Make an episode (the whole job)", "h2"),
          P("① Save the script as a plain text file, e.g. <font face='Courier'>jobs\\ep02.txt</font> "
            "(Chinese text, one episode per file).", "step"),
          P("② Copy a job file: duplicate any <font face='Courier'>.job.json</font> from "
            "<font face='Courier'>jobs\\done\\</font>, rename it (e.g. <font face='Courier'>ep02.job.json</font>), "
            "and edit the two lines that matter — <b>job_id</b> (any new unique name) and "
            "<b>script_path</b> (your new script file).", "step"),
          P("③ Drop the job file into <font face='Courier'>jobs\\inbox\\</font>", "step"),
          P("④ Wait. The factory checks the inbox every 30 minutes and renders automatically "
            "(even if you are logged out). In a hurry? Double-click "
            "<font face='Courier'>bin\\batch_task.cmd</font> to start immediately.", "step"),
          P("⑤ Collect the video: <font face='Courier'>out\\&lt;job_id&gt;\\final.mp4</font> "
            "— captions burned in, loudness mastered, ready to watch.", "step"),
          Spacer(1, 4)]

story += [P("A job file, in full (7 lines you can copy):", "body"), Spacer(1, 3),
          P('{&nbsp;&nbsp;"job_id": "2026-07-12_ep02",<br/>'
            '&nbsp;&nbsp;&nbsp;"speaker": "your_speaker",<br/>'
            '&nbsp;&nbsp;&nbsp;"script_path": "jobs/ep02.txt",<br/>'
            '&nbsp;&nbsp;&nbsp;"lang": "zh",&nbsp;&nbsp;"tier": "T1",<br/>'
            '&nbsp;&nbsp;&nbsp;"captions": true,&nbsp;&nbsp;"label": true&nbsp;&nbsp;}', "mono"),
          Spacer(1, 2),
          P("captions = subtitles on/off. label = the small “" + (zh("AI生成") or "AI-generated")
            + "” mark — keep it <b>true</b> for anything shared publicly (required on Chinese platforms).",
            "small")]

story += [P("2 · Where things live", "h2")]
tbl = Table([
    ["jobs\\inbox\\", "Drop job files here. The factory eats them."],
    ["jobs\\done\\", "Jobs that finished (good templates to copy)."],
    ["jobs\\failed\\", "Jobs that failed, each with an .err note saying why."],
    ["out\\<job_id>\\", "Finished videos: final.mp4 + captions + report card."],
    ["speakers\\", "Enrolled people (voice + face). Private — never share."],
], colWidths=[1.95 * inch, 4.85 * inch])
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (0, -1), "Courier"), ("FONTNAME", (1, 0), (1, -1), FONT),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, BG]),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#d7e0ea")),
]))
story += [tbl]

story += [P("3 · The report card (trust, verified)", "h2"),
          P("Every episode is machine-checked before it counts as done — open "
            "<font face='Courier'>out\\&lt;job_id&gt;\\manifest.jsonl</font> if curious:", "body"),
          P("• <b>audio_cer</b> — did the voice say every word of the script correctly? "
            "(0.00 is perfect; anything over 0.05 is auto-retried, then flagged.)<br/>"
            "• <b>speaker_sim</b> — does it sound like the real person? (above 0.75 = verified match)<br/>"
            "• <b>av_len_delta_ms</b> — are sound and picture in sync? (under 100 ms)<br/>"
            "• <b>pass: true</b> — all green. If a job fails its checks it goes to "
            "<font face='Courier'>jobs\\failed\\</font> instead of pretending it worked.", "step")]

story += [P("4 · Add a new person (one-time, ~5 minutes)", "h2"),
          P("Record on a phone, quiet room: <b>(a)</b> 1–2 minutes of them talking (clean, no music) "
            "and <b>(b)</b> 1–2 minutes of video, camera steady, them facing it. "
            "Then open a terminal and run:", "body"), Spacer(1, 3),
          P("bin\\ferryman.cmd&nbsp; enroll-voice&nbsp; &lt;name&gt;&nbsp; path\\to\\voice.mp3<br/>"
            "bin\\ferryman.cmd&nbsp; make-idle&nbsp;&nbsp;&nbsp; &lt;name&gt;&nbsp; path\\to\\video.mp4", "mono"),
          Spacer(1, 2),
          P("Use the new &lt;name&gt; in the job file's \"speaker\" line. Only enroll people who have "
            "given their consent — that is a hard rule of this machine.", "small")]

story += [P("5 · When something goes wrong", "h2"),
          P("① Look in <font face='Courier'>jobs\\failed\\</font> — read the <font face='Courier'>.err</font> "
            "file next to your job (plain-language reason).<br/>"
            "② Fix the cause — usually: script file path typo, speaker name not enrolled, "
            "or disk nearly full.<br/>"
            "③ Move the <font face='Courier'>.job.json</font> back into "
            "<font face='Courier'>jobs\\inbox\\</font> — it retries automatically. "
            "Already-good voice segments are cached, so retries are fast.", "step")]

story += [P("6 · House rules", "h2"),
          P("• Voices and faces are cloned <b>only with the person's consent</b>.<br/>"
            "• The “" + (zh("AI生成") or "AI-generated") + "” label stays ON for anything published.<br/>"
            "• Recordings, enrolled voices and rendered videos live on this PC and are not uploaded anywhere.<br/>"
            "• Publishing is a human decision: copy the final.mp4 yourself — the machine never posts.", "step"),
          Spacer(1, 10), rule(),
          P("FERRYMAN " + zh("（渡）") + " · local sovereign video factory · manual v1 · 2026-07 · "
            "regenerate: <font face='Courier'>python docs\\make_user_manual.py</font>", "small")]

doc.build(story)
print(f"OK: {OUT} ({os.path.getsize(OUT):,} bytes)")
