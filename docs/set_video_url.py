"""Drop your YouTube URL into the report and rebuild the PDF.

Run this after you record and upload the video:

    python docs/set_video_url.py https://youtu.be/XXXXXXXXXXX

It replaces the placeholder in the Artifacts section, runs the full
pdflatex/bibtex/pdflatex/pdflatex cycle, and leaves you with an updated
docs/ieee_report/main.pdf. Safe to run more than once - passing a new URL
replaces whatever is there now.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent / "ieee_report"
TEX = REPORT_DIR / "main.tex"
PLACEHOLDER = r"\textsc{[URL to be added]}"


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(__file__).name} <youtube-url>")
    url = sys.argv[1].strip()
    if not url.startswith(("http://", "https://")):
        sys.exit(f"that does not look like a URL: {url!r}")

    if not shutil.which("pdflatex"):
        sys.exit("pdflatex not found on PATH - install MiKTeX or TeX Live")

    s = TEX.read_text(encoding="utf-8")
    replacement = f"\\url{{{url}}}"

    if PLACEHOLDER in s:
        s = s.replace(PLACEHOLDER, replacement)
        what = "placeholder replaced"
    else:
        # already set once; swap whatever URL currently follows the label
        pattern = r"(\\textbf\{Video Demonstration:\}\s*)\\url\{[^}]*\}"
        s, n = re.subn(pattern, r"\1" + replacement.replace("\\", "\\\\"), s)
        if not n:
            sys.exit("could not find the Video Demonstration entry in main.tex")
        what = "existing URL updated"

    TEX.write_text(s, encoding="utf-8")
    print(f"{what}: {url}")

    print("rebuilding (this takes ~30s)...")
    for step in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
                 ["bibtex", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                 ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(step, cwd=REPORT_DIR, capture_output=True)

    pdf = REPORT_DIR / "main.pdf"
    if pdf.exists():
        print(f"done -> {pdf}  ({pdf.stat().st_size / 1000:.0f} KB)")
    else:
        sys.exit("build failed - check docs/ieee_report/main.log")


if __name__ == "__main__":
    main()
