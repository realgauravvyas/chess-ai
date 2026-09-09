"""Check that every deliverable quotes the same numbers.

The report, slides, video script, README and RESULTS.md were written and
revised at different times. A figure corrected in one and left stale in
another is exactly the kind of thing a reader notices and a writer does not.

    python tests/check_consistency.py
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
problems, notes = [], []


def read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None


def read_pptx(rel):
    """All slide text, straight out of the .pptx XML."""
    p = ROOT / rel
    if not p.exists():
        return None
    out = []
    with zipfile.ZipFile(p) as z:
        for n in sorted(x for x in z.namelist()
                        if re.match(r"ppt/slides/slide\d+\.xml$", x)):
            out.append(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8")))
    return " ".join(out)


DOCS = {
    "report (main.tex)": read("docs/ieee_report/main.tex"),
    "video script": read("docs/youtube_script.md"),
    "slides": read_pptx("docs/chess_ai_slides.pptx"),
    "README": read("README.md"),
    "RESULTS.md": read("RESULTS.md"),
}
for name, text in DOCS.items():
    if text is None:
        problems.append(f"{name}: file missing")

# ---- headline figures, and the forms each document may write them in ----
FACTS = [
    ("parameter count", [r"760[,\\\s]?717"]),
    ("training positions", [r"676[,\\\s]?648", r"677,?000", r"677k"]),
    ("policy cross-entropy", [r"2\.06"]),
    ("gated model vs baseline", [r"46\.2"]),
    ("candidate vs baseline", [r"42\.2"]),
    ("gate promotions", [r"8\s*/\s*20", r"8 of 20"]),
    ("v5 regression", [r"31\.2", r"2\.5\s*(?:-|--|—|to)\s*5\.5"]),
    ("vs random", [r"86\.2", r"87\.5", r"85\s*%", r"85 to 90", r"85-90"]),
]

print("=== headline figures ===")
for label, pats in FACTS:
    present = []
    for name, text in DOCS.items():
        if text and any(re.search(p, text) for p in pats):
            present.append(name.split()[0])
    print(f"  {label:26s} {', '.join(present) if present else 'NOWHERE'}")
    if not present:
        problems.append(f"{label}: appears in no document")

# ---- counts that drift when tests are added ---------------------------
print("\n=== test counts (these drift) ===")
r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_suite.py")],
                   cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
m = re.search(r"(\d+) passed, (\d+) failed, (\d+) checks", r.stdout)
actual_checks = int(m.group(3)) if m else None
actual_fail = int(m.group(2)) if m else None
print(f"  suite actually reports   : {actual_checks} checks, {actual_fail} failed")
if actual_fail:
    problems.append(f"test suite has {actual_fail} failing checks")

muts = read("tests/test_mutations.py")
actual_muts = muts.count('",\n     "') if muts else 0
actual_muts = len(re.findall(r'^\s{4}\("', muts, re.M)) if muts else 0
print(f"  mutation cases defined   : {actual_muts}")

# A count is only wrong if it claims to describe the present. RESULTS.md
# narrates successive rounds, so a figure explicitly marked as
# point-in-time is correct history, not a stale claim.
HISTORICAL = ("at this point", "stood at", "reached", "as of this round",
              "at the time")

def stale(text, pattern, current, label):
    out = []
    for line in text.splitlines():
        if any(h in line.lower() for h in HISTORICAL):
            continue
        for m in re.finditer(pattern, line):
            got = int(m.group(m.lastindex))
            if current and got != current:
                out.append(f"says '{m.group(0).strip()}' but {label} is {current}")
    return out

for name, text in DOCS.items():
    if not text:
        continue
    for msg in stale(text, r"(\d+)\s+checks", actual_checks, "the suite"):
        problems.append(f"{name}: {msg}")
    for msg in stale(text, r"\d+\s*/\s*(\d+)\s+(?:historical bugs|caught)",
                     actual_muts, "the mutation set"):
        problems.append(f"{name}: {msg}")

# ---- identity fields ---------------------------------------------------
print("\n=== identity ===")
tex = DOCS["report (main.tex)"] or ""
for need, what in [("23035010370", "roll number"),
                   ("g.vyas@op.iitg.ac.in", "email"),
                   ("Gaurav Vyas", "name"),
                   ("Trimester 9", "trimester"),
                   ("Term Project Report", "report type")]:
    ok = need in tex
    print(f"  {what:14s} {'ok' if ok else 'MISSING'}")
    if not ok:
        problems.append(f"report missing {what}")

script = DOCS["video script"] or ""
if "Trimester 9" not in script:
    problems.append("video script does not say Trimester 9")

# ---- placeholders still to fill ---------------------------------------
print("\n=== placeholders ===")
for name, text in DOCS.items():
    if not text:
        continue
    for ph in re.findall(r"\[URL to be added\]", text, re.I):
        notes.append(f"{name}: a URL placeholder is still unfilled")
seen = set()
for n in notes:
    if n not in seen:
        print(f"  {n}")
        seen.add(n)
if not notes:
    print("  none")

# ---- report page budget ------------------------------------------------
log = read("docs/ieee_report/main.log")
if log:
    pages = re.findall(r"Output written on main\.pdf \((\d+) pages", log)
    if pages:
        n = int(pages[-1])
        print(f"\n=== report is {n} pages (want 5: 4 content + 1 references) ===")
        if n != 5:
            problems.append(f"report is {n} pages, not 5")

# ---- slide count -------------------------------------------------------
p = ROOT / "docs/chess_ai_slides.pptx"
if p.exists():
    with zipfile.ZipFile(p) as z:
        n = len([x for x in z.namelist()
                 if re.match(r"ppt/slides/slide\d+\.xml$", x)])
    print(f"=== deck is {n} slides (want 9, under 10 with a title slide) ===")
    if n > 9:
        problems.append(f"deck has {n} slides, more than 9")

print()
if problems:
    print(f"{len(problems)} INCONSISTENCY(IES):")
    for pr in problems:
        print(f"  - {pr}")
else:
    print("all deliverables agree")
sys.exit(1 if problems else 0)
