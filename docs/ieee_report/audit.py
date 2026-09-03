"""Audit the report for damage from the whitespace-insensitive trim passes.

Those passes matched with `\\s+` between tokens, which can also match a LaTeX
`\\\\` line break. That is how the address block lost "Project 3". This looks
for the same failure elsewhere: truncated sentences, missing line breaks in
blocks that need them, and dangling references.
"""
import pathlib
import re

TEX = pathlib.Path(__file__).with_name("main.tex")
t = TEX.read_text(encoding="utf-8")
body = t.split(r"\begin{document}")[1].split(r"\vfill")[0]

problems = []

# 1. paragraphs ending mid-sentence
paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
for p in paras:
    if p.startswith("%") or p.startswith("\\"):
        continue
    tail = p.rstrip()
    if re.search(r"[a-z,;]$", tail) and not tail.endswith(("\\\\",)):
        problems.append(("truncated paragraph", tail[-85:].replace("\n", " ")))

# 2. \address / title blocks must keep their \\ separators
addr = re.search(r"\\address\{(.*?)\n\}", t, re.S)
if addr:
    lines = [l for l in addr.group(1).strip().split("\n") if l.strip()]
    for l in lines[:-1]:
        if not l.rstrip().endswith(("\\\\", "\\\\[0.5em]")):
            problems.append(("address line missing \\\\", l.strip()))

# 3. every \ref should have a matching \label
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
for r in set(re.findall(r"\\ref\{([^}]+)\}", t)):
    if r not in labels:
        problems.append(("dangling \\ref", r))

# 4. every \cite key should exist in refs.bib
bib = pathlib.Path(__file__).with_name("refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
for c in re.findall(r"\\cite\{([^}]+)\}", t):
    for k in (x.strip() for x in c.split(",")):
        if k not in keys:
            problems.append(("missing bib key", k))

# 5. required identity fields, per the brief
for need, where in (("23035010370", "roll number"),
                    ("g.vyas@op.iitg.ac.in", "email"),
                    ("Gaurav Vyas", "name"),
                    ("Trimester 9", "trimester"),
                    ("Project 3", "project number")):
    if need not in t:
        problems.append(("MISSING REQUIRED FIELD", f"{where}: {need!r}"))

# 6. unbalanced braces / math delimiters
if body.count("$") % 2:
    problems.append(("odd number of $", "math delimiter mismatch"))

if problems:
    print(f"{len(problems)} issue(s):")
    for kind, detail in problems:
        print(f"  [{kind}] {detail}")
else:
    print("no issues found")
