"""Convert analysis.py into a Jupyter notebook with section headers per task."""
import json
import os
import re
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(ROOT, "notebook", "analysis.py")
out_path = os.path.join(ROOT, "notebook", "Crime_Analysis.ipynb")

with open(src_path) as fh:
    src = fh.read()

# In a notebook, __file__ is undefined. Use the parent of the notebook's CWD.
src = src.replace(
    'ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
    'ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))'
)

# Split on the "# ----..." section banners
banner_re = re.compile(r"^# -{60,}\n# (.*?)\n# -{60,}\n", re.MULTILINE)
parts = banner_re.split(src)
# parts[0] is the preamble before any banner; then [title, body, title, body, ...]

nb = nbf.v4.new_notebook()
nb.cells = []

nb.cells.append(nbf.v4.new_markdown_cell(
    "# CSE 555 Final Project — Communities and Crime Analysis\n\n"
    "**Author:** Serkan Eren  \n"
    "**Dataset:** UCI Communities and Crime  \n"
    "**Target:** `ViolentCrimesPerPop` discretised into 3 classes (Low / Medium / High) by tertiles.\n\n"
    "All figures are written to `../figures/`."
))

# Preamble code cell (imports + setup, before first banner)
nb.cells.append(nbf.v4.new_code_cell(parts[0].rstrip() + "\n"))

eda_code = (
    "print('Shape:', df.shape)\n"
    "print('\\nClass counts:')\n"
    "print(df['CrimeClass'].value_counts().sort_index())\n"
    "print('\\nHead:')\n"
    "display(df[selected + ['CrimeClass']].head())\n"
    "print('\\nDescribe:')\n"
    "display(df[selected].describe().round(3))\n"
)

i = 1
while i < len(parts):
    title = parts[i].strip()
    body = parts[i + 1] if i + 1 < len(parts) else ""
    nb.cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
    nb.cells.append(nbf.v4.new_code_cell(body.rstrip() + "\n"))
    # Insert EDA preview right after the discretisation section runs (df + CrimeClass ready)
    if title.startswith("Discretize"):
        nb.cells.append(nbf.v4.new_markdown_cell("## Quick EDA preview"))
        nb.cells.append(nbf.v4.new_code_cell(eda_code))
    i += 2

nbf.write(nb, out_path)
print(f"Wrote {out_path} with {len(nb.cells)} cells.")
