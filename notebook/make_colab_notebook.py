"""Build a Google-Colab-runnable copy of Crime_Analysis.ipynb.

The analysis is identical to the local notebook; only the environment setup is
adapted for Colab:
  * a cell that pip-installs MiniSom (the one dependency Colab lacks),
  * a cell that downloads communities.data / communities.names from UCI
    (with a manual-upload fallback),
  * the working directory (ROOT) is pointed at the Colab session dir.
All analysis cells are copied verbatim and their outputs are cleared so the
notebook runs top-to-bottom on a fresh Colab runtime.
"""
import os
import copy
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(ROOT, "notebook", "Crime_Analysis.ipynb")
out_path = os.path.join(ROOT, "notebook", "Crime_Analysis_Colab.ipynb")

nb = nbf.read(src_path, as_version=4)

# 1. Repoint ROOT at the Colab working directory (the local notebook used the
#    parent of the notebook's CWD).
LOCAL_ROOT = "ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))"
COLAB_ROOT = "ROOT = os.getcwd()  # Colab session dir (e.g. /content)"
patched = 0
for cell in nb.cells:
    if cell.cell_type == "code" and LOCAL_ROOT in cell.source:
        cell.source = cell.source.replace(LOCAL_ROOT, COLAB_ROOT)
        patched += 1
assert patched == 1, f"expected exactly one ROOT line to patch, found {patched}"

# 2. Clear all outputs / execution counts for a clean runnable notebook.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

# 3. Build the Colab setup cells.
title_cell = nbf.v4.new_markdown_cell(
    "# CSE 555: Statistical Data Analysis — Final Project (Google Colab)\n\n"
    "## Communities and Crime Analysis\n\n"
    "**Author:** Serkan Eren  \n"
    "This is the Colab-runnable copy of `Crime_Analysis.ipynb`. The analysis is "
    "identical to the local notebook; only the setup (package install, dataset "
    "download, working directory) is adapted for Colab. Run the cells in order, "
    "top to bottom. All figures are written to `figures/` in the Colab session."
)

install_md = nbf.v4.new_markdown_cell(
    "## Setup 1 — install packages\n"
    "Colab already ships NumPy, pandas, Matplotlib, seaborn, SciPy and "
    "scikit-learn; only MiniSom (the Self-Organizing Map) has to be installed."
)
install_code = nbf.v4.new_code_cell("!pip install minisom -q")

download_md = nbf.v4.new_markdown_cell(
    "## Setup 2 — download the dataset\n"
    "The two raw files (`communities.data`, `communities.names`) are fetched "
    "from the UCI Machine Learning Repository into the session directory. If the "
    "automatic download fails, a manual upload prompt is shown instead."
)
download_code = nbf.v4.new_code_cell(
    "import os, urllib.request\n"
    "\n"
    "BASE = \"https://archive.ics.uci.edu/ml/machine-learning-databases/communities/\"\n"
    "FILES = [\"communities.data\", \"communities.names\"]\n"
    "\n"
    "def _have_all():\n"
    "    return all(os.path.exists(f) for f in FILES)\n"
    "\n"
    "if not _have_all():\n"
    "    try:\n"
    "        for f in FILES:\n"
    "            urllib.request.urlretrieve(BASE + f, f)\n"
    "            print(\"downloaded\", f)\n"
    "    except Exception as e:\n"
    "        print(\"Automatic download failed:\", e)\n"
    "        print(\"Please upload communities.data and communities.names below.\")\n"
    "        from google.colab import files as _colab_files\n"
    "        _colab_files.upload()\n"
    "\n"
    "assert _have_all(), \"communities.data / communities.names are missing.\"\n"
    "print(\"Dataset ready in\", os.getcwd())"
)

# 4. Replace the original title cell (first markdown cell) with the Colab title,
#    then prepend the setup cells before the rest of the notebook body.
body = nb.cells
if body and body[0].cell_type == "markdown":
    body = body[1:]  # drop the original local title cell
nb.cells = [title_cell, install_md, install_code,
            download_md, download_code] + body

nbf.write(nb, out_path)
print(f"Wrote {out_path} with {len(nb.cells)} cells.")
