# CLAUDE.md

Notes for Claude Code sessions in this folder.

## What this is

A clone of the upstream Cellpose repo (https://github.com/MouseLand/cellpose), used for
learning. The goal is to understand how Cellpose segmentation works, apply it to my own
images, and compare it against other segmentation methods (classical thresholding/watershed,
StarDist, etc.) on equal terms.

About me: physicist and analytical chemist with a long background in fluorescence
microscopy, imaging cores and Python lab automation. No need to explain microscopy or
image-formation basics. Do explain the deep-learning side (architecture, training, what the
network actually predicts) when it comes up. I prefer thorough reference-style explanations
and pointers to the relevant source code over step-by-step hand-holding.

## Environment

- Windows, Python via Miniforge (conda).
- Check `conda env list` for the env that has cellpose installed and use that one. If none
  exists, ask before creating one.
- Ignore `environment.yml` in the repo root; it pins Python 3.8 and is stale. The README
  recommends Python 3.12.
- Prefer an editable install of this clone (`python -m pip install -e .`) so the code I am
  reading is the code that runs. Check `cellpose.__file__` / `cellpose.version` if unsure.
- Report whether torch sees a GPU (`torch.cuda.is_available()`) before long runs.

## Ground rules

- Do not modify anything under `cellpose/`, `tests/`, `docs/` or other upstream files unless
  I ask. I want `git pull` to stay clean.
- My own work goes in `my_work/`:
  - `my_work/scripts/` – Python scripts
  - `my_work/notebooks/` – Jupyter notebooks
  - `my_work/data/` – input images (do not commit)
  - `my_work/results/` – masks, metrics tables, figures (do not commit)
- `CLAUDE.md` and `my_work/` should be listed in `.git/info/exclude`, not `.gitignore`, so
  the upstream files stay untouched. Add them there if they are missing.
- Do not delete files without asking.

## Where things live in the package

- `cellpose/models.py` – `CellposeModel`, `eval()`; default model `cpsam_v2`.
  Main knobs: `flow_threshold`, `cellprob_threshold`, `diameter`, `niter`.
- `cellpose/dynamics.py` – flow-following that turns predicted flows into masks.
- `cellpose/metrics.py` – `average_precision`, `aggregated_jaccard_index`,
  `boundary_scores`. Use these for every method so comparisons share one yardstick.
- `cellpose/train.py` – fine-tuning; see `notebooks/train_Cellpose-SAM.ipynb`.
- `cellpose/vit.py` – ViT backbone (Cellpose-SAM / DINO models).
- `notebooks/test_Cellpose-SAM.ipynb` – runnable 2D/3D examples.
- `paper/cpsam/` – how the authors ran their own benchmarks.

## Conventions for comparisons

- Save masks as uint16 label images (`.tif`), 0 = background, one integer per object.
- For each run, record method, model/version, all parameters, and the input image name
  alongside the results (a small JSON or CSV next to the masks).
- Report AP at IoU 0.5 / 0.75 / 0.9 per image, not just a single averaged number.
- Be plain about results. Do not overstate differences between methods; note the sample
  size and where a result could be down to parameter choice rather than the method itself.
