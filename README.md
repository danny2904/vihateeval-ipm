# ViHateEval — Dataset and Reproduction Package

Vietnamese TikTok hate-speech benchmark with comment text, video titles,
thumbnail OCR text, and thumbnail images, plus an evaluation-design study of
shared source-video context in short-video toxicity benchmarks.

## 📦 Contents

- `data.csv` — full dataset: 16,505 comments from 93 videos, labeled
  `Normal`, `Offensive`, or `Hate Speech` (Fleiss' κ = 0.67), each paired with its
  video title, OCR thumbnail text, and thumbnail image.
- `images_blurred/` — the 93 thumbnail images with faces and direct
  identifiers blurred for privacy. **These are the only images distributed.**
  The original unblurred thumbnails are not included.
- `splits/` — the canonical video-disjoint split plus five independently
  constructed video-disjoint replicate splits used for the split-robustness
  analysis (`video_disjoint*`). All comments of a source video stay in one
  partition.
- `train_oc.py` — Out-of-Context branch: PhoBERT-base + ViT on comment text.
- `train_ic.py` — In-Context branch: PhoBERT-base + ViT on concatenated
  comment text, video title, and thumbnail OCR text.
- `train_c2tox.py` — C2Tox: freezes the OC and IC branches and trains a small
  router MLP to combine their logits with a learned per-sample weight.

The package is **reproduction-oriented**: three self-contained scripts with no
shared internal framework, intended to retrain each branch and verify the
reported results. Development-time ablations, sweeps, and analysis utilities
are deliberately omitted.

## 🔒 Privacy

Thumbnails have been processed to blur faces and direct identifiers such as
account handles, contact information, URLs, and other personally identifying
text. Non-identifying content text is retained when possible because thumbnail
OCR is part of the released context. Video IDs are TikTok public numeric
identifiers, not private user data. Usernames and profile IDs are removed from
all released records.

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for practical training time. On the full
dataset, each branch typically takes about 1–2 hours on a single 16 GB GPU.

## 🚀 Usage

Run the scripts in order; `train_c2tox.py` requires checkpoints from the first
two models.

```bash
python train_oc.py --data-dir . --image-dir images_blurred --output oc_model.pth --seed 42
python train_ic.py --data-dir . --image-dir images_blurred --output ic_model.pth --seed 42
python train_c2tox.py --data-dir . --oc-checkpoint oc_model.pth \
  --ic-checkpoint ic_model.pth --output c2tox_router.pth --seed 42
```

All three scripts share core hyperparameters confirmed to reproduce the
reported numbers: `lr=2e-5`, `batch_size=8`, `max_length=256`, `epochs=4`,
dropout `0.3`, a linear warmup schedule, gradient clipping with max norm
`1.0`, and mixed-precision training. Defaults can be overridden via CLI flags.

**Known training instability:** in roughly 1 in 3 runs the IC branch (and more
rarely the OC branch) can collapse to a degenerate single-class solution,
producing a much lower macro-F1 despite identical hyperparameters and seed.
This is a real, reproducible instability of the architecture/training setup,
not a bug in these scripts. A run that stays near a constant loss from epoch 1
has likely entered this failure mode; re-running usually resolves it.

## 📈 Expected results (seed 42)

| Model                   | Accuracy   | Macro-F1   |
| ----------------------- | ---------- | ---------- |
| OC (PhoBERT-base + ViT) | ~0.69–0.70 | ~0.68–0.69 |
| IC (PhoBERT-base + ViT) | ~0.69–0.70 | ~0.68–0.70 |
| C2Tox                   | ~0.70–0.71 | ~0.69–0.70 |

## 🗂️ Data format

`data.csv` columns:

| Column             | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| `id`               | Row identifier                                                      |
| `video_title`      | TikTok video title                                                  |
| `thumbnail_text`   | OCR text extracted from the video thumbnail                         |
| `comment_text`     | The comment being classified                                        |
| `img`              | Filename of the corresponding thumbnail in `images_blurred/`        |
| `label`            | One of `Normal`, `Offensive`, `Hate Speech`                         |
| `split`            | One of `train`, `val`, `test` (the canonical split)                 |

See `DATASET_CARD.md` for intended use, prohibition, de-identification steps,
known biases, and takedown process.

## 📄 License

Author-created annotations, metadata, splits, and scripts are licensed under
CC BY 4.0 (see `LICENSE`). Underlying TikTok comment text and images remain
rights-holder content and are distributed only to the extent permitted by
applicable law.