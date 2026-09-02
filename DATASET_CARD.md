# ViHateEval — Dataset Card

## Dataset summary

- **Task:** three-way classification of Vietnamese comments as `Normal`,
  `Offensive`, or `Hate Speech`.
- **Domain:** Vietnamese short-video (TikTok) comment streams.
- **Size:** 16,505 comments from 93 source videos.
- **Context per comment:** the comment text, its video title, OCR text derived
  from the video thumbnail, and the (de-identified) thumbnail image. Multiple
  comments share the same source-video context.
- **Annotation:** three trained native Vietnamese annotators labeled every
  comment independently using title, OCR, and thumbnail context; majority vote
  with fourth-expert adjudication of full three-way disagreements (~5%);
  Fleiss' κ = 0.67 pre-adjudication (per-class: Normal 0.80, Offensive 0.525,
  Hate Speech 0.625).

## Fields

| Field        | Description                                            |
| ------------ | ------------------------------------------------------ |
| `id`         | Row identifier                                         |
| `video_title`| TikTok video title                                     |
| `thumbnail_text` | OCR text extracted from the thumbnail              |
| `comment_text`   | The comment being classified                       |
| `img`        | Thumbnail filename in `images_blurred/`                |
| `label`      | `Normal`, `Offensive`, or `Hate Speech`                |
| `split`      | `train`, `val`, or `test` (canonical video-disjoint)   |

## Split policy

- **Video-disjoint by construction:** all comments of one source video stay in
  a single partition, so a model never sees train-set video context at test
  time.
- **Canonical split:** `splits/video_disjoint/`.
- **Replicate splits:** `splits/video_disjoint_rep1..rep5` — five
  independently constructed video-disjoint partitions used for the
  split-robustness analysis. This split policy is itself a reported
  benchmark-design variable: comment-level partitioning inflates macro-F1 by
  up to ~0.22 relative to video-disjoint evaluation.

## Intended use

- Research on short-video toxicity detection, multimodal hate-speech
  detection, and context-sensitive comment filtering.
- Benchmarking and evaluation-design studies for moderation-support systems.
- Vietnamese and low-resource abusive-language research.

## Prohibited use

- Surveillance, targeting, or discriminatory moderation of individuals or
  groups.
- Re-identification of users (beyond the public TikTok identifiers already
  released).
- Use for any purpose inconsistent with the releasing institution's terms.

## De-identification steps

Thumbnails are blurred to obscure faces and direct identifiers (account
handles, contact information, URLs, other personally identifying text).
Usernames and profile IDs are removed from all records. Original unblurred
thumbnails are not distributed. Thumbnail OCR is released as automatically
extracted, uncorrected text; empty OCR fields are retained as empty.

## Known biases

- Collected from a single signed-in For You feed in 2025; content and topic
  coverage reflect that sampling.
- 93 source videos; comment counts per video are highly skewed.
- English translation of comments is not provided in the released data.
- Label decisions reflect the annotators' shared cultural context for
  locally salient groups.

## Takedown / correction

Requests to remove or correct a released comment or thumbnail will be honored:
contact the dataset maintainers (via the venue's review system and, after
publication, the DOI record).

## License

Author-created annotations, metadata, splits, and scripts: CC BY 4.0.
Underlying TikTok comment text and images remain rights-holder content and are
distributed only to the extent permitted by applicable law.