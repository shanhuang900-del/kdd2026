# kdd2026
<img width="1423" height="625" alt="image" src="https://github.com/user-attachments/assets/2dd824c8-ff05-49f4-96e7-10236a359dde" />
# TAAC 2026 Tencent Ads CVR Prediction

This repository contains a PyTorch solution for the TAAC 2026 / KDD Cup Tencent advertising conversion-rate prediction task. The model is built for anonymized industrial ad logs with user features, item features, dense embeddings, and multi-domain user behavior sequences.

The main goal of this version is to improve test-set AUC through targeted EDA and feature engineering while keeping the training and inference pipeline aligned for online evaluation.

## Highlights

- Built a CVR prediction pipeline for the official flat Parquet dataset schema.
- Added 7 synthetic user-side sparse features from timestamp and user behavior recency.
- Added 1 global sequence recency feature before the final prediction head.
- Added optional `item_int_feats_16` smoothed CVR prior buckets as item-side sparse features.
- Used RankMixer-style user/item non-sequential tokenization with multi-domain sequence modeling.
- Kept checkpoint sidecar files aligned with inference, including schema, training config, NS grouping reference, and fid16 statistics.

## Feature Engineering

### User Int Synthetic Features

The dataset appends seven synthetic sparse features to `user_int_feats`:

| Feature ID | Meaning | Construction |
| --- | --- | --- |
| `900001` | Time-of-day bucket | UTC+8 local hour, split into 8 buckets by 3-hour windows |
| `900002` | Week-part bucket | Mon-Thu, Friday, Saturday, Sunday |
| `900003` | User recency bucket | Stream-order time since the same user's previous sample, bucketed by minutes |
| `900004` | Month-part bucket | First 10 days, middle days, last 10 days |
| `900005` | Weekend flag | Saturday or Sunday |
| `900006` | Coarse timepart | Morning, afternoon, night/early morning |
| `900007` | Weekend x timepart | Non-weekend = 0; weekend uses the coarse timepart bucket |

These features are consumed as sparse user-side inputs through the existing user non-sequential tokenizer, so the model can learn time-sensitive conversion patterns without changing the raw data schema.

### Global Sequence Recency Feature

For each sample, the data pipeline scans the timestamp sequences from all four behavior domains and finds the most recent valid historical sequence event before the current impression timestamp. The time gap is bucketed and passed as `global_last_seq_time_bucket`.

Unlike per-token sequence time buckets, this feature is injected immediately before the final classifier through an embedding and fusion layer. This keeps the "how recently did the user have any historical activity" signal visible to the prediction head instead of letting it be diluted by sequence attention.

### Fid16 Item Prior

`item_int_feats_16` is treated as a high-signal item cluster feature. During training, the pipeline can compute smoothed CVR statistics by fid16 value and convert them into two low-cardinality item-side sparse buckets:

- `930016`: fid16 smoothed CVR bucket
- `930017`: fid16 count bucket

The statistics are written to `fid16_te_stats.json` and copied with checkpoints so inference can reproduce the same features.

## Model

The model is a PCVR HyFormer-style architecture:

- sparse user/item features are embedded into non-sequential tokens;
- dense user features are projected into a dense token;
- four sequence domains are embedded independently;
- query tokens attend to behavior sequences through stacked HyFormer blocks;
- DIN-style target-aware pooling fuses item-aware sequence interests;
- the global sequence recency bucket is fused before the final classifier.

The active training configuration in `run.sh` uses:

- `ns_tokenizer_type=rankmixer`
- `user_ns_tokens=3`
- `item_ns_tokens=4`
- `num_queries=2`
- epoch-end validation (`eval_every_n_steps=0`)
- cosine learning-rate schedule with warmup

## Repository Structure

```text
.
├── dataset.py       # Parquet dataset, schema parsing, feature engineering
├── model.py         # PCVRHyFormer model and sequence/user/item token modules
├── train.py         # Training entrypoint and model construction
├── trainer.py       # Training loop, validation AUC, checkpoint sidecars
├── infer.py         # Evaluation/inference entrypoint for online submission
├── run.sh           # Active training command
├── ns_groups.json   # Reference NS grouping config for group tokenizer experiments
└── utils.py         # Logging, seeding, early stopping, helper utilities
```

## Training

Set the training data and output paths through the environment variables expected by the competition runtime, then run:

```bash
bash run.sh
```

The trainer saves model checkpoints together with the files needed for reproducible inference:

- `model.pt`
- `schema.json`
- `train_config.json`
- `fid16_te_stats.json` when fid16 target encoding is enabled
- `ns_groups.json` when a grouping config is used

## Inference

The inference script mirrors the training-time model construction. In the competition evaluation container, it expects:

- `MODEL_OUTPUT_PATH`: checkpoint directory containing `model.pt`
- `EVAL_DATA_PATH`: test Parquet data directory
- `EVAL_RESULT_PATH`: output directory for `predictions.json`

```bash
python infer.py
```

## Result

With the above EDA-driven feature engineering, this version improved the test-set AUC to the 0.82x range in online evaluation. The largest gains came from adding user-side time/recency sparse features and injecting the global sequence recency feature directly before the prediction head.

## Notes

The official dataset is anonymized and is not included in this repository. To run the project, download the TAAC 2026 dataset from the official competition source and place it in the expected runtime data directory.
