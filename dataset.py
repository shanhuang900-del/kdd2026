"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# Extra absolute-time integer features appended to ``user_int_feats``.
#
# Layout:
#   900001: time-of-day bucket, 1..8 for 3-hour windows
#   900002: week-part bucket, 1=Mon-Thu, 2=Fri, 3=Sat, 4=Sun
#   900003: user recency bucket, 0=no previous event, 1..7 by elapsed minutes
#   900004: month-part bucket, 1=first 10 days, 2=middle, 3=last 10 days
#   900005: is_weekend bit
#   900006: timepart bucket, 1=morning, 2=noon/afternoon, 3=night/early morning
#   900007: weekend-timepart bucket, 0=not weekend, 1/2/3=weekend timepart
#
# Timestamps are interpreted in the main business timezone. The dataset comes
# from a China advertising setting, so UTC+8 is a reasonable default and is
# intentionally isolated here for easy ablation.
TIME_FEATURE_TZ_OFFSET_SECONDS = 8 * 3600
NUM_ABS_TIME_INT_FEATURES = 7
ABS_TIME_INT_FID_START = 900001
ABS_TIME_BINARY_VOCAB_SIZE = 2
ABS_TIME_RECENCY_BOUNDARIES_MINUTES = np.array(
    [5, 30, 120, 720, 1440, 10080],
    dtype=np.float32,
)
ABS_TIME_INT_FEATURE_SPECS = [
    (900001, 9),  # ids 1..8
    (900002, 5),  # ids 1..4
    (900003, 8),  # ids 0..7
    (900004, 4),  # ids 1..3
    (900005, ABS_TIME_BINARY_VOCAB_SIZE),
    (900006, 4),  # ids 1..3
    (900007, 4),  # ids 0..3
]

# Synthetic item-int features derived from item_int_feats_16.
#
# FID16 is a high-signal item cluster feature in the competition EDA. We turn
# its training-set label statistics into two low-cardinality priors and append
# them to item_int so the existing item NS tokenizer can consume them without
# changing the number of NS tokens.
FID16_TE_SOURCE_FID = 16
FID16_CVR_BUCKET_FID = 930016
FID16_COUNT_BUCKET_FID = 930017
FID16_CVR_BUCKETS = 100
FID16_COUNT_BUCKET_BOUNDARIES = np.array(
    [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
    dtype=np.int64,
)
FID16_COUNT_BUCKETS = len(FID16_COUNT_BUCKET_BOUNDARIES) + 1
FID16_TE_STATS_FILENAME = 'fid16_te_stats.json'


def _list_parquet_files(path: str) -> List[str]:
    """Return sorted Parquet files from either a directory or one file path."""
    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {path}")
        return files
    return [path]


def _fid16_count_to_bucket(count: int) -> int:
    """Map a positive count to bucket ids 1..FID16_COUNT_BUCKETS.

    Bucket 0 is reserved for unknown / unseen fid16 values at inference.
    """
    return int(np.searchsorted(
        FID16_COUNT_BUCKET_BOUNDARIES,
        int(count),
        side='right',
    ) + 1)


def _fid16_cvr_to_bucket(cvr: float, cvr_buckets: int = FID16_CVR_BUCKETS) -> int:
    """Map a CVR prior to bucket ids 1..cvr_buckets.

    Bucket 0 is reserved for missing stats rather than a real zero-CVR value.
    """
    return int(np.clip(np.rint(float(cvr) * cvr_buckets), 1, cvr_buckets))


def build_fid16_te_stats(
    data_dir: str,
    output_path: str,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    smoothing: float = 50.0,
    source_fid: int = FID16_TE_SOURCE_FID,
    positive_label: int = 2,
    cvr_buckets: int = FID16_CVR_BUCKETS,
) -> str:
    """Build smoothed target-encoding statistics for ``item_int_feats_16``.

    Only the training Row Groups used by ``get_pcvr_data`` are scanned. This
    avoids leaking validation labels into the feature while keeping the split
    logic aligned with the actual training dataset.

    The emitted JSON contains raw ``count`` / ``pos_count`` /
    ``smoothed_cvr`` diagnostics plus the two bucket maps consumed at dataset
    conversion time.
    """
    source_col = f'item_int_feats_{source_fid}'
    label_col = 'label_type'
    files = _list_parquet_files(data_dir)

    rg_info = []
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)
    if total_rgs == 0:
        raise ValueError(f"No Row Groups found under {data_dir}")

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
    train_rg_info = rg_info[:n_train_rgs]
    if not train_rg_info:
        raise ValueError(
            f"No training Row Groups available for fid16 TE stats: "
            f"total_rgs={total_rgs}, valid_ratio={valid_ratio}, train_ratio={train_ratio}")

    counts: Dict[int, int] = {}
    pos_counts: Dict[int, int] = {}
    total_count = 0
    total_pos = 0

    logging.info(
        f"Building fid16 TE stats from {len(train_rg_info)} train Row Groups "
        f"into {output_path}")
    for file_path, rg_idx, _ in train_rg_info:
        pf = pq.ParquetFile(file_path)
        table = pf.read_row_group(rg_idx, columns=[source_col, label_col])
        vals = (
            table.column(source_col)
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        labels = (
            table.column(label_col)
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        valid = vals > 0
        if not valid.any():
            continue
        vals_valid = vals[valid]
        pos_valid = (labels[valid] == positive_label).astype(np.int64)
        uniq, inv = np.unique(vals_valid, return_inverse=True)
        cnt = np.bincount(inv).astype(np.int64)
        pos = np.bincount(inv, weights=pos_valid).astype(np.int64)

        total_count += int(cnt.sum())
        total_pos += int(pos.sum())
        for key, c, p in zip(uniq, cnt, pos):
            k = int(key)
            counts[k] = counts.get(k, 0) + int(c)
            pos_counts[k] = pos_counts.get(k, 0) + int(p)

    if total_count <= 0:
        raise ValueError(f"No positive fid16 values found while scanning {data_dir}")

    smoothing = max(float(smoothing), 0.0)
    global_cvr = float(total_pos / total_count)
    default_cvr_bucket = _fid16_cvr_to_bucket(global_cvr, cvr_buckets)

    value_stats: Dict[str, Dict[str, Any]] = {}
    value_to_cvr_bucket: Dict[str, int] = {}
    value_to_count_bucket: Dict[str, int] = {}
    for key in sorted(counts):
        c = int(counts[key])
        p = int(pos_counts.get(key, 0))
        smoothed_cvr = float((p + smoothing * global_cvr) / (c + smoothing))
        key_s = str(key)
        value_stats[key_s] = {
            'count': c,
            'pos_count': p,
            'smoothed_cvr': smoothed_cvr,
        }
        value_to_cvr_bucket[key_s] = _fid16_cvr_to_bucket(smoothed_cvr, cvr_buckets)
        value_to_count_bucket[key_s] = _fid16_count_to_bucket(c)

    stats = {
        'feature': source_col,
        'source_fid': source_fid,
        'cvr_bucket_fid': FID16_CVR_BUCKET_FID,
        'count_bucket_fid': FID16_COUNT_BUCKET_FID,
        'smoothing': smoothing,
        'positive_label': positive_label,
        'global_count': int(total_count),
        'global_pos_count': int(total_pos),
        'global_cvr': global_cvr,
        'default_cvr_bucket': default_cvr_bucket,
        'default_count_bucket': 0,
        'cvr_buckets': int(cvr_buckets),
        'count_buckets': FID16_COUNT_BUCKETS,
        'count_bucket_boundaries': FID16_COUNT_BUCKET_BOUNDARIES.astype(int).tolist(),
        'num_values': len(counts),
        'value_stats': value_stats,
        'value_to_cvr_bucket': value_to_cvr_bucket,
        'value_to_count_bucket': value_to_count_bucket,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False)

    logging.info(
        f"fid16 TE stats written: num_values={len(counts)}, "
        f"global_cvr={global_cvr:.6f}, default_cvr_bucket={default_cvr_bucket}")
    return output_path


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        fid16_te_stats_path: Optional[str] = None,
        use_fid16_te: bool = False,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            fid16_te_stats_path: JSON produced by ``build_fid16_te_stats``.
            use_fid16_te: append fid16 CVR/count bucket features to item_int.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.use_fid16_te = use_fid16_te
        self.fid16_te_stats_path = fid16_te_stats_path
        self._fid16_cvr_bucket_map: Dict[int, int] = {}
        self._fid16_count_bucket_map: Dict[int, int] = {}
        self._fid16_default_cvr_bucket = 0
        self._fid16_default_count_bucket = 0
        self._fid16_cvr_bucket_offset: Optional[int] = None
        self._fid16_count_bucket_offset: Optional[int] = None
        self._fid16_source_col_idx: Optional[int] = None
        self._load_fid16_te_stats(fid16_te_stats_path)
        self._last_user_event_ts: Dict[Any, int] = {}
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim
        self._abs_time_int_offset = offset

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim
        if self.use_fid16_te:
            self._fid16_source_col_idx = self._col_idx.get(
                f'item_int_feats_{FID16_TE_SOURCE_FID}')
            if self._fid16_source_col_idx is None:
                logging.warning(
                    f"item_int_feats_{FID16_TE_SOURCE_FID} not found; "
                    "fid16 TE features will stay at defaults")

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_fid16_te_stats(self, stats_path: Optional[str]) -> None:
        """Load bucket maps for the synthetic fid16 item-int features."""
        if not self.use_fid16_te:
            return
        if not stats_path:
            logging.warning(
                "use_fid16_te=True but no fid16_te_stats_path was provided; "
                "fid16 TE buckets will default to 0")
            return
        if not os.path.exists(stats_path):
            logging.warning(
                f"fid16_te_stats_path={stats_path!r} does not exist; "
                "fid16 TE buckets will default to 0")
            return

        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        self._fid16_cvr_bucket_map = {
            int(k): int(v) for k, v in stats.get('value_to_cvr_bucket', {}).items()
        }
        self._fid16_count_bucket_map = {
            int(k): int(v) for k, v in stats.get('value_to_count_bucket', {}).items()
        }
        self._fid16_default_cvr_bucket = int(stats.get('default_cvr_bucket', 0))
        self._fid16_default_count_bucket = int(stats.get('default_count_bucket', 0))
        logging.info(
            f"Loaded fid16 TE stats from {stats_path}: "
            f"{len(self._fid16_cvr_bucket_map)} values, "
            f"default_cvr_bucket={self._fid16_default_cvr_bucket}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)
        for fid, vs in ABS_TIME_INT_FEATURE_SPECS:
            self.user_int_schema.add(fid, 1)
            self.user_int_vocab_sizes.append(vs)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)
        if self.use_fid16_te:
            self._fid16_cvr_bucket_offset = self.item_int_schema.total_dim
            self.item_int_schema.add(FID16_CVR_BUCKET_FID, 1)
            self.item_int_vocab_sizes.append(FID16_CVR_BUCKETS)
            self._fid16_count_bucket_offset = self.item_int_schema.total_dim
            self.item_int_schema.add(FID16_COUNT_BUCKET_FID, 1)
            self.item_int_vocab_sizes.append(FID16_COUNT_BUCKETS)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        self._last_user_event_ts = {}
        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _build_user_recency_feature(
        self,
        timestamps: "npt.NDArray[np.int64]",
        user_ids: List[Any],
    ) -> "npt.NDArray[np.float32]":
        """Build ``log1p(minutes_since_last_user_event)`` in stream order.

        Rows with no prior event for the same user receive 0.0, matching the
        natural missing value for a log1p-transformed non-negative interval.
        """
        recency = np.zeros(timestamps.shape[0], dtype=np.float32)
        for i, (user_id, ts) in enumerate(zip(user_ids, timestamps)):
            if user_id is None:
                continue

            ts_int = int(ts)
            prev_ts = self._last_user_event_ts.get(user_id)
            if prev_ts is not None:
                delta_seconds = max(ts_int - prev_ts, 0)
                recency[i] = np.log1p(delta_seconds / 60.0)

            if prev_ts is None or ts_int >= prev_ts:
                self._last_user_event_ts[user_id] = ts_int

        return recency

    def _build_abs_time_int_features(
        self,
        timestamps: "npt.NDArray[np.int64]",
        user_ids: List[Any],
    ) -> "npt.NDArray[np.int64]":
        """Build absolute-time features from sample timestamps.

        The returned matrix is appended to ``user_int_feats`` and follows the
        layout described by ``NUM_ABS_TIME_INT_FEATURES``.
        """
        local_ts = timestamps + TIME_FEATURE_TZ_OFFSET_SECONDS
        seconds_per_day = 24 * 3600
        days = local_ts // seconds_per_day
        seconds_of_day = local_ts % seconds_per_day

        hour = seconds_of_day // 3600
        day_bucket = np.clip(hour // 3, 0, 7).astype(np.int64) + 1
        # Unix epoch 1970-01-01 is Thursday. This maps Monday=0 ... Sunday=6.
        day_of_week = ((days + 3) % 7).astype(np.int64)
        is_weekend = day_of_week >= 5

        dates = local_ts.astype('datetime64[s]').astype('datetime64[D]')
        month_starts = dates.astype('datetime64[M]').astype('datetime64[D]')
        day_of_month = (dates - month_starts).astype(np.int64) + 1
        days_in_month = (
            (dates.astype('datetime64[M]') + np.timedelta64(1, 'M')).astype('datetime64[D]')
            - month_starts
        ).astype(np.float32)

        feats = np.zeros((timestamps.shape[0], NUM_ABS_TIME_INT_FEATURES), dtype=np.int64)

        # 1. Time-of-day, split into 8 equal buckets.
        feats[:, 0] = day_bucket

        # 2. Week-part bucket: Mon-Thu, Friday, Saturday, Sunday.
        week_part = np.zeros_like(day_of_week)
        week_part[day_of_week == 4] = 1
        week_part[day_of_week == 5] = 2
        week_part[day_of_week == 6] = 3
        feats[:, 1] = week_part + 1

        # 3. User-level recency bucket in minutes.
        recency = self._build_user_recency_feature(timestamps, user_ids)
        recency_minutes = np.expm1(recency)
        has_previous_event = recency > 0
        feats[has_previous_event, 2] = (
            np.searchsorted(
                ABS_TIME_RECENCY_BOUNDARIES_MINUTES,
                recency_minutes[has_previous_event],
                side='right',
            ) + 1
        )

        # 4. Month-part bucket: first 10 days, middle, last 10 days.
        # Months with 31 days put day 11..21 in the middle bucket so every
        # calendar day receives a non-zero learnable bucket.
        days_in_month_int = days_in_month.astype(np.int64)
        feats[:, 3] = 2
        feats[day_of_month <= 10, 3] = 1
        feats[day_of_month > (days_in_month_int - 10), 3] = 3

        # 5. Weekend bit.
        feats[:, 4] = is_weekend.astype(np.int64)

        # 6. Coarse timepart bucket.
        timepart = np.full_like(hour, 3, dtype=np.int64)
        timepart[(hour >= 5) & (hour < 12)] = 1
        timepart[(hour >= 12) & (hour < 18)] = 2
        feats[:, 5] = timepart

        # 7. Weekend x timepart bucket. Non-weekend remains 0.
        feats[is_weekend, 6] = timepart[is_weekend]

        return feats

    def _fill_fid16_te_features(
        self,
        batch: "pa.RecordBatch",
        item_int: "npt.NDArray[np.int64]",
        B: int,
    ) -> None:
        """Fill synthetic fid16 CVR/count buckets into the item_int buffer."""
        if (
            not self.use_fid16_te
            or self._fid16_cvr_bucket_offset is None
            or self._fid16_count_bucket_offset is None
        ):
            return

        cvr_bucket = np.full(
            B, self._fid16_default_cvr_bucket, dtype=np.int64)
        count_bucket = np.full(
            B, self._fid16_default_count_bucket, dtype=np.int64)

        if self._fid16_source_col_idx is not None:
            vals = (
                batch.column(self._fid16_source_col_idx)
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )
            for i, value in enumerate(vals):
                key = int(value)
                if key <= 0:
                    continue
                cvr_bucket[i] = self._fid16_cvr_bucket_map.get(
                    key, self._fid16_default_cvr_bucket)
                count_bucket[i] = self._fid16_count_bucket_map.get(
                    key, self._fid16_default_count_bucket)

        item_int[:, self._fid16_cvr_bucket_offset] = cvr_bucket
        item_int[:, self._fid16_count_bucket_offset] = count_bucket

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded
        abs_time_feats = self._build_abs_time_int_features(timestamps, user_ids)
        time_offset = self._abs_time_int_offset
        user_int[:, time_offset:time_offset + NUM_ABS_TIME_INT_FEATURES] = abs_time_feats

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded
        self._fill_fid16_te_features(batch, item_int, B)

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        global_latest_seq_ts = np.zeros(B, dtype=np.int64)
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                valid_ts = (ts_padded > 0) & (ts_padded <= ts_expanded)
                latest_ts = np.where(valid_ts, ts_padded, 0).max(axis=1)
                global_latest_seq_ts = np.maximum(global_latest_seq_ts, latest_ts)

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        global_last_seq_time_bucket = np.zeros(B, dtype=np.int64)
        has_global_latest = global_latest_seq_ts > 0
        if has_global_latest.any():
            global_time_diff = np.maximum(timestamps - global_latest_seq_ts, 0)
            raw_global_buckets = np.clip(
                np.searchsorted(BUCKET_BOUNDARIES, global_time_diff),
                0, len(BUCKET_BOUNDARIES) - 1,
            )
            global_last_seq_time_bucket[has_global_latest] = (
                raw_global_buckets[has_global_latest] + 1
            )
        result['global_last_seq_time_bucket'] = torch.from_numpy(
            global_last_seq_time_bucket)

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    fid16_te_stats_path: Optional[str] = None,
    use_fid16_te: bool = True,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        fid16_te_stats_path=fid16_te_stats_path,
        use_fid16_te=use_fid16_te,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        fid16_te_stats_path=fid16_te_stats_path,
        use_fid16_te=use_fid16_te,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
