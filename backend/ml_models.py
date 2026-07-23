# -*- coding: utf-8 -*-
"""
ml_models.py — Chameleon Hybrid Personalization Engine

WHAT CHANGED FROM THE ORIGINAL NOTEBOOK AND WHY
------------------------------------------------
1. Context (region/device/country/source/medium) is now encoded with a
   *saved* OneHotEncoder (`context_encoder.joblib`) that is fit once at
   train time and reused identically at inference time. The original code
   used `pd.get_dummies` at train time and then silently dropped all
   context columns at inference time (only 4 numeric fields survived into
   `feature_cols.json`). That mismatch meant every cold-start user produced
   an identical, context-blind feature vector — the model had no way to
   tell a mobile visitor from Delhi apart from a desktop visitor from
   Mumbai. This is now fixed: behavioral models are trained AND queried on
   [numeric_features + context_one_hot + cluster_id], consistently.

2. A dedicated `ColdStartResolver` is added. The RF/Deep/LSTM stack needs
   behavioral aggregates (total_events, total_revenue, ...) that a genuinely
   new/anonymous visitor does not have — feeding it zeros just collapses
   every new visitor into the same bucket. The resolver instead does
   hierarchical demographic/contextual matching (the "KNN / demographic
   filtering / regional trend" fallback the brief explicitly asks for),
   using only signals available on a visitor's very first pageview.
"""

import io
import json
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_models")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Context columns available even for a brand-new, anonymous visitor.
# 'category' is device category (mobile/desktop/tablet) in the raw data.
CONTEXT_COLS = ["category", "region", "country", "source", "medium", "Age"]

# A known/returning user needs at least this many logged events before we
# trust the behavioral models over the cold-start resolver.
MIN_EVENTS_FOR_BEHAVIORAL = 3

# total_events / total_revenue / unique_pages are almost always heavily
# right-skewed in e-commerce data (most users near 0, a long tail of high
# spenders/browsers). KMeans uses Euclidean distance on the scaled values —
# StandardScaler alone doesn't fix skew, so without a log transform the
# clustering tends to collapse into one dominant "everyone near zero"
# cluster plus a tiny outlier cluster. log1p compresses the tail so
# clustering actually separates users meaningfully.
LOG_TRANSFORM_COLS = ["total_events", "unique_pages", "total_revenue"]


def apply_numeric_transform(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    """Log1p-transforms skewed count/revenue columns, leaves others as-is."""
    out = np.zeros((len(df), len(numeric_cols)), dtype=np.float64)
    for i, col in enumerate(numeric_cols):
        vals = df[col].values.astype(np.float64)
        if col in LOG_TRANSFORM_COLS:
            vals = np.log1p(np.clip(vals, a_min=0, a_max=None))
        out[:, i] = vals
    return out


# ---------------------------------------------------------------------------
# PyTorch datasets & models
# ---------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, pad_value=0, max_len=25):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx][: self.max_len], dtype=torch.long)
        if len(seq) == 0:
            padded_seq = torch.full((self.max_len,), self.pad_value, dtype=torch.long)
            seq_len = torch.tensor(1, dtype=torch.long)
        else:
            padded_seq = torch.nn.functional.pad(
                seq, (0, self.max_len - len(seq)), "constant", self.pad_value
            )
            seq_len = torch.tensor(len(seq), dtype=torch.long)
        return padded_seq, seq_len, self.labels[idx]


class DeepPredictor(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, max(hidden_size // 2, 8))
        self.bn2 = nn.BatchNorm1d(max(hidden_size // 2, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(max(hidden_size // 2, 8), num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        return self.out(x)


class UserSequenceLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=64, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, min_cluster_frac: float = 0.01, min_cluster_abs: int = 20
) -> np.ndarray:
    """
    KMeans sometimes carves off a handful of extreme outliers (e.g. one or
    two enormous single purchases) into their own 2-6 user "cluster". A
    cluster that small can't teach a classifier anything reliable and just
    adds label noise, so we reassign its members to the nearest surviving
    (large-enough) cluster centroid instead of treating it as a real segment.
    """
    labels = labels.copy()
    n = len(labels)
    min_size = max(min_cluster_abs, int(n * min_cluster_frac))

    counts = pd.Series(labels).value_counts()
    small_clusters = counts[counts < min_size].index.tolist()
    large_clusters = counts[counts >= min_size].index.tolist()

    if not small_clusters or not large_clusters:
        return labels  # nothing to merge, or everything is small (leave as-is)

    large_centroids = {c: X[labels == c].mean(axis=0) for c in large_clusters}

    for c in small_clusters:
        idx = np.where(labels == c)[0]
        for i in idx:
            dists = {lc: np.linalg.norm(X[i] - cen) for lc, cen in large_centroids.items()}
            labels[i] = min(dists, key=dists.get)

    logger.info(
        f"merge_small_clusters: reassigned {sum(counts[c] for c in small_clusters)} users "
        f"from {len(small_clusters)} tiny cluster(s) {small_clusters} into nearest large cluster(s)."
    )
    return labels


class SelectiveScaler:
    """
    Standard-scales only the specified columns of a matrix and passes the
    rest through unchanged. Used whenever a feature matrix mixes continuous
    numeric features with one-hot context/dummy columns: StandardScaler on
    a rare, mostly-zero dummy column inflates its contribution to Euclidean
    distance disproportionately (its variance is tiny, so scaling to unit
    variance blows up its coefficient), which destabilizes downstream
    KMeans clustering. Binary 0/1 columns don't need (and shouldn't get)
    that treatment.

    `scale_indices` can be either an int (treated as "the first N columns")
    or an explicit list of column indices (needed when a non-numeric block,
    e.g. one-hot context, sits between numeric feature blocks).
    """

    def __init__(self, scale_indices):
        self.scale_indices = list(range(scale_indices)) if isinstance(scale_indices, int) else list(scale_indices)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        if self.scale_indices:
            self.scaler.fit(X[:, self.scale_indices])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy().astype(np.float64)
        if self.scale_indices:
            X[:, self.scale_indices] = self.scaler.transform(X[:, self.scale_indices])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def train_kmeans(X_train: np.ndarray, numeric_dim: int = None, max_k: int = 6) -> Tuple[KMeans, SelectiveScaler]:
    logger.info("Training KMeans for feature extraction...")
    if numeric_dim is None:
        numeric_dim = X_train.shape[1]  # scale everything (old behavior) if caller doesn't specify
    scaler = SelectiveScaler(scale_indices=numeric_dim)
    Xs = scaler.fit_transform(X_train)

    best_k, best_score = 2, -1
    upper = min(max_k, max(3, len(X_train) - 1))
    for k in range(2, upper):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        try:
            score = silhouette_score(Xs, km.labels_, sample_size=min(10000, len(Xs)), random_state=SEED)
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score = k, score

    logger.info(f"Chosen k={best_k} (silhouette={best_score:.4f})")
    final_km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20).fit(Xs)
    return final_km, scaler


def train_deep_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    best_loss = float("inf")
    patience_counter = 0
    best_model_buffer = io.BytesIO()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_buffer.seek(0)
            best_model_buffer.truncate()
            torch.save(model.state_dict(), best_model_buffer)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Deep model early stopping at epoch {epoch + 1}")
                break

    best_model_buffer.seek(0)
    model.load_state_dict(torch.load(best_model_buffer))
    return model


def train_sequence_model(model, train_loader, val_loader, epochs=15, lr=0.001, patience=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_buffer = io.BytesIO()

    for epoch in range(epochs):
        model.train()
        for seq, lengths, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq, lengths), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, lengths, labels in val_loader:
                outputs = model(seq.to(device), lengths)
                val_loss += criterion(outputs, labels.to(device)).item() * labels.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_buffer.seek(0)
            best_model_buffer.truncate()
            torch.save(model.state_dict(), best_model_buffer)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                break

    best_model_buffer.seek(0)
    model.load_state_dict(torch.load(best_model_buffer))
    return model


# ---------------------------------------------------------------------------
# Cold-start resolver — hierarchical demographic/contextual fallback
# ---------------------------------------------------------------------------
class ColdStartResolver:
    """
    Answers "what should we show a visitor we've never seen before?" using
    ONLY signals available on a first pageview: device category, region,
    country, traffic source/medium, and declared age bracket if present.

    Strategy: precompute, from historical data, the dominant behavioral
    cluster and top purchased category for every combination of context
    columns, at multiple levels of specificity. At resolve time, back off
    from the most specific combination to the most general (global) one
    until a level with enough supporting users is found. This is the
    "similar users / regional trend / demographic filtering" fallback the
    brief calls for.
    """

    # Ordered most -> least specific. Each tuple is a subset of CONTEXT_COLS.
    # Broadened from the original 3-level hierarchy: real traffic-source data
    # is often sparse per exact region, so we now try several mid-specificity
    # combinations (source+medium, region+source, medium alone, source alone)
    # before falling back to a single context dimension or the global default.
    BACKOFF_LEVELS: List[Tuple[str, ...]] = [
        ("region", "category", "source"),
        ("region", "category"),
        ("region", "source"),
        ("category", "source"),
        ("source", "medium"),
        ("region",),
        ("category",),
        ("source",),
        ("medium",),
        (),  # global fallback
    ]

    def __init__(self, min_support: int = None):
        self.tables: Dict[Tuple[str, ...], pd.DataFrame] = {}
        self.global_top_category = "Featured Products"
        self.global_cluster = 0
        # Adaptive default: at least 10 users, but scaled to ~0.05% of the
        # dataset so a level isn't trusted on a statistically thin sample.
        # Overridable if you want stricter/looser matching.
        self._min_support_override = min_support

    def fit(self, user_features: pd.DataFrame, merged_df: pd.DataFrame):
        logger.info("Fitting ColdStartResolver lookup tables...")

        self.min_support = self._min_support_override or max(10, int(len(user_features) * 0.0005))
        logger.info(f"Using MIN_SUPPORT={self.min_support} for {len(user_features)} users.")

        # Global fallback first
        if "ItemCategory" in merged_df.columns and merged_df["ItemCategory"].notna().any():
            top = merged_df["ItemCategory"].mode()
            if not top.empty:
                self.global_top_category = str(top.iloc[0])
        if "target_cluster" in user_features.columns and not user_features.empty:
            self.global_cluster = int(user_features["target_cluster"].mode().iloc[0])

        # Attach each user's top purchased category for aggregation
        cat_by_user = None
        if "ItemCategory" in merged_df.columns:
            cat_by_user = (
                merged_df.dropna(subset=["ItemCategory"])
                .groupby("user_pseudo_id")["ItemCategory"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                .rename("top_category")
            )

        base = user_features.copy()
        if cat_by_user is not None:
            base = base.join(cat_by_user, on="user_pseudo_id")

        for level in self.BACKOFF_LEVELS:
            cols = list(level)
            present = [c for c in cols if c in base.columns]
            if len(present) != len(cols):
                continue

            if cols:
                grouped = base.groupby(cols, observed=True)
            else:
                # global level: single group
                grouped = [((), base)]

            rows = []
            for key, group in grouped if cols else grouped:
                if len(group) < self.min_support and cols:
                    continue
                top_cat = self.global_top_category
                if "top_category" in group.columns and group["top_category"].notna().any():
                    m = group["top_category"].mode()
                    if not m.empty:
                        top_cat = str(m.iloc[0])
                cluster = self.global_cluster
                if "target_cluster" in group.columns and not group.empty:
                    cluster = int(group["target_cluster"].mode().iloc[0])
                row = {
                    "support": len(group),
                    "top_category": top_cat,
                    "cluster": cluster,
                    "avg_revenue": float(group["total_revenue"].mean()) if "total_revenue" in group.columns else 0.0,
                }
                if cols:
                    key_tuple = key if isinstance(key, tuple) else (key,)
                    for c, v in zip(cols, key_tuple):
                        row[c] = v
                rows.append(row)

            if rows:
                self.tables[level] = pd.DataFrame(rows)
                logger.info(f"   level {level or '(global)'}: {len(rows)} groups above min_support")
            else:
                logger.info(f"   level {level or '(global)'}: no groups met min_support={self.min_support} -- will be skipped at resolve time")

        logger.info(f"ColdStartResolver ready with {len(self.tables)} backoff levels populated.")

    def resolve(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Backs off from most to least specific context match."""
        for level in self.BACKOFF_LEVELS:
            table = self.tables.get(level)
            if table is None or table.empty:
                continue

            if not level:
                row = table.iloc[0]
                return {
                    "cluster": int(row["cluster"]),
                    "top_category": row["top_category"],
                    "avg_revenue": float(row["avg_revenue"]),
                    "source": "cold_start_global_default",
                    "match_level": "global",
                }

            mask = pd.Series(True, index=table.index)
            ok = True
            for col in level:
                val = context.get(col, "unknown")
                mask &= table[col].astype(str) == str(val)
            matched = table[mask]
            if not matched.empty:
                row = matched.sort_values("support", ascending=False).iloc[0]
                return {
                    "cluster": int(row["cluster"]),
                    "top_category": row["top_category"],
                    "avg_revenue": float(row["avg_revenue"]),
                    "source": f"cold_start_lookup:{'+'.join(level)}",
                    "match_level": "+".join(level),
                }

        return {
            "cluster": self.global_cluster,
            "top_category": self.global_top_category,
            "avg_revenue": 0.0,
            "source": "cold_start_global_default",
            "match_level": "global",
        }


# ---------------------------------------------------------------------------
# High-level behavioral model orchestrator (for KNOWN / returning users)
# ---------------------------------------------------------------------------
class PersonalizationModel:
    def __init__(self, rf_threshold: float = 0.65, lstm_threshold: float = 0.70):
        self.rf = None
        self.kmeans = None
        self.kmeans_scaler = None
        self.feature_scaler = None
        self.deep = None
        self.sequence_model = None
        self.event2id = None
        self.numeric_feature_cols: List[str] = []
        self.context_cols: List[str] = [c for c in CONTEXT_COLS if c != "Age"] + ["Age"]
        self.context_encoder: Optional[OneHotEncoder] = None
        self.sequence_max_len = 25

        self.rf_threshold = rf_threshold
        self.lstm_threshold = lstm_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- feature assembly -----------------------------------------------
    def _build_context_matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        if fit:
            present_cols = [c for c in self.context_cols if c in df.columns]
            ctx_df = df[present_cols].astype(str).fillna("unknown")
            self.context_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.context_encoder.fit(ctx_df)
            return self.context_encoder.transform(ctx_df)

        # Inference: align to EXACTLY the columns the encoder was fit on. The
        # training frame may have carried fewer context columns than a live
        # request supplies (the API always sends all of them), and handing the
        # encoder a different column count raises. Reindex to feature_names_in_,
        # filling any column the caller didn't provide with "unknown".
        fit_cols = list(getattr(self.context_encoder, "feature_names_in_", self.context_cols))
        ctx_df = pd.DataFrame(index=df.index)
        for c in fit_cols:
            ctx_df[c] = df[c].astype(str).fillna("unknown") if c in df.columns else "unknown"
        return self.context_encoder.transform(ctx_df[fit_cols])

    def _build_full_matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        numeric = apply_numeric_transform(df, self.numeric_feature_cols)
        context = self._build_context_matrix(df, fit=fit)
        return np.hstack([numeric, context])

    # -- training ----------------------------------------------------------
    def train_all(
        self,
        df: pd.DataFrame,
        target_col: str,
        merged_df: pd.DataFrame,
        numeric_feature_cols: List[str],
        eval_sample_size: int = 500,
    ):
        logger.info("Starting behavioral model training pipeline (known users only)...")
        self.numeric_feature_cols = numeric_feature_cols
        self.eval_metrics = {}

        df = df.reset_index(drop=True)
        X_full = self._build_full_matrix(df, fit=True)
        y = df[target_col].astype(int)

        idx_all = np.arange(len(df))
        idx_train, idx_val = train_test_split(
            idx_all, test_size=0.2, random_state=SEED, stratify=y if y.nunique() > 1 else None
        )
        X_train, X_val = X_full[idx_train], X_full[idx_val]
        y_train, y_val = y.iloc[idx_train].reset_index(drop=True), y.iloc[idx_val].reset_index(drop=True)
        df_val = df.iloc[idx_val].reset_index(drop=True)  # kept for end-to-end cascade evaluation below

        self.kmeans, self.kmeans_scaler = train_kmeans(X_train, numeric_dim=len(self.numeric_feature_cols))
        train_clusters = self.kmeans.predict(self.kmeans_scaler.transform(X_train))
        val_clusters = self.kmeans.predict(self.kmeans_scaler.transform(X_val))

        X_train_aug = np.column_stack((X_train, train_clusters))
        X_val_aug = np.column_stack((X_val, val_clusters))

        # Layout is [numeric | one-hot context | cluster_id]. Scale the
        # numeric block and the cluster_id column; leave the one-hot context
        # block untouched (see SelectiveScaler docstring for why).
        scale_idx = list(range(len(self.numeric_feature_cols))) + [X_train_aug.shape[1] - 1]
        self.feature_scaler = SelectiveScaler(scale_indices=scale_idx)
        X_train_scaled = self.feature_scaler.fit_transform(X_train_aug)
        X_val_scaled = self.feature_scaler.transform(X_val_aug)

        logger.info("Training Random Forest...")
        rf_base = RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=200, max_depth=12)
        min_class_count = pd.Series(y_train).value_counts().min()
        if min_class_count < 2:
            logger.warning(f"Smallest class has {min_class_count} sample(s); using uncalibrated RF.")
            self.rf = rf_base.fit(X_train_scaled, y_train)
        else:
            cv_folds = min(3, min_class_count)
            self.rf = CalibratedClassifierCV(rf_base, cv=cv_folds).fit(X_train_scaled, y_train)

        logger.info("Training Deep Predictor...")
        train_ds = TabularDataset(X_train_scaled, y_train.values)
        val_ds = TabularDataset(X_val_scaled, y_val.values)
        self.deep = DeepPredictor(input_size=X_train_scaled.shape[1], num_classes=y.nunique())
        # BatchNorm1d raises in train mode on a batch of size 1. Drop the final
        # batch only when it would be a lone sample, so a full-batch training
        # set is never emptied.
        drop_last = (len(train_ds) % 64 == 1)
        self.deep = train_deep_model(
            self.deep,
            DataLoader(train_ds, 64, shuffle=True, drop_last=drop_last),
            DataLoader(val_ds, 64),
        )

        logger.info("Training Sequence Model (LSTM)...")
        self.event2id = {
            event: i + 1 for i, event in enumerate(sorted(merged_df["event_name"].dropna().unique().tolist()))
        }
        self.event2id["<PAD>"] = 0
        merged_df = merged_df.copy()
        merged_df["event_id"] = merged_df["event_name"].astype(object).map(self.event2id).fillna(0).astype(int)

        time_col = "eventTimestamp" if "eventTimestamp" in merged_df.columns else "timestamp"
        user_sequences = (
            merged_df.sort_values(time_col).groupby("user_pseudo_id")["event_id"].agg(list).rename("sequence")
        )
        user_event_names = (
            merged_df.sort_values(time_col).groupby("user_pseudo_id")["event_name"].agg(list).rename("event_names")
        )
        aligned = df.join(user_sequences, on="user_pseudo_id").dropna(subset=["sequence", target_col])

        if len(aligned) >= 20 and aligned[target_col].nunique() > 1:
            seqs = aligned["sequence"].tolist()
            labels = aligned[target_col].tolist()
            seq_train, seq_val, lab_train, lab_val = train_test_split(seqs, labels, test_size=0.2, random_state=SEED)

            train_loader = DataLoader(
                SequenceDataset(seq_train, lab_train, max_len=self.sequence_max_len), batch_size=64, shuffle=True
            )
            val_loader = DataLoader(SequenceDataset(seq_val, lab_val, max_len=self.sequence_max_len), batch_size=64)

            self.sequence_model = UserSequenceLSTM(len(self.event2id), y.nunique(), pad_idx=0)
            self.sequence_model = train_sequence_model(self.sequence_model, train_loader, val_loader)
        else:
            logger.warning("Not enough aligned sequence data to train LSTM reliably; skipping.")
            self.sequence_model = None
            seq_val, lab_val = [], []

        self._evaluate(
            X_val_scaled, y_val, y_train, seq_val, lab_val, df_val, user_event_names, eval_sample_size, target_col
        )

    # -- evaluation ----------------------------------------------------------
    def _evaluate(self, X_val_scaled, y_val, y_train, seq_val, lab_val, df_val, user_event_names, eval_sample_size, target_col):
        """
        Real accuracy, not vibes. Reports:
          - majority_baseline: accuracy of always guessing the most common
            class -- context for whether the models are actually better
            than just guessing.
          - random_forest / deep_model / lstm: standalone validation
            accuracy + weighted F1 for each classifier in isolation.
          - deployed_pipeline: end-to-end accuracy of the ACTUAL decision
            cascade used at inference (Deep -> RF fallback -> LSTM
            override, with real thresholds), run through predict_for_user
            exactly as a live request would be -- this is the number that
            answers "how accurate is the model when it actually runs."
        """
        from sklearn.metrics import accuracy_score, f1_score

        logger.info("Evaluating trained models on held-out validation data...")
        metrics = {}

        majority_class = y_train.mode().iloc[0]
        metrics["majority_baseline_accuracy"] = float((y_val == majority_class).mean())

        if hasattr(self.rf, "predict"):
            rf_pred = self.rf.predict(X_val_scaled)
            metrics["random_forest"] = {
                "accuracy": float(accuracy_score(y_val, rf_pred)),
                "f1_weighted": float(f1_score(y_val, rf_pred, average="weighted")),
            }

        self.deep.eval()
        with torch.no_grad():
            deep_logits = self.deep(torch.from_numpy(X_val_scaled.astype(np.float32)).to(self.device))
            deep_pred = torch.argmax(deep_logits, dim=1).cpu().numpy()
        metrics["deep_model"] = {
            "accuracy": float(accuracy_score(y_val, deep_pred)),
            "f1_weighted": float(f1_score(y_val, deep_pred, average="weighted")),
        }

        if self.sequence_model is not None and len(seq_val) > 0:
            self.sequence_model.eval()
            loader = DataLoader(SequenceDataset(seq_val, lab_val, max_len=self.sequence_max_len), batch_size=64)
            all_preds, all_true = [], []
            with torch.no_grad():
                for seq, lengths, labels in loader:
                    logits = self.sequence_model(seq.to(self.device), lengths)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                    all_true.extend(labels.numpy().tolist())
            metrics["lstm"] = {
                "accuracy": float(accuracy_score(all_true, all_preds)),
                "f1_weighted": float(f1_score(all_true, all_preds, average="weighted")),
            }

        # End-to-end: run the real inference cascade (predict_for_user) on a
        # sample of validation users, exactly as a live API request would.
        sample_n = min(eval_sample_size, len(df_val))
        sample_df = df_val.sample(n=sample_n, random_state=SEED) if sample_n > 0 else df_val
        cascade_preds, cascade_true, source_counts = [], [], {}
        for _, row in sample_df.iterrows():
            numeric_features = {c: float(row[c]) for c in self.numeric_feature_cols}
            context = {c: str(row.get(c, "unknown")) for c in self.context_cols}
            user_sequence = None
            if row.get("user_pseudo_id") in user_event_names.index:
                user_sequence = user_event_names.loc[row["user_pseudo_id"]][-self.sequence_max_len:]
            pred_info = self.predict_for_user(numeric_features, context, user_sequence=user_sequence)
            cascade_preds.append(pred_info["cluster"])
            cascade_true.append(int(row[target_col]))
            source_counts[pred_info["source"]] = source_counts.get(pred_info["source"], 0) + 1

        if cascade_true:
            metrics["deployed_pipeline"] = {
                "accuracy": float(accuracy_score(cascade_true, cascade_preds)),
                "f1_weighted": float(f1_score(cascade_true, cascade_preds, average="weighted")),
                "n_evaluated": len(cascade_true),
                "prediction_source_breakdown": source_counts,
            }

        self.eval_metrics = metrics
        logger.info(f"Evaluation complete: {json.dumps(metrics, indent=2)}")

    # -- inference -----------------------------------------------------------
    def predict_for_user(
        self, numeric_features: Dict[str, float], context: Dict[str, Any], user_sequence: Optional[list] = None
    ) -> dict:
        if not all([self.feature_scaler, self.deep, self.rf, self.kmeans, self.context_encoder]):
            raise RuntimeError("Models are not trained. Call train_all() first.")

        row = {c: numeric_features.get(c, 0.0) for c in self.numeric_feature_cols}
        for c in self.context_cols:
            row[c] = context.get(c, "unknown")
        row_df = pd.DataFrame([row])

        x_vector = self._build_full_matrix(row_df, fit=False)
        cluster_feature = self.kmeans.predict(self.kmeans_scaler.transform(x_vector))[0]
        x_vector_aug = np.column_stack((x_vector, [cluster_feature]))
        x_scaled = self.feature_scaler.transform(x_vector_aug)

        self.deep.eval()
        self.deep.to(self.device)
        x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.deep(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            cluster = int(np.argmax(probs))
            confidence = float(np.max(probs))
            source = "deep_model"

        if confidence < self.rf_threshold:
            if hasattr(self.rf, "predict_proba"):
                rf_probs = self.rf.predict_proba(x_scaled).flatten()
                # argmax gives the COLUMN index into predict_proba, which maps
                # to self.rf.classes_[idx] -- NOT the class label itself. They
                # differ whenever cluster ids aren't a contiguous 0..k-1 range,
                # so translate through classes_ to recover the real cluster.
                best_idx = int(np.argmax(rf_probs))
                cluster = int(self.rf.classes_[best_idx])
                confidence = float(np.max(rf_probs))
            else:
                cluster = int(self.rf.predict(x_scaled)[0])
                confidence = 0.5
            source = "random_forest_fallback"

        if self.sequence_model is not None and self.event2id and user_sequence:
            try:
                self.sequence_model.eval()
                self.sequence_model.to(self.device)
                seq_ids = [self.event2id.get(event, 0) for event in user_sequence]
                if seq_ids:
                    seq_tensor = torch.tensor([seq_ids[: self.sequence_max_len]], dtype=torch.long).to(self.device)
                    seq_len = torch.tensor([len(seq_ids[: self.sequence_max_len])], dtype=torch.long)
                    with torch.no_grad():
                        lstm_logits = self.sequence_model(seq_tensor, seq_len)
                        lstm_probs = torch.softmax(lstm_logits, dim=1).cpu().numpy().flatten()
                        lstm_pred = int(np.argmax(lstm_probs))
                        lstm_conf = float(np.max(lstm_probs))
                    if lstm_conf > self.lstm_threshold:
                        cluster, confidence, source = lstm_pred, lstm_conf, "lstm_override"
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        return {"cluster": cluster, "confidence": confidence, "source": source}