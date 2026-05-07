
"""
Refactored ml_models.py for the "Chameleon" storefront.
Includes fixes for data leakage, small-cluster CV crashes, and Categorical type strictness.
"""

import os
import io
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------- Configuration & logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_models_improved")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- PyTorch Models & Datasets ----------
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
        seq = torch.tensor(self.sequences[idx][:self.max_len], dtype=torch.long)
        seq_len = torch.tensor(len(seq), dtype=torch.long)

        if len(seq) == 0:
            padded_seq = torch.full((self.max_len,), self.pad_value, dtype=torch.long)
            seq_len = torch.tensor(1, dtype=torch.long)
        else:
            padded_seq = torch.nn.functional.pad(seq, (0, self.max_len - len(seq)), 'constant', self.pad_value)

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
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])


# ---------- Model Training Functions ----------
def train_kmeans(X_train: pd.DataFrame, max_k: int = 6) -> Tuple[KMeans, StandardScaler]:
    logger.info("Training KMeans for feature extraction...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    best_k, best_score = -1, -1
    for k in range(2, min(max_k, len(X_train) - 1)):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        score = silhouette_score(Xs, km.labels_,sample_size=10000, random_state=SEED)
        if score > best_score:
            best_k, best_score = k, score

    logger.info(f"Chosen k={best_k} with silhouette score={best_score:.4f}")
    final_km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20).fit(Xs)
    return final_km, scaler

def train_deep_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_loss = float('inf')
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
                logger.info(f"Deep Model early stopping at epoch {epoch+1}")
                break

    best_model_buffer.seek(0)
    model.load_state_dict(torch.load(best_model_buffer))
    return model

def train_sequence_model(model, train_loader, val_loader, epochs=15, lr=0.001, patience=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
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
                logger.info(f"LSTM early stopping at epoch {epoch+1}")
                break

    best_model_buffer.seek(0)
    model.load_state_dict(torch.load(best_model_buffer))
    return model


# ---------- High-level Orchestrator ----------
class PersonalizationModel:
    def __init__(self, rf_threshold: float = 0.65, lstm_threshold: float = 0.70):
        self.rf = None
        self.kmeans = None
        self.kmeans_scaler = None
        self.feature_scaler = None
        self.deep = None
        self.sequence_model = None
        self.event2id = None
        self.feature_cols = None
        self.sequence_max_len = 25

        self.rf_threshold = rf_threshold
        self.lstm_threshold = lstm_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_all(self, df: pd.DataFrame, target_col: str, merged_df: pd.DataFrame, feature_cols: Optional[list] = None):
        logger.info("Starting full model training pipeline...")
        self.feature_cols = feature_cols or [c for c in df.columns if c != target_col]

        X = df[self.feature_cols]
        y = df[target_col].astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

        self.kmeans, self.kmeans_scaler = train_kmeans(X_train)

        train_clusters = self.kmeans.predict(self.kmeans_scaler.transform(X_train))
        val_clusters = self.kmeans.predict(self.kmeans_scaler.transform(X_val))

        X_train_aug = np.column_stack((X_train.values, train_clusters))
        X_val_aug = np.column_stack((X_val.values, val_clusters))

        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train_aug)
        X_val_scaled = self.feature_scaler.transform(X_val_aug)

        logger.info("Training Random Forest...")
        rf_base = RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=100, max_depth=10)
        min_class_count = pd.Series(y_train).value_counts().min()

        if min_class_count < 2:
            logger.warning(f"Smallest cluster has {min_class_count} sample(s). Using uncalibrated fallback.")
            self.rf = rf_base.fit(X_train_scaled, y_train)
        else:
            cv_folds = min(3, min_class_count)
            self.rf = CalibratedClassifierCV(rf_base, cv=cv_folds).fit(X_train_scaled, y_train)

        logger.info("Training Deep Predictor...")
        train_ds = TabularDataset(X_train_scaled, y_train.values)
        val_ds = TabularDataset(X_val_scaled, y_val.values)
        self.deep = DeepPredictor(input_size=X_train_scaled.shape[1], num_classes=y.nunique())
        self.deep = train_deep_model(self.deep, DataLoader(train_ds, 64, True), DataLoader(val_ds, 64))

        logger.info("Training Sequence Model...")
        self.event2id = {event: i + 1 for i, event in enumerate(merged_df['event_name'].dropna().unique())}
        self.event2id['<PAD>'] = 0

        # FIX: Cast to object to remove categorical restrictions before filling with 0
        merged_df['event_id'] = merged_df['event_name'].astype(object).map(self.event2id).fillna(0).astype(int)

        user_sequences = merged_df.sort_values('timestamp').groupby('user_pseudo_id')['event_id'].agg(list).rename('sequence')
        aligned_data = df.join(user_sequences, on='user_pseudo_id').dropna(subset=['sequence', target_col])

        seqs = aligned_data['sequence'].tolist()
        labels = aligned_data[target_col].tolist()

        seq_train, seq_val, labels_train, labels_val = train_test_split(seqs, labels, test_size=0.2, random_state=SEED)

        train_seq_loader = DataLoader(SequenceDataset(seq_train, labels_train, max_len=self.sequence_max_len), batch_size=64, shuffle=True)
        val_seq_loader = DataLoader(SequenceDataset(seq_val, labels_val, max_len=self.sequence_max_len), batch_size=64)

        self.sequence_model = UserSequenceLSTM(len(self.event2id), y.nunique(), pad_idx=0)
        self.sequence_model = train_sequence_model(self.sequence_model, train_seq_loader, val_seq_loader)

    def predict_for_user(self, user_features: dict, user_sequence: list = None) -> dict:
        if not all([self.feature_scaler, self.deep, self.rf, self.kmeans]):
             raise RuntimeError("Models are not trained. Cannot make predictions.")

        x_vector = np.array([user_features.get(c, 0.0) for c in self.feature_cols]).reshape(1, -1)
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
            source = 'deep_model'

        if confidence < self.rf_threshold:
            if hasattr(self.rf, 'predict_proba'):
                rf_probs = self.rf.predict_proba(x_scaled).flatten()
                cluster = int(np.argmax(rf_probs))
                confidence = float(np.max(rf_probs))
            else:
                cluster = int(self.rf.predict(x_scaled)[0])
                confidence = 0.5
            source = 'random_forest_fallback'

        if self.sequence_model and self.event2id and user_sequence:
            try:
                self.sequence_model.eval()
                self.sequence_model.to(self.device)
                seq_ids = [self.event2id.get(event, 0) for event in user_sequence]
                if not seq_ids:
                    return {'cluster': cluster, 'confidence': confidence, 'source': source}

                seq_tensor = torch.tensor([seq_ids[:self.sequence_max_len]], dtype=torch.long).to(self.device)
                seq_len = torch.tensor([len(seq_ids[:self.sequence_max_len])], dtype=torch.long)

                with torch.no_grad():
                    lstm_logits = self.sequence_model(seq_tensor, seq_len)
                    lstm_probs = torch.softmax(lstm_logits, dim=1).cpu().numpy().flatten()
                    lstm_pred = int(np.argmax(lstm_probs))
                    lstm_conf = float(np.max(lstm_probs))

                if lstm_conf > self.lstm_threshold:
                    cluster = lstm_pred
                    confidence = lstm_conf
                    source = 'lstm_override'
            except Exception as e:
                logging.warning(f"LSTM prediction failed: {e}")

        return {'cluster': cluster, 'confidence': confidence, 'source': source}