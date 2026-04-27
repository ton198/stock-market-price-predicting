import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Path ───────────────────────────────────────────────────────────────────────
CSV_PATH = 'datasets/nasdaq_multivariate.csv'
SAVE_PATH = 'models/stage1_model.pth'

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 60
TRAIN_STRIDE = 3
HIDDEN_SIZE  = 32
DROPOUT      = 0.3
LR           = 0.001
N_EPOCHS     = 300
PATIENCE     = 40   # early-stopping patience (notebook markdown said 40; code had 10 which matched the LR scheduler and prevented LR reduction from helping)
LR_PATIENCE  = 10   # ReduceLROnPlateau patience
BATCH_SIZE   = 32
N_RESTARTS   = 10

# ── Feature lists (identical to notebook) ─────────────────────────────────────
FEATURE_COLS = [
    'Return', 'Return_lag1', 'Return_lag2', 'Return_lag5',
    'Price_SMA20', 'Price_SMA50',
    'RSI_14', 'MACD_Hist', 'BB_pos', 'BB_width',
    'Volume_ratio', 'Volume_change',
    'VIX', 'TNX', 'FedRate', 'CPI_MoM',
    'Regime', 'VIX_percentile',
    'Momentum_20d', 'Momentum_60d', 'Momentum_252d',
    'Volatility_20d', 'ATR_14',
    'Yield_slope', 'Dist_52w_high', 'Dist_52w_low',
    'VIX_change5', 'Gap',
    'FedRate_chg20', 'FedRate_chg60',
]
MACRO_COLS = [
    'FedRate', 'FedRate_chg20', 'FedRate_chg60',
    'TNX', 'Yield_slope', 'VIX', 'VIX_percentile',
]
TARGET_COL = 'target'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model ──────────────────────────────────────────────────────────────────────
class ReturnPredictor(nn.Module):
    def __init__(self, n_features, n_macro, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm      = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.macro_mlp = nn.Sequential(nn.Linear(n_macro, 16), nn.ReLU())
        self.head      = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + 16, 1))

    def forward(self, x_seq, macro):
        lstm_out, _ = self.lstm(x_seq)
        h_last      = lstm_out[:, -1, :]
        m           = self.macro_mlp(macro)
        return self.head(torch.cat([h_last, m], dim=-1)).squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, macro_cols, target_col, window=60, stride=1):
        self.X   = df[feature_cols].values.astype(np.float32)
        self.Xm  = df[macro_cols].values.astype(np.float32)
        self.Y   = df[target_col].values.astype(np.float32)
        self.w   = window
        self.idx = list(range(0, len(df) - window, stride))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.idx[i]
        return (
            torch.tensor(self.X[s : s + self.w]),
            torch.tensor(self.Xm[s + self.w - 1]),
            torch.tensor(self.Y[s + self.w - 1]),
        )


# ── Pipeline steps ─────────────────────────────────────────────────────────────
def load_and_engineer_features(csv_path):
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows, {df.shape[1]} columns')

    df['Forward_20d']  = df['Close'].shift(-20) / df['Close'] - 1
    df['target']       = (df['Forward_20d'] > df['Forward_20d'].median()).astype(int)

    df['Regime']       = (df['Close'] > df['Close'].rolling(60).mean()).astype(int)
    df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    df['Momentum_60d'] = df['Close'] / df['Close'].shift(60) - 1
    df['Momentum_252d']= df['Close'] / df['Close'].shift(252) - 1
    df['Dist_52w_high']= df['Close'] / df['Close'].rolling(252).max() - 1
    df['Dist_52w_low'] = df['Close'] / df['Close'].rolling(252).min() - 1

    df['Volatility_20d'] = df['Return'].rolling(20).std()
    df['ATR_14']         = (df['High'] - df['Low']).rolling(14).mean() / df['Close']

    df['VIX_percentile'] = df['VIX'].rolling(252).rank(pct=True)
    df['VIX_change5']    = df['VIX'].diff(5)
    df['Yield_slope']    = df['TNX'] - df['FedRate']
    df['FedRate_chg20']  = df['FedRate'].diff(20)
    df['FedRate_chg60']  = df['FedRate'].diff(60)

    df['Gap'] = df['Open'] / df['Close'].shift(1) - 1

    df.dropna(inplace=True)
    print(f'After dropna: {len(df)} rows  |  target positive rate: {df["target"].mean():.3f}')
    return df


def split_data(df):
    """Chronological 70 / 10 / 10 / 10 split.

    Returns train_df, val_df, lstm_test_df, rl_test_df.
    The rl_test_df is reserved for Stage 2 and never used during LSTM training.
    """
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.80)
    t3 = int(n * 0.90)

    train_df     = df.iloc[:t1]
    val_df       = df.iloc[t1:t2]
    lstm_test_df = df.iloc[t2:t3]
    rl_test_df   = df.iloc[t3:]

    print(
        f'Train {train_df.shape} | Val {val_df.shape} | '
        f'LSTM test {lstm_test_df.shape} | RL test {rl_test_df.shape}'
    )
    return train_df, val_df, lstm_test_df, rl_test_df


def build_dataloaders(train_df, val_df, lstm_test_df):
    """Fit scaler on train only; return loaders and fitted scaler."""
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    train_scaled[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])

    val_scaled = val_df.copy()
    val_scaled[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])

    test_scaled = lstm_test_df.copy()
    test_scaled[FEATURE_COLS] = scaler.transform(lstm_test_df[FEATURE_COLS])

    train_ds = SequenceDataset(train_scaled, FEATURE_COLS, MACRO_COLS, TARGET_COL, WINDOW_SIZE, TRAIN_STRIDE)
    val_ds   = SequenceDataset(val_scaled,   FEATURE_COLS, MACRO_COLS, TARGET_COL, WINDOW_SIZE, 1)
    test_ds  = SequenceDataset(test_scaled,  FEATURE_COLS, MACRO_COLS, TARGET_COL, WINDOW_SIZE, 1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train samples: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}')
    return train_loader, val_loader, test_loader, scaler


def train(train_loader, val_loader, n_restarts=10):
    """Multi-restart training; returns the model with the best val loss."""
    loss_fn = nn.BCEWithLogitsLoss()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train_one_run(seed):
        set_seed(seed)
        model = ReturnPredictor(len(FEATURE_COLS), len(MACRO_COLS), HIDDEN_SIZE, DROPOUT).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)
        sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=LR_PATIENCE, min_lr=1e-5)

        best_val, best_state, no_improve = float('inf'), None, 0

        for epoch in range(N_EPOCHS):
            model.train()
            t_loss = 0.0
            for x, xm, y in train_loader:
                x, xm, y = x.to(DEVICE), xm.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(model(x, xm), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                t_loss += loss.item()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, xm, y in val_loader:
                    x, xm, y = x.to(DEVICE), xm.to(DEVICE), y.to(DEVICE)
                    v_loss += loss_fn(model(x, xm), y).item()
            v_loss /= len(val_loader)
            sch.step(v_loss)

            if v_loss < best_val:
                best_val   = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:  # allow LR reduction to take effect before stopping
                break

        return best_val, best_state, epoch + 1

    print(f'Device: {DEVICE}')
    print(f'Running {n_restarts} restarts...')

    best_val_overall, best_state_overall, best_run = float('inf'), None, -1

    for i in range(n_restarts):
        val, state, epochs = train_one_run(seed=i * 17)
        print(f'  Restart {i+1:2d}: val_loss={val:.6f}  ({epochs} epochs)')
        if val < best_val_overall:
            best_val_overall, best_state_overall, best_run = val, state, i

    print(f'\nBest restart: #{best_run + 1}  val_loss={best_val_overall:.6f}')

    model = ReturnPredictor(len(FEATURE_COLS), len(MACRO_COLS), HIDDEN_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(best_state_overall)
    return model


def evaluate(model, test_loader):
    """Print AUC, Accuracy, F1, and high-confidence subset results."""
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, xm, y in test_loader:
            prob = torch.sigmoid(model(x.to(DEVICE), xm.to(DEVICE)))
            preds_list.append(prob.cpu().numpy())
            targets_list.append(y.numpy())

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    true_dir = (targets > 0.5).astype(int)
    pred_dir = (preds   > 0.5).astype(int)

    auc      = roc_auc_score(true_dir, preds)
    corr     = np.corrcoef(preds, targets)[0, 1]
    baseline = true_dir.mean()
    acc      = (pred_dir == true_dir).mean()
    f1       = f1_score(true_dir, pred_dir)

    print('===== LSTM TEST RESULTS =====')
    print(f'AUC-ROC  : {auc:.4f}')
    print(f'Corr     : {corr:.4f}')
    print(f'Baseline : {baseline:.4f}  (naive always-up accuracy)')
    print(f'Accuracy : {acc:.4f}')
    print(f'F1 Score : {f1:.4f}')

    CONF    = 0.6
    hc_mask = (preds > CONF) | (preds < (1 - CONF))
    if hc_mask.sum() > 0:
        hc_acc = (pred_dir[hc_mask] == true_dir[hc_mask]).mean()
        print(f'\n===== HIGH-CONFIDENCE SUBSET (prob >{CONF} or <{1 - CONF}) =====')
        print(f'Coverage : {hc_mask.mean():.1%}  ({hc_mask.sum()} samples)')
        print(f'Accuracy : {hc_acc:.4f}')
        if len(np.unique(true_dir[hc_mask])) > 1:
            hc_auc = roc_auc_score(true_dir[hc_mask], preds[hc_mask])
            print(f'AUC-ROC  : {hc_auc:.4f}')
        else:
            print('AUC-ROC  : n/a (only one class in subset)')


def main():
    df = load_and_engineer_features(CSV_PATH)
    train_df, val_df, lstm_test_df, rl_test_df = split_data(df)
    train_loader, val_loader, test_loader, scaler = build_dataloaders(train_df, val_df, lstm_test_df)

    model = train(train_loader, val_loader, n_restarts=N_RESTARTS)
    evaluate(model, test_loader)

    torch.save(model.state_dict(), SAVE_PATH)
    joblib.dump(scaler, 'scaler.joblib')
    print(f'\nSaved: {SAVE_PATH}  |  scaler.joblib')


if __name__ == '__main__':
    main()
