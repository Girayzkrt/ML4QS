##############################################################
#                                                            #
#    Deep learning approach for transportation mode          #
#    classification using raw sensor windows.                #
#    Complements the classical ML in our_ch7.py.             #
#                                                            #
##############################################################

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score)

from Chapter7.Evaluation import ClassificationEvaluation

DATA_PATH = Path(__file__).parent / 'intermediate_datafiles'
DATASET_FNAME = 'chapter3_result_final.csv'
FIGURES_DIR = Path('figures') / 'our_ch8'

SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'bar_pressure']
LABEL_PREFIX = 'label'

DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available()
                       else 'cpu')


def load_dataset():
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset
    except IOError as e:
        print('File not found, try to run our_ch3_rest.py first!')
        raise e


def derive_label_column(dataset):
    dataset = dataset.copy()
    label_cols = [c for c in dataset.columns if c.startswith(LABEL_PREFIX) and c != 'label']
    dataset['label'] = 'unknown'
    for col in label_cols:
        mode = col[len(LABEL_PREFIX):]
        dataset.loc[dataset[col] == 1, 'label'] = mode
    return dataset


def create_windows(dataset, window_size, step_size):
    """Slice raw sensor data into fixed-length windows per session.

    Returns:
        X: np.ndarray of shape (n_windows, n_channels, window_size)
        y: np.ndarray of string labels, shape (n_windows,)
        sessions: np.ndarray of session ids, shape (n_windows,)
    """
    X_windows, y_windows, session_ids = [], [], []

    for session_id, session_data in dataset.groupby('session', sort=False):
        sensor_values = session_data[SENSOR_COLS].values  # (T, C)
        labels = session_data['label'].values

        for start in range(0, len(sensor_values) - window_size + 1, step_size):
            window = sensor_values[start:start + window_size]  # (window_size, C)
            window_labels = labels[start:start + window_size]

            if np.any(pd.isna(window)):
                continue

            # Majority label for this window
            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[counts.argmax()]
            if majority_label == 'unknown':
                continue

            X_windows.append(window.T)  # (C, window_size)
            y_windows.append(majority_label)
            session_ids.append(session_id)

    return np.array(X_windows, dtype=np.float32), np.array(y_windows), np.array(session_ids)


class SensorWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_size=128, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if n_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, channels, time) -> (batch, time, channels) for LSTM
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        # Concatenate final hidden states from both directions
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h_cat)


def predict(model, X_np):
    """Run inference on a numpy array. Returns predicted class indices and probabilities."""
    model.eval()
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


def train_model(model, train_loader, val_X, val_y, label_encoder,
                n_epochs=50, lr=1e-3, patience=10):
    """Train a PyTorch model, optionally with early stopping on validation F1.

    When ``patience`` is ``None``, training runs for all ``n_epochs`` without
    consulting the validation data, which prevents test-data leakage when the
    caller passes the test set as ``val_X``/``val_y``.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        if patience is None:
            # No early stopping — skip validation entirely.
            continue

        # Validation
        val_pred, _ = predict(model, val_X)
        val_pred_labels = label_encoder.inverse_transform(val_pred)
        val_f1 = f1_score(val_y, val_pred_labels, average='weighted', zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model, best_f1


def standardize_windows(train_X, test_X):
    """Standardize each sensor channel using train statistics."""
    train_X = train_X.copy()
    test_X = test_X.copy()
    for c in range(train_X.shape[1]):
        mean = train_X[:, c, :].mean()
        std = train_X[:, c, :].std()
        if std < 1e-8:
            std = 1.0
        train_X[:, c, :] = (train_X[:, c, :] - mean) / std
        test_X[:, c, :] = (test_X[:, c, :] - mean) / std
    return train_X, test_X


def leave_one_session_out_cv(X, y, sessions, label_encoder, window_size,
                              n_epochs=50, lr=1e-3, batch_size=32):
    """Evaluate CNN and LSTM with Leave-One-Session-Out cross-validation."""
    evaluator = ClassificationEvaluation()
    unique_sessions = np.unique(sessions)
    n_channels = X.shape[1]
    n_classes = len(label_encoder.classes_)
    model_names = ['CNN', 'LSTM']
    fold_results = {name: [] for name in model_names}

    for i, held_out in enumerate(unique_sessions):
        train_mask = sessions != held_out
        test_mask = sessions == held_out

        train_X, test_X = standardize_windows(X[train_mask], X[test_mask])
        train_y, test_y = y[train_mask], y[test_mask]

        train_y_enc = label_encoder.transform(train_y)
        train_ds = SensorWindowDataset(train_X, train_y_enc)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        print(f'  LOSO fold {i+1}/{len(unique_sessions)}: test={held_out} ({test_mask.sum()} windows)')

        models = {
            'CNN': CNN1D(n_channels, n_classes),
            'LSTM': LSTMClassifier(n_channels, n_classes),
        }

        for name in model_names:
            model, _ = train_model(models[name], train_loader, test_X, test_y,
                                   label_encoder, n_epochs=n_epochs, lr=lr)
            preds_idx, _ = predict(model, test_X)
            pred_labels = label_encoder.inverse_transform(preds_idx)

            acc = evaluator.accuracy(test_y, pred_labels)
            f1 = f1_score(test_y, pred_labels, average='weighted', zero_division=0)
            fold_results[name].append({'session': held_out, 'acc': acc, 'f1': f1})

    summary = {}
    for name in model_names:
        accs = [r['acc'] for r in fold_results[name]]
        f1s = [r['f1'] for r in fold_results[name]]
        summary[name] = {
            'mean_acc': np.mean(accs), 'std_acc': np.std(accs),
            'mean_f1': np.mean(f1s), 'std_f1': np.std(f1s),
            'folds': fold_results[name],
        }
    return summary


def plot_loso_comparison(loso_summary, path):
    """Bar chart comparing CNN and LSTM LOSO accuracy."""
    algo_names = list(loso_summary.keys())
    means = [loso_summary[a]['mean_acc'] for a in algo_names]
    stds = [loso_summary[a]['std_acc'] for a in algo_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algo_names, means, yerr=stds, capsize=5,
                  color=['#4C72B0', '#DD8452'], edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_title('Deep Learning LOSO CV — Mean Accuracy (± std)')
    ax.set_ylim(0, 1.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.2f}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_training_comparison(results, path):
    """Side-by-side bar chart of CNN vs LSTM test accuracy."""
    names = list(results.keys())
    accs = [results[n]['acc_test'] for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, accs, color=['#4C72B0', '#DD8452'], edgecolor='black')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Deep Learning — Test Accuracy (Stratified Session Split)')
    ax.set_ylim(0, 1.05)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.01,
                f'{a:.2f}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def split_by_session(sessions, y, test_fraction=0.3, random_state=42):
    """Split window indices by session, stratified by label."""
    session_label = {}
    for s in np.unique(sessions):
        mask = sessions == s
        labels, counts = np.unique(y[mask], return_counts=True)
        session_label[s] = labels[counts.argmax()]

    train_sessions, test_sessions = [], []
    rng = np.random.RandomState(random_state)

    label_to_sessions = {}
    for s, lbl in session_label.items():
        label_to_sessions.setdefault(lbl, []).append(s)

    for lbl, sess_list in label_to_sessions.items():
        rng.shuffle(sess_list)
        n_test = max(1, int(len(sess_list) * test_fraction))
        test_sessions.extend(sess_list[:n_test])
        train_sessions.extend(sess_list[n_test:])

    train_labels_present = set()
    for s in train_sessions:
        train_labels_present.add(session_label[s])
    for lbl in label_to_sessions:
        if lbl not in train_labels_present:
            print(f'WARNING: label "{lbl}" has no sessions in train set — '
                  f'all {len(label_to_sessions[lbl])} session(s) are in test!')

    print(f'Train sessions ({len(train_sessions)}): {sorted(train_sessions)}')
    print(f'Test sessions  ({len(test_sessions)}):  {sorted(test_sessions)}')

    train_mask = np.isin(sessions, train_sessions)
    test_mask = np.isin(sessions, test_sessions)
    return train_mask, test_mask


def main():
    start_time = time.time()
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset()
    dataset = derive_label_column(dataset)
    dataset = dataset[dataset['label'] != 'unknown']
    dataset = dataset.dropna(subset=SENSOR_COLS)

    print(f'Dataset shape: {dataset.shape}')
    print(f'Label distribution:\n{dataset["label"].value_counts().to_string()}')
    print(f'Device: {DEVICE}')

    window_size = FLAGS.window_size
    step_size = FLAGS.step_size

    X, y, sessions = create_windows(dataset, window_size, step_size)
    print(f'\nWindows: {X.shape[0]}, Channels: {X.shape[1]}, Window length: {X.shape[2]}')
    print(f'Labels: {dict(zip(*np.unique(y, return_counts=True)))}')
    print(f'Sessions: {np.unique(sessions)}')

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    labels = sorted(label_encoder.classes_)
    n_classes = len(labels)
    n_channels = X.shape[1]

    # ================================================================
    # PART 1: Leave-One-Session-Out Cross-Validation
    # ================================================================
    print('\n' + '=' * 70)
    print('PART 1: Leave-One-Session-Out Cross-Validation')
    print('=' * 70)

    loso = leave_one_session_out_cv(
        X, y, sessions, label_encoder, window_size,
        n_epochs=FLAGS.epochs, lr=FLAGS.lr, batch_size=FLAGS.batch_size)

    for name, s in loso.items():
        print(f'  {name}: acc={s["mean_acc"]:.3f}±{s["std_acc"]:.3f}  '
              f'f1={s["mean_f1"]:.3f}±{s["std_f1"]:.3f}')

    plot_loso_comparison(loso, FIGURES_DIR / 'loso_accuracy')
    print('Saved loso_accuracy')

    # ================================================================
    # PART 2: Stratified Session Split — Detailed Evaluation
    # ================================================================
    print('\n' + '=' * 70)
    print('PART 2: Stratified Session Split — Detailed Evaluation')
    print('=' * 70)

    train_mask, test_mask = split_by_session(sessions, y, test_fraction=FLAGS.test_fraction)
    train_X, test_X = standardize_windows(X[train_mask], X[test_mask])
    train_y, test_y = y[train_mask], y[test_mask]

    print(f'Train: {len(train_y)} windows, Test: {len(test_y)} windows')
    print(f'Train labels: {dict(zip(*np.unique(train_y, return_counts=True)))}')
    print(f'Test labels:  {dict(zip(*np.unique(test_y, return_counts=True)))}')

    train_y_enc = label_encoder.transform(train_y)
    train_ds = SensorWindowDataset(train_X, train_y_enc)
    train_loader = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)

    evaluator = ClassificationEvaluation()
    results = {}

    for name, model_cls in [('CNN', CNN1D), ('LSTM', LSTMClassifier)]:
        print(f'\n--- Training {name} ---')

        model = model_cls(n_channels, n_classes)

        model, best_val_f1 = train_model(
            model, train_loader, test_X, test_y, label_encoder,
            n_epochs=FLAGS.epochs, lr=FLAGS.lr, patience=None)

        pred_idx, pred_probs = predict(model, test_X)
        pred_labels = label_encoder.inverse_transform(pred_idx)

        train_pred_idx, _ = predict(model, train_X)
        train_pred_labels = label_encoder.inverse_transform(train_pred_idx)

        acc_train = evaluator.accuracy(train_y, train_pred_labels)
        acc_test = evaluator.accuracy(test_y, pred_labels)

        results[name] = {
            'acc_train': acc_train,
            'acc_test': acc_test,
            'pred_test': pred_labels,
        }
        print(f'  {name}: train={acc_train:.3f}  test={acc_test:.3f}')

    # Summary
    print('\n' + '-' * 50)
    print(f'{"Model":<8} {"Train Acc":>10} {"Test Acc":>10}')
    print('-' * 50)
    for name, res in results.items():
        print(f'{name:<8} {res["acc_train"]:>10.3f} {res["acc_test"]:>10.3f}')

    # Best model confusion matrix and classification report
    best_name = max(results, key=lambda n: results[n]['acc_test'])
    best_pred = results[best_name]['pred_test']

    print(f'\n=== Classification report ({best_name}) ===')
    print(classification_report(test_y, best_pred, target_names=labels, zero_division=0))

    plot_confusion_matrix(test_y, best_pred, labels,
                          f'Confusion Matrix — {best_name} (Deep Learning)',
                          FIGURES_DIR / 'confusion_matrix_best')
    print('Saved confusion_matrix_best')

    plot_training_comparison(results, FIGURES_DIR / 'accuracy_comparison')
    print('Saved accuracy_comparison')

    print(f'\nTotal runtime: {time.time() - start_time:.1f}s')


class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def save_output(path):
    """Context manager that tees stdout to both terminal and a text file."""
    from contextlib import contextmanager

    @contextmanager
    def _tee(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, 'w')
        f.write(f'# our_ch8_final.py — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old
            f.close()
            print(f'Output saved to {path}')

    return _tee(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', type=int, default=40,
                        help='Window size in time steps (default: 40 = 10s at 4Hz)')
    parser.add_argument('--step-size', type=int, default=20,
                        help='Step size between windows (default: 20 = 50%% overlap)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs per model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--test-fraction', type=float, default=0.3,
                        help='Fraction of sessions held out for testing')

    FLAGS, _ = parser.parse_known_args()

    with save_output(FIGURES_DIR / 'output.txt'):
        main()
