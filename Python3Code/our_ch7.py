##############################################################
#                                                            #
#    Adapted from crowdsignals_ch7_classification.py for     #
#    our transportation mode detection dataset.              #
#    Chapter 7: Classical machine learning classification    #
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import LeaveOneGroupOut

from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation

DATA_PATH = Path(__file__).parent / 'intermediate_datafiles'
DATASET_FNAME = 'chapter4_result.csv'
RESULT_FNAME = 'chapter7_result.csv'
FIGURES_DIR = Path('figures') / 'our_ch7'

LABEL_PREFIX = 'label'


def load_dataset():
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset
    except IOError as e:
        print('File not found, try to run our_ch4.py first!')
        raise e


def derive_label_column(dataset):
    dataset = dataset.copy()
    label_cols = [c for c in dataset.columns if c.startswith(LABEL_PREFIX) and c != 'label']
    dataset['label'] = 'unknown'
    for col in label_cols:
        mode = col[len(LABEL_PREFIX):]
        dataset.loc[dataset[col] == 1, 'label'] = mode
    return dataset


def split_by_session(dataset, test_fraction=0.3, random_state=42):
    """Split data by session, stratified by the dominant label per session.

    Entire sessions go into either train or test so the model is
    evaluated on recordings it has never seen.  Within each transport
    mode we hold out roughly `test_fraction` of the sessions, ensuring
    every class appears on both sides of the split.
    """
    session_label = (dataset.groupby('session')['label']
                     .agg(lambda s: s.value_counts().idxmax()))

    train_sessions, test_sessions = [], []
    rng = np.random.RandomState(random_state)

    for label, sessions in session_label.groupby(session_label):
        sess_list = list(sessions.index)
        rng.shuffle(sess_list)
        n_test = max(1, int(len(sess_list) * test_fraction))
        test_sessions.extend(sess_list[:n_test])
        train_sessions.extend(sess_list[n_test:])

    train_sessions = set(train_sessions)
    test_sessions = set(test_sessions)

    train_mask = dataset['session'].isin(train_sessions)
    test_mask = dataset['session'].isin(test_sessions)

    print(f'Train sessions ({len(train_sessions)}): {sorted(train_sessions)}')
    print(f'Test sessions  ({len(test_sessions)}):  {sorted(test_sessions)}')

    return dataset[train_mask], dataset[test_mask]


def prepare_X_y(dataset, feature_cols):
    label_cols = [c for c in dataset.columns if c.startswith(LABEL_PREFIX) and c != 'label']
    meta_cols = label_cols + ['person', 'session', 'label']
    X_cols = [c for c in feature_cols if c not in meta_cols]
    X = dataset[X_cols].copy()
    y = dataset['label'].copy()
    return X, y


def build_feature_sets(all_columns):
    meta = {'person', 'session', 'label'}
    label_cols = {c for c in all_columns if c.startswith(LABEL_PREFIX)}
    exclude = meta | label_cols

    sensor_cols = [c for c in all_columns
                   if c not in exclude and '_temp_' not in c and '_freq' not in c and '_pse' not in c]
    temporal_cols = [c for c in all_columns if '_temp_' in c]
    frequency_cols = [c for c in all_columns if '_freq' in c or '_pse' in c]
    all_features = [c for c in all_columns if c not in exclude]

    return {
        'sensor_only': sensor_cols,
        'sensor+temporal': sensor_cols + temporal_cols,
        'all_features': all_features,
    }


def make_classifier_fns(train_X, train_y, test_X, gridsearch=True):
    learner = ClassificationAlgorithms()
    # Default-argument binding to capture current values (avoids late-binding)
    return [
        ('DT',  lambda l=learner, trX=train_X, trY=train_y, teX=test_X, gs=gridsearch:
             l.decision_tree(trX, trY, teX, gridsearch=gs, print_model_details=False)),
        ('RF',  lambda l=learner, trX=train_X, trY=train_y, teX=test_X, gs=gridsearch:
             l.random_forest(trX, trY, teX, gridsearch=gs, print_model_details=False)),
        ('KNN', lambda l=learner, trX=train_X, trY=train_y, teX=test_X, gs=gridsearch:
             l.k_nearest_neighbor(trX, trY, teX, gridsearch=gs)),
        ('NB',  lambda l=learner, trX=train_X, trY=train_y, teX=test_X:
             l.naive_bayes(trX, trY, teX)),
    ]


def run_classifiers(train_X, train_y, test_X, test_y, gridsearch=True):
    evaluator = ClassificationEvaluation()
    results = {}

    for name, run_fn in make_classifier_fns(train_X, train_y, test_X, gridsearch):
        print(f'  Training {name} ...')
        t0 = time.time()
        pred_train, pred_test, prob_train, prob_test = run_fn()
        elapsed = time.time() - t0

        acc_train = evaluator.accuracy(train_y, pred_train)
        acc_test = evaluator.accuracy(test_y, pred_test)

        results[name] = {
            'acc_train': acc_train,
            'acc_test': acc_test,
            'pred_test': pred_test,
            'prob_test': prob_test,
            'time': elapsed,
        }
        print(f'    {name}: train={acc_train:.3f}  test={acc_test:.3f}  ({elapsed:.1f}s)')

    return results


def leave_one_session_out_cv(dataset, feature_cols, gridsearch=False):
    """Evaluate classifiers with Leave-One-Session-Out cross-validation.

    Each fold holds out one session as the test set and trains on all
    others.  This gives a robust per-session generalization estimate
    even with very few sessions.  Grid search is disabled by default
    to keep runtime manageable (N_sessions * N_classifiers fits).
    """
    evaluator = ClassificationEvaluation()
    sessions = dataset['session'].unique()
    label_cols_bin = [c for c in dataset.columns if c.startswith(LABEL_PREFIX) and c != 'label']
    meta_cols = set(label_cols_bin) | {'person', 'session', 'label'}
    X_cols = [c for c in feature_cols if c not in meta_cols]

    algo_names = ['DT', 'RF', 'KNN', 'NB']
    fold_results = {a: [] for a in algo_names}

    for i, held_out in enumerate(sessions):
        train_mask = dataset['session'] != held_out
        test_mask = dataset['session'] == held_out

        train_X = dataset.loc[train_mask, X_cols]
        test_X = dataset.loc[test_mask, X_cols]
        train_y = dataset.loc[train_mask, 'label']
        test_y = dataset.loc[test_mask, 'label']

        if test_y.nunique() == 0 or train_y.nunique() < 2:
            continue

        scaler = StandardScaler()
        train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=X_cols, index=train_X.index)
        test_X = pd.DataFrame(scaler.transform(test_X), columns=X_cols, index=test_X.index)

        print(f'  LOSO fold {i+1}/{len(sessions)}: test={held_out} ({len(test_X)} rows)')

        for name, run_fn in make_classifier_fns(train_X, train_y, test_X, gridsearch):
            _, pred_test, _, _ = run_fn()
            acc = evaluator.accuracy(test_y, pred_test)
            f1 = f1_score(test_y, pred_test, average='weighted', zero_division=0)
            fold_results[name].append({'session': held_out, 'acc': acc, 'f1': f1})

    summary = {}
    for algo in algo_names:
        accs = [r['acc'] for r in fold_results[algo]]
        f1s = [r['f1'] for r in fold_results[algo]]
        summary[algo] = {
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'folds': fold_results[algo],
        }
    return summary


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


def plot_accuracy_comparison(results_by_set, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    set_names = list(results_by_set.keys())
    algo_names = list(results_by_set[set_names[0]].keys())
    x = np.arange(len(algo_names))
    width = 0.8 / len(set_names)

    for i, set_name in enumerate(set_names):
        accs = [results_by_set[set_name][algo]['acc_test'] for algo in algo_names]
        ax.bar(x + i * width, accs, width, label=set_name)

    ax.set_xlabel('Classifier')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification accuracy by feature set and algorithm')
    ax.set_xticks(x + width * (len(set_names) - 1) / 2)
    ax.set_xticklabels(algo_names)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, set_name in enumerate(set_names):
        for j, algo in enumerate(algo_names):
            acc = results_by_set[set_name][algo]['acc_test']
            ax.text(x[j] + i * width, acc + 0.01, f'{acc:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=45)

    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_feature_importance(train_X, train_y, top_n, path):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(train_X, train_y)

    importances = pd.Series(rf.feature_importances_, index=train_X.columns)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top.sort_values().plot.barh(ax=ax)
    ax.set_xlabel('Feature importance (Gini)')
    ax.set_title(f'Top {top_n} most important features (Random Forest)')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)

    return importances


def plot_loso_results(loso_summary, path):
    algo_names = list(loso_summary.keys())
    means = [loso_summary[a]['mean_acc'] for a in algo_names]
    stds = [loso_summary[a]['std_acc'] for a in algo_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algo_names, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_title('Leave-One-Session-Out CV — Mean Accuracy (± std)')
    ax.set_ylim(0, 1.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.2f}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    start_time = time.time()

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset()
    dataset = derive_label_column(dataset)
    dataset = dataset[dataset['label'] != 'unknown']
    dataset = dataset.dropna()

    print(f'Dataset shape: {dataset.shape}')
    print(f'Label distribution:\n{dataset["label"].value_counts().to_string()}')

    sessions_overview = dataset.groupby('session')['label'].agg(
        lambda s: s.value_counts().idxmax()).reset_index()
    sessions_overview.columns = ['session', 'dominant_label']
    sessions_overview['n_rows'] = dataset.groupby('session').size().values
    print(f'\nSession overview:\n{sessions_overview.to_string(index=False)}')
    print()

    # --- Build feature sets ---
    feature_sets = build_feature_sets(dataset.columns)
    for name, cols in feature_sets.items():
        print(f'Feature set "{name}": {len(cols)} features')

    # ================================================================
    # PART 1: Leave-One-Session-Out CV (robust evaluation)
    # ================================================================
    print('\n' + '=' * 70)
    print('PART 1: Leave-One-Session-Out Cross-Validation')
    print('=' * 70)

    best_loso_f1 = 0
    best_loso_config = None
    loso_all = {}

    for set_name, feature_cols in feature_sets.items():
        print(f'\n--- Feature set: {set_name} ({len(feature_cols)} features) ---')
        loso = leave_one_session_out_cv(dataset, feature_cols, gridsearch=False)
        loso_all[set_name] = loso

        for algo, s in loso.items():
            print(f'  {algo}: acc={s["mean_acc"]:.3f}±{s["std_acc"]:.3f}  '
                  f'f1={s["mean_f1"]:.3f}±{s["std_f1"]:.3f}')
            if s['mean_f1'] > best_loso_f1:
                best_loso_f1 = s['mean_f1']
                best_loso_config = (set_name, algo)

    print(f'\nBest LOSO config: {best_loso_config[1]} with {best_loso_config[0]} '
          f'(mean F1 = {best_loso_f1:.3f})')

    # LOSO summary table
    print('\n' + '-' * 80)
    header = f'{"Feature Set":<20} {"Algo":<6} {"Acc (mean±std)":>18} {"F1 (mean±std)":>18}'
    print(header)
    print('-' * 80)
    for set_name, loso in loso_all.items():
        for algo, s in loso.items():
            print(f'{set_name:<20} {algo:<6} '
                  f'{s["mean_acc"]:.3f}±{s["std_acc"]:.3f}      '
                  f'{s["mean_f1"]:.3f}±{s["std_f1"]:.3f}')

    plot_loso_results(loso_all[best_loso_config[0]], FIGURES_DIR / 'loso_accuracy')
    print('Saved loso_accuracy')

    # ================================================================
    # PART 2: Stratified session split (detailed eval of best setup)
    # ================================================================
    print('\n' + '=' * 70)
    print('PART 2: Stratified Session Split — Detailed Evaluation')
    print('=' * 70)

    train_data, test_data = split_by_session(dataset, test_fraction=FLAGS.test_fraction)

    print(f'Train: {len(train_data)} rows, Test: {len(test_data)} rows')
    print(f'Train labels:\n{train_data["label"].value_counts().to_string()}')
    print(f'Test labels:\n{test_data["label"].value_counts().to_string()}')

    # Scale features
    all_feature_cols = feature_sets['all_features']
    scaler = StandardScaler()
    scaler.fit(train_data[all_feature_cols])

    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[all_feature_cols] = scaler.transform(train_data[all_feature_cols])
    test_scaled[all_feature_cols] = scaler.transform(test_data[all_feature_cols])

    # Run classifiers on each feature set
    results_by_set = {}
    best_test_acc = 0
    best_config = None

    for set_name, feature_cols in feature_sets.items():
        print(f'\n=== Feature set: {set_name} ({len(feature_cols)} features) ===')

        train_X, train_y = prepare_X_y(train_scaled, feature_cols)
        test_X, test_y = prepare_X_y(test_scaled, feature_cols)

        results = run_classifiers(
            train_X, train_y, test_X, test_y,
            gridsearch=FLAGS.gridsearch)

        results_by_set[set_name] = results

        for algo_name, res in results.items():
            if res['acc_test'] > best_test_acc:
                best_test_acc = res['acc_test']
                best_config = (set_name, algo_name)

    # Summary table
    print('\n' + '-' * 70)
    header = f'{"Feature Set":<20} {"Algo":<6} {"Train Acc":>10} {"Test Acc":>10} {"Time (s)":>10}'
    print(header)
    print('-' * 70)
    for set_name, results in results_by_set.items():
        for algo_name, res in results.items():
            print(f'{set_name:<20} {algo_name:<6} {res["acc_train"]:>10.3f} {res["acc_test"]:>10.3f} {res["time"]:>10.1f}')

    print(f'\nBest split config: {best_config[1]} with {best_config[0]} (test acc = {best_test_acc:.3f})')

    # Detailed classification report for best model
    best_set, best_algo = best_config
    best_pred = results_by_set[best_set][best_algo]['pred_test']
    _, test_y = prepare_X_y(test_scaled, feature_sets[best_set])
    labels = sorted(dataset['label'].unique())

    print(f'\n=== Classification report ({best_algo}, {best_set}) ===')
    print(classification_report(test_y, best_pred, target_names=labels, zero_division=0))

    # Figures
    plot_accuracy_comparison(results_by_set, FIGURES_DIR / 'accuracy_comparison')
    print('Saved accuracy_comparison')

    plot_confusion_matrix(test_y, best_pred, labels,
                          f'Confusion Matrix — {best_algo} ({best_set})',
                          FIGURES_DIR / 'confusion_matrix_best')
    print('Saved confusion_matrix_best')

    train_X_all, train_y_all = prepare_X_y(train_scaled, feature_sets['all_features'])
    importances = plot_feature_importance(train_X_all, train_y_all, 20,
                                          FIGURES_DIR / 'feature_importance')
    print('Saved feature_importance')

    print(f'\nTop 10 features by importance:')
    for feat, imp in importances.nlargest(10).items():
        print(f'  {feat}: {imp:.4f}')

    # Save predictions
    test_output = test_data[['session', 'person', 'label']].copy()
    test_output['predicted'] = best_pred
    test_output.to_csv(DATA_PATH / RESULT_FNAME)
    print(f'\nSaved predictions to {DATA_PATH / RESULT_FNAME}')
    print(f'Total runtime: {time.time() - start_time:.1f}s')


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
        f.write(f'# our_ch7.py — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
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

    parser.add_argument('--test-fraction', type=float, default=0.3,
                        help='Fraction of sessions held out for testing.')
    parser.add_argument('--gridsearch', action='store_true', default=True,
                        help='Enable grid search for hyperparameter tuning.')
    parser.add_argument('--no-gridsearch', dest='gridsearch', action='store_false',
                        help='Disable grid search (faster, uses defaults).')

    FLAGS, _ = parser.parse_known_args()

    with save_output(FIGURES_DIR / 'output.txt'):
        main()
