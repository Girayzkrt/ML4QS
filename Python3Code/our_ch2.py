##############################################################
#                                                            #
#    Adapted from crowdsignals_ch2.py for our Phyphox        #
#    transportation mode detection dataset.                  #
#    Chapter 2: Initial exploration of the dataset.          #
#                                                            #
##############################################################

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

DATASET_PATH = Path(__file__).parent / 'datasets' / 'our_data'
RESULT_PATH = Path(__file__).parent / 'intermediate_datafiles'
RESULT_FNAME = 'chapter2_result.csv'

GRANULARITIES = [250]

[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    dataset.add_numerical_dataset('accelerometer.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'acc_')
    dataset.add_numerical_dataset('gyroscope.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_')
    dataset.add_numerical_dataset('barometer.csv', 'timestamps', ['pressure'], 'avg', 'bar_')

    dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')

    dataset = dataset.data_table

    # Add person and session columns by matching timestamps to session ranges
    sessions = pd.read_csv(DATASET_PATH / 'sessions.csv')
    sessions['label_start'] = pd.to_datetime(sessions['label_start'])
    sessions['label_end'] = pd.to_datetime(sessions['label_end'])

    dataset['person'] = pd.array([pd.NA] * len(dataset), dtype='string')
    dataset['session'] = pd.array([pd.NA] * len(dataset), dtype='string')
    for _, row in sessions.iterrows():
        mask = (dataset.index >= row['label_start']) & (dataset.index <= row['label_end'])
        dataset.loc[mask, 'person'] = row['person']
        dataset.loc[mask, 'session'] = row['session_id']

    # Drop rows that fall in the gaps between sessions (no sensor data, no label)
    dataset = dataset.dropna(subset=['session'])
    print(f'Dataset shape after removing inter-session gaps: {dataset.shape}')

    # Convert sensor columns to numeric (CreateDataset stores them as object dtype)
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'bar_pressure']
    for col in sensor_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

    FIGURES_DIR = Path('figures') / 'our_ch2'
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    # Derive a label column for easy grouping
    dataset['label'] = 'unknown'
    for col in [c for c in dataset.columns if c.startswith('label') and c != 'label']:
        mode = col[len('label'):]
        dataset.loc[dataset[col] == 1, 'label'] = mode

    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    modes = [m for m in dataset['label'].unique() if m != 'unknown']

    # ---- Figure 1: Per-mode accelerometer time-series ----
    fig, axes = plt.subplots(len(modes), 1, figsize=(12, 4 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        mode_data = dataset[dataset['label'] == mode]
        first_session = mode_data['session'].dropna().unique()[0]
        seg = mode_data[mode_data['session'] == first_session].copy()
        t = (seg.index - seg.index[0]).total_seconds()
        ax.plot(t, seg['acc_x'], label='acc_x', alpha=0.8)
        ax.plot(t, seg['acc_y'], label='acc_y', alpha=0.8)
        ax.plot(t, seg['acc_z'], label='acc_z', alpha=0.8)
        ax.set_title(f'Linear Accelerometer — {mode}')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'accelerometer_per_mode.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'accelerometer_per_mode.pdf')
    plt.close(fig)
    print('Saved accelerometer_per_mode')

    # ---- Figure 2: Per-mode gyroscope time-series ----
    fig, axes = plt.subplots(len(modes), 1, figsize=(12, 4 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        mode_data = dataset[dataset['label'] == mode]
        first_session = mode_data['session'].dropna().unique()[0]
        seg = mode_data[mode_data['session'] == first_session].copy()
        t = (seg.index - seg.index[0]).total_seconds()
        ax.plot(t, seg['gyr_x'], label='gyr_x', alpha=0.8)
        ax.plot(t, seg['gyr_y'], label='gyr_y', alpha=0.8)
        ax.plot(t, seg['gyr_z'], label='gyr_z', alpha=0.8)
        ax.set_title(f'Gyroscope — {mode}')
        ax.set_ylabel('Angular velocity (rad/s)')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'gyroscope_per_mode.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'gyroscope_per_mode.pdf')
    plt.close(fig)
    print('Saved gyroscope_per_mode')

    # ---- Figure 3: Per-mode barometer time-series ----
    fig, axes = plt.subplots(len(modes), 1, figsize=(12, 3 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        mode_data = dataset[dataset['label'] == mode]
        first_session = mode_data['session'].dropna().unique()[0]
        seg = mode_data[mode_data['session'] == first_session].copy()
        t = (seg.index - seg.index[0]).total_seconds()
        bar = seg['bar_pressure'].dropna()
        t_bar = (bar.index - seg.index[0]).total_seconds()
        ax.plot(t_bar, bar, 'b-o', markersize=2, label='bar_pressure')
        ax.set_title(f'Barometer — {mode}')
        ax.set_ylabel('Pressure (hPa)')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'barometer_per_mode.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'barometer_per_mode.pdf')
    plt.close(fig)
    print('Saved barometer_per_mode')

    # ---- Figure 4: Boxplots per mode ----
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    gyr_cols = ['gyr_x', 'gyr_y', 'gyr_z']
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        mode_data = dataset[dataset['label'] == mode]
        mode_data[acc_cols + gyr_cols].boxplot(ax=ax)
        ax.set_title(f'Sensor distributions — {mode}')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'boxplot_per_mode.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'boxplot_per_mode.pdf')
    plt.close(fig)
    print('Saved boxplot_per_mode')

    # ---- Figure 5: FFT / frequency spectrum comparison per mode ----
    fs = 1000.0 / milliseconds_per_instance
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for mode in modes:
        mode_data = dataset[dataset['label'] == mode]
        first_session = mode_data['session'].dropna().unique()[0]
        seg = mode_data[mode_data['session'] == first_session]
        signal = seg['acc_x'].dropna().values
        N = len(signal)
        freqs = rfftfreq(N, d=1.0 / fs)
        amplitudes = np.abs(rfft(signal)) * 2.0 / N
        axes[0].plot(freqs, amplitudes, label=mode, alpha=0.8)

    axes[0].set_title('FFT of acc_x per transport mode')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Amplitude (m/s²)')
    axes[0].legend()
    axes[0].set_xlim(0, fs / 2)

    for mode in modes:
        mode_data = dataset[dataset['label'] == mode]
        first_session = mode_data['session'].dropna().unique()[0]
        seg = mode_data[mode_data['session'] == first_session]
        signal = seg['gyr_x'].dropna().values
        N = len(signal)
        freqs = rfftfreq(N, d=1.0 / fs)
        amplitudes = np.abs(rfft(signal)) * 2.0 / N
        axes[1].plot(freqs, amplitudes, label=mode, alpha=0.8)

    axes[1].set_title('FFT of gyr_x per transport mode')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude (rad/s)')
    axes[1].legend()
    axes[1].set_xlim(0, fs / 2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fft_per_mode.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'fft_per_mode.pdf')
    plt.close(fig)
    print('Saved fft_per_mode')

    # ---- Figure 6: Summary statistics table ----
    print('\n=== Per-mode summary statistics ===')
    for mode in modes:
        mode_data = dataset[dataset['label'] == mode]
        print(f'\n--- {mode} ({len(mode_data)} samples) ---')
        print(mode_data[acc_cols + gyr_cols + ['bar_pressure']].describe().round(3).to_string())

    # ---- Figure 7: NaN percentage overview ----
    print('\n=== Missing values (NaN %) ===')
    total = len(dataset)
    for col in sensor_cols:
        n_missing = dataset[col].isnull().sum()
        print(f'  {col}: {n_missing}/{total} ({n_missing/total*100:.1f}%)')

# Save the dataset (drop the helper 'label' column, keep binary label columns)
dataset = dataset.drop(columns=['label'])
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

print('\nThe code has run through successfully!')
