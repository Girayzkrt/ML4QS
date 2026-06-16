##############################################################
#                                                            #
#    Adapted from crowdsignals_ch4.py for our Phyphox        #
#    transportation mode detection dataset.                  #
#    Chapter 4: Feature engineering                          #
#                                                            #
##############################################################

import sys
import argparse
import copy
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction


DATA_PATH = Path(__file__).parent / 'intermediate_datafiles'
DATASET_FNAME = 'chapter3_result_final.csv'
RESULT_FNAME = 'chapter4_result.csv'

SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'bar_pressure']
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
GYR_COLS = ['gyr_x', 'gyr_y', 'gyr_z']
DERIVED_COLS = ['acc_mag', 'gyr_mag', 'acc_jerk_mag', 'gyr_jerk_mag']
TEMPORAL_AGGREGATIONS = ['mean', 'std', 'min', 'max', 'slope']


def read_dataset():
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset
    except IOError as e:
        print('File not found, try to run our_ch3_rest.py first!')
        raise e


def infer_sampling_interval_seconds(dataset):
    deltas = []
    groups = [dataset]

    if 'session' in dataset.columns:
        groups = [group for _, group in dataset.groupby('session', sort=False)]

    for group in groups:
        group_deltas = group.index.to_series().diff().dropna().dt.total_seconds()
        deltas.extend(group_deltas[group_deltas > 0].tolist())

    if not deltas:
        raise ValueError('Could not infer the sampling interval from the datetime index.')

    return float(np.median(deltas))


def add_magnitude_features(dataset, sampling_interval_seconds):
    dataset = copy.deepcopy(dataset)

    # Magnitudes reduce the dependency on the exact phone orientation in the pocket.
    dataset['acc_mag'] = np.sqrt((dataset[ACC_COLS] ** 2).sum(axis=1))
    dataset['gyr_mag'] = np.sqrt((dataset[GYR_COLS] ** 2).sum(axis=1))

    # Jerk is computed within sessions so new recordings do not inherit previous movement.
    if 'session' in dataset.columns:
        acc_diff = dataset.groupby('session', sort=False)[ACC_COLS].diff()
        gyr_diff = dataset.groupby('session', sort=False)[GYR_COLS].diff()
    else:
        acc_diff = dataset[ACC_COLS].diff()
        gyr_diff = dataset[GYR_COLS].diff()

    acc_diff = acc_diff.fillna(0) / sampling_interval_seconds
    gyr_diff = gyr_diff.fillna(0) / sampling_interval_seconds

    dataset['acc_jerk_mag'] = np.sqrt((acc_diff ** 2).sum(axis=1))
    dataset['gyr_jerk_mag'] = np.sqrt((gyr_diff ** 2).sum(axis=1))

    return dataset


def apply_per_session(dataset, transform):
    if 'session' not in dataset.columns:
        return transform(copy.deepcopy(dataset).sort_index())

    transformed_sessions = []
    for _, session_data in dataset.groupby('session', sort=False):
        transformed_sessions.append(transform(copy.deepcopy(session_data).sort_index()))

    return pd.concat(transformed_sessions).sort_index()


def add_temporal_features(dataset, predictor_cols, temporal_windows_seconds):
    def transform(session_data):
        num_abs = NumericalAbstraction()
        for window_size in temporal_windows_seconds:
            for aggregation in TEMPORAL_AGGREGATIONS:
                session_data = num_abs.abstract_numerical(
                    session_data, predictor_cols, int(window_size), aggregation)
        return session_data

    return apply_per_session(dataset, transform)


def add_frequency_features(dataset, predictor_cols, frequency_window_seconds, sampling_rate):
    window_size = int(round(frequency_window_seconds * sampling_rate))

    def transform(session_data):
        freq_abs = FourierTransformation()
        return freq_abs.abstract_frequency(session_data, predictor_cols, window_size, sampling_rate)

    return apply_per_session(dataset, transform), window_size


def reduce_window_overlap(dataset, skip_points):
    if skip_points <= 1:
        return dataset

    if 'session' not in dataset.columns:
        return dataset.iloc[::skip_points, :]

    reduced_sessions = []
    for _, session_data in dataset.groupby('session', sort=False):
        reduced_sessions.append(session_data.iloc[::skip_points, :])

    return pd.concat(reduced_sessions).sort_index()


def print_summary(dataset, sampling_rate, frequency_window_size, skip_points):
    label_cols = [col for col in dataset.columns if col.startswith('label')]
    temporal_features = [col for col in dataset.columns if '_temp_' in col]
    frequency_features = [
        col for col in dataset.columns if ('_freq' in col) or ('_pse' in col)
    ]
    metadata_cols = label_cols + ['person', 'session']
    feature_cols = [col for col in dataset.columns if col not in metadata_cols]

    print('\n=== Chapter 4 feature engineering summary ===')
    print(f'Sampling rate: {sampling_rate:.2f} Hz')
    print(f'Frequency window: {frequency_window_size} samples')
    print(f'Downsampling step after overlap reduction: {skip_points} rows')
    print(f'Rows after feature engineering: {len(dataset)}')
    print(f'Total feature columns: {len(feature_cols)}')
    print(f'Temporal feature columns: {len(temporal_features)}')
    print(f'Frequency feature columns: {len(frequency_features)}')

    if label_cols:
        print('\nLabel counts after overlap reduction:')
        for col in label_cols:
            print(f'  {col}: {int(dataset[col].sum())}')


def main():
    start_time = time.time()
    dataset = read_dataset()

    if FLAGS.window_overlap < 0 or FLAGS.window_overlap >= 1:
        raise ValueError('window-overlap should be at least 0 and below 1.')

    sampling_interval_seconds = infer_sampling_interval_seconds(dataset)
    sampling_rate = 1.0 / sampling_interval_seconds

    temporal_windows_seconds = FLAGS.temporal_windows
    frequency_window_seconds = FLAGS.frequency_window

    dataset = add_magnitude_features(dataset, sampling_interval_seconds)

    temporal_predictor_cols = SENSOR_COLS + DERIVED_COLS
    periodic_predictor_cols = ACC_COLS + GYR_COLS + ['acc_mag', 'gyr_mag']

    dataset = add_temporal_features(dataset, temporal_predictor_cols, temporal_windows_seconds)
    dataset, frequency_window_size = add_frequency_features(
        dataset, periodic_predictor_cols, frequency_window_seconds, sampling_rate)

    dataset = dataset.dropna()

    skip_points = int(round(frequency_window_size * (1 - FLAGS.window_overlap)))
    skip_points = max(1, skip_points)
    dataset = reduce_window_overlap(dataset, skip_points)

    dataset.to_csv(DATA_PATH / RESULT_FNAME)

    print_summary(dataset, sampling_rate, frequency_window_size, skip_points)
    print(f'\nSaved to {DATA_PATH / RESULT_FNAME}')
    print('--- %s seconds ---' % (time.time() - start_time))


FIGURES_DIR = Path('figures') / 'our_ch4'


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
        f.write(f'# our_ch4.py — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
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

    parser.add_argument('--temporal-windows', type=int, nargs='+', default=[5, 10],
                        help='Temporal abstraction windows in seconds.')
    parser.add_argument('--frequency-window', type=int, default=10,
                        help='Frequency abstraction window in seconds.')
    parser.add_argument('--window-overlap', type=float, default=0.5,
                        help='Allowed overlap between consecutive feature windows.')

    FLAGS, unparsed = parser.parse_known_args()

    with save_output(FIGURES_DIR / 'output.txt'):
        main()
