##############################################################
#                                                            #
#    Adapted from crowdsignals_ch3_outliers.py for our       #
#    transportation mode detection dataset.                  #
#    Chapter 3: Outlier detection                            #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import sys
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

DATA_PATH = Path(__file__).parent / 'intermediate_datafiles'
DATASET_FNAME = 'chapter2_result.csv'
RESULT_FNAME = 'chapter3_result_outliers.csv'

SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'bar_pressure']


def main():

    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run our_ch2.py first!')
        raise e

    DataViz = VisualizeDataset(__file__)
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    if FLAGS.mode == 'chauvenet':
        for col in SENSOR_COLS:
            print(f"Applying Chauvenet outlier criteria for column {col}")
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            DataViz.plot_binary_outliers(dataset, col, col + '_outlier')

    elif FLAGS.mode == 'mixture':
        for col in SENSOR_COLS:
            print(f"Applying mixture model for column {col}")
            dataset = OutlierDistr.mixture_model(dataset, col)
            DataViz.plot_dataset(dataset,
                                 [col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])

    elif FLAGS.mode == 'distance':
        for col in SENSOR_COLS:
            try:
                dataset = OutlierDist.simple_distance_based(
                    dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
                DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
            except MemoryError as e:
                print('Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')

    elif FLAGS.mode == 'LOF':
        for col in SENSOR_COLS:
            try:
                dataset = OutlierDist.local_outlier_factor(
                    dataset, [col], 'euclidean', FLAGS.K)
                DataViz.plot_dataset(dataset, [col, 'lof'],
                                     ['exact', 'exact'], ['line', 'points'])
            except MemoryError as e:
                print('Not enough memory available for lof...')
                print('Skipping.')

    elif FLAGS.mode == 'final':

        FIGURES_DIR = Path('figures') / 'our_ch3_outliers'
        FIGURES_DIR.mkdir(exist_ok=True, parents=True)

        dataset_before = copy.deepcopy(dataset)
        outlier_counts = {}

        for col in [c for c in dataset.columns if c in SENSOR_COLS]:
            print(f'Measurement is now: {col}')
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            n_outliers = dataset[f'{col}_outlier'].sum()
            outlier_counts[col] = n_outliers
            print(f'  -> {n_outliers} outliers found ({n_outliers/len(dataset)*100:.2f}%)')
            dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
            del dataset[col + '_outlier']

        # Figure: show outliers on acc_y (the column with most outliers typically)
        show_col = max(outlier_counts, key=outlier_counts.get)
        first_session = dataset['session'].dropna().unique()[0]
        mask = dataset['session'] == first_session
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = (dataset.index[mask] - dataset.index[mask][0]).total_seconds()

        axes[0].plot(t, dataset_before.loc[mask, show_col], 'b-', linewidth=0.8)
        axes[0].set_title(f'{show_col} — before outlier removal')
        axes[0].set_ylabel('Value')

        before_vals = dataset_before.loc[mask, show_col]
        after_vals = dataset.loc[mask, show_col]
        outlier_mask = before_vals.notna() & after_vals.isna()
        axes[1].plot(t, after_vals, 'b-', linewidth=0.8, label='kept')
        axes[1].plot(t[outlier_mask], before_vals[outlier_mask], 'rx', markersize=8, label='outlier (removed)')
        axes[1].set_title(f'{show_col} — after Chauvenet outlier removal (C={FLAGS.C})')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend(fontsize='small')

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'outlier_before_after.png', dpi=150)
        fig.savefig(FIGURES_DIR / 'outlier_before_after.pdf')
        plt.close(fig)
        print('Saved outlier_before_after')

        # Print summary table
        print('\n=== Outlier summary (Chauvenet, C={}) ==='.format(FLAGS.C))
        for col, count in outlier_counts.items():
            print(f'  {col}: {count} outliers ({count/len(dataset)*100:.2f}%)')

        dataset.to_csv(DATA_PATH / RESULT_FNAME)
        print(f'Saved to {DATA_PATH / RESULT_FNAME}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final",
                        choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")

    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor: K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based: dmin")

    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based: fmin")

    FLAGS, unparsed = parser.parse_known_args()

    main()
