##############################################################
#                                                            #
#    Adapted from crowdsignals_ch3_rest.py for our           #
#    transportation mode detection dataset.                  #
#    Chapter 3: Imputation + Low-pass filtering              #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues

DATA_PATH = Path(__file__).parent / 'intermediate_datafiles'
DATASET_FNAME = 'chapter3_result_outliers.csv'
RESULT_FNAME = 'chapter3_result_final.csv'
ORIG_DATASET_FNAME = 'chapter2_result.csv'

SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'bar_pressure']
PERIODIC_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']


def main():

    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run our_ch3_outliers.py first!')
        raise e

    DataViz = VisualizeDataset(__file__)

    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()

    if FLAGS.mode == 'imputation':
        imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'bar_pressure')
        imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'bar_pressure')
        imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'bar_pressure')

        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'bar_pressure',
                                    imputed_mean_dataset['bar_pressure'],
                                    imputed_median_dataset['bar_pressure'],
                                    imputed_interpolation_dataset['bar_pressure'])

    elif FLAGS.mode == 'lowpass':
        fs = float(1000) / milliseconds_per_instance
        cutoff = 1.5

        new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'acc_x', fs, cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :],
                             ['acc_x', 'acc_x_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'final':

        FIGURES_DIR = Path('figures') / 'our_ch3_rest'
        FIGURES_DIR.mkdir(exist_ok=True, parents=True)

        # Keep a copy before processing for before/after comparison
        dataset_before = copy.deepcopy(dataset)

        # --- Step 1: Impute missing values using interpolation ---
        for col in [c for c in dataset.columns if c in SENSOR_COLS]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # --- Figure 1: Barometer imputation before/after ---
        # Pick one session to show the effect clearly
        first_session = dataset['session'].dropna().unique()[0]
        mask = dataset['session'] == first_session
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = (dataset.index[mask] - dataset.index[mask][0]).total_seconds()

        bar_before = dataset_before.loc[mask, 'bar_pressure']
        t_valid = t[bar_before.notna()]
        axes[0].plot(t_valid, bar_before.dropna(), 'bo', markersize=3)
        axes[0].set_title('Barometer — before imputation')
        axes[0].set_ylabel('Pressure (hPa)')

        axes[1].plot(t, dataset.loc[mask, 'bar_pressure'], 'b-', linewidth=0.8)
        axes[1].set_title('Barometer — after interpolation')
        axes[1].set_ylabel('Pressure (hPa)')
        axes[1].set_xlabel('Time (s)')

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'barometer_imputation.png', dpi=150)
        fig.savefig(FIGURES_DIR / 'barometer_imputation.pdf')
        plt.close(fig)
        print('Saved barometer_imputation')

        # --- Step 2: Low-pass filter ---
        fs = float(1000) / milliseconds_per_instance
        cutoff = 1.5
        dataset_pre_filter = copy.deepcopy(dataset)

        for col in PERIODIC_COLS:
            dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']

        # --- Figure 2: Accelerometer before/after low-pass (one session, 30s window) ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        seg_pre = dataset_pre_filter[mask].iloc[:120]
        seg_post = dataset[mask].iloc[:120]
        t_seg = (seg_pre.index - seg_pre.index[0]).total_seconds()

        axes[0].plot(t_seg, seg_pre['acc_x'], 'b-', linewidth=0.8, label='acc_x')
        axes[0].plot(t_seg, seg_pre['acc_y'], 'g-', linewidth=0.8, label='acc_y')
        axes[0].plot(t_seg, seg_pre['acc_z'], 'r-', linewidth=0.8, label='acc_z')
        axes[0].set_title('Accelerometer — before low-pass filter (first 30s)')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].legend(loc='upper right', fontsize='small')

        axes[1].plot(t_seg, seg_post['acc_x'], 'b-', linewidth=0.8, label='acc_x')
        axes[1].plot(t_seg, seg_post['acc_y'], 'g-', linewidth=0.8, label='acc_y')
        axes[1].plot(t_seg, seg_post['acc_z'], 'r-', linewidth=0.8, label='acc_z')
        axes[1].set_title(f'Accelerometer — after low-pass filter (cutoff={cutoff} Hz)')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend(loc='upper right', fontsize='small')

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'lowpass_before_after.png', dpi=150)
        fig.savefig(FIGURES_DIR / 'lowpass_before_after.pdf')
        plt.close(fig)
        print('Saved lowpass_before_after')

        # --- Figure 3: Final processed dataset overview (one session per mode) ---
        dataset['label'] = 'unknown'
        for col in [c for c in dataset.columns if c.startswith('label') and c != 'label']:
            mode = col[len('label'):]
            dataset.loc[dataset[col] == 1, 'label'] = mode
        modes = [m for m in dataset['label'].unique() if m != 'unknown']

        fig, axes = plt.subplots(len(modes), 1, figsize=(12, 4 * len(modes)), sharex=False)
        if len(modes) == 1:
            axes = [axes]
        for ax, mode in zip(axes, modes):
            mode_data = dataset[dataset['label'] == mode]
            session = mode_data['session'].dropna().unique()[0]
            seg = mode_data[mode_data['session'] == session]
            t = (seg.index - seg.index[0]).total_seconds()
            ax.plot(t, seg['acc_x'], label='acc_x', alpha=0.8)
            ax.plot(t, seg['acc_y'], label='acc_y', alpha=0.8)
            ax.plot(t, seg['acc_z'], label='acc_z', alpha=0.8)
            ax.set_title(f'Final processed accelerometer — {mode}')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.set_xlabel('Time (s)')
            ax.legend(loc='upper right', fontsize='small')
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'final_acc_per_mode.png', dpi=150)
        fig.savefig(FIGURES_DIR / 'final_acc_per_mode.pdf')
        plt.close(fig)
        print('Saved final_acc_per_mode')

        dataset = dataset.drop(columns=['label'])
        dataset.to_csv(DATA_PATH / RESULT_FNAME)
        print(f'Saved to {DATA_PATH / RESULT_FNAME}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, or lowpass",
                        choices=['lowpass', 'imputation', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
