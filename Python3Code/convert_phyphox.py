##############################################################
#                                                            #
#    Convert Phyphox sensor CSVs into the format expected    #
#    by CreateDataset.add_numerical_dataset() and            #
#    CreateDataset.add_event_dataset().                      #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
from pathlib import Path
import re

DATA_PATH = Path(__file__).parent / 'data'
OUTPUT_PATH = Path(__file__).parent / 'datasets' / 'our_data'

SENSOR_FILES = {
    ('Linear Accelerometer.csv', 'Accelerometer.csv'): {
        'output': 'accelerometer.csv',
        'rename': {
            'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z',
            'Acceleration x (m/s^2)': 'x', 'Acceleration y (m/s^2)': 'y', 'Acceleration z (m/s^2)': 'z',
        },
    },
    ('Gyroscope.csv',): {
        'output': 'gyroscope.csv',
        'rename': {
            'X (rad/s)': 'x', 'Y (rad/s)': 'y', 'Z (rad/s)': 'z',
            'Gyroscope x (rad/s)': 'x', 'Gyroscope y (rad/s)': 'y', 'Gyroscope z (rad/s)': 'z',
        },
    },
    ('Barometer.csv', 'Pressure.csv'): {
        'output': 'barometer.csv',
        'rename': {
            'X (hPa)': 'pressure',
            'Pressure (hPa)': 'pressure',
        },
    },
}

FOLDER_PATTERN = re.compile(r'^(.+)_(.+)_s(\d+)$')


def parse_session_folder(folder_name):
    m = FOLDER_PATTERN.match(folder_name)
    if not m:
        return None
    return {'person': m.group(1), 'mode': m.group(2), 'session': int(m.group(3))}


def get_start_time(session_path):
    time_csv = session_path / 'meta' / 'time.csv'
    meta = pd.read_csv(time_csv)
    start_row = meta[meta['event'] == 'START'].iloc[0]
    epoch_s = float(start_row['system time'])
    return pd.Timestamp(epoch_s, unit='s')


def convert_sensor(session_path, sensor_files, rename_map, start_time):
    filepath = None
    for sensor_file in sensor_files:
        candidate = session_path / sensor_file
        if candidate.exists():
            filepath = candidate
            break
    if filepath is None:
        return None

    df = pd.read_csv(filepath)
    time_col = 'Time (s)'

    df['timestamps'] = start_time + pd.to_timedelta(df[time_col], unit='s')
    df = df.drop(columns=[time_col])
    df = df.rename(columns=rename_map)

    return df


def main():
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    session_dirs = sorted([
        d for d in DATA_PATH.iterdir()
        if d.is_dir() and FOLDER_PATTERN.match(d.name)
    ])

    all_sensors = {info['output']: [] for info in SENSOR_FILES.values()}
    sessions_meta = []

    for session_dir in session_dirs:
        info = parse_session_folder(session_dir.name)
        print(f"Processing {session_dir.name}: person={info['person']}, mode={info['mode']}, session={info['session']}")

        start_time = get_start_time(session_dir)

        for sensor_file, sensor_info in SENSOR_FILES.items():
            df = convert_sensor(session_dir, sensor_file, sensor_info['rename'], start_time)
            if df is not None:
                all_sensors[sensor_info['output']].append(df)

        # Determine session time range from accelerometer (highest sample rate)
        acc_df = all_sensors['accelerometer.csv'][-1]
        t_start = acc_df['timestamps'].min()
        t_end = acc_df['timestamps'].max()

        sessions_meta.append({
            'label_start': t_start,
            'label_end': t_end,
            'label': info['mode'],
            'person': info['person'],
            'session_id': session_dir.name,
        })

    # Write concatenated sensor files
    for output_name, dfs in all_sensors.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(OUTPUT_PATH / output_name, index=False)
        print(f"Wrote {OUTPUT_PATH / output_name} ({len(combined)} rows)")

    # Write labels.csv for CreateDataset.add_event_dataset()
    labels_df = pd.DataFrame(sessions_meta)
    labels_df.to_csv(OUTPUT_PATH / 'labels.csv', index=False)
    print(f"Wrote {OUTPUT_PATH / 'labels.csv'} ({len(labels_df)} rows)")

    # Write sessions.csv for merging person/session back later
    sessions_df = pd.DataFrame(sessions_meta)
    sessions_df.to_csv(OUTPUT_PATH / 'sessions.csv', index=False)
    print(f"Wrote {OUTPUT_PATH / 'sessions.csv'}")

    print('\nDone! Converted files are in', OUTPUT_PATH)


if __name__ == '__main__':
    main()
