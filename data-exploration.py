import pandas as pd
from os import path
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
import getopt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

"""
A simple Python script for performing basic data exploration on the datasets
used in the IPIN 2023 paper to analyze the ranges of each dataset's features
as well as the number of detected APs and how often each specific AP was detected.
"""


def get_dataset_stats(stat_run_number):
    train_file_name = 'trainingData.csv'
    test_file_name = 'testingData.csv'

    # bookkeeping for output
    max_rssi = -121
    min_rssi = 1

    match stat_run_number:
        case 1:

            filepath = path.join('datasets', '11-UJIndoorLoc', 'UJIndoorLoc', train_file_name)
            output_directory = path.join('output_stats', '11-UJIndoorLoc', 'training')

            curr_df = pd.read_csv(filepath)

            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['PHONEID'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

            grouped_df = curr_df.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-space-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID',
                 'TIMESTAMP'])]
            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != 100][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 2:
            filepath = path.join('datasets', '11-UJIndoorLoc', 'UJIndoorLoc', 'validationData.csv')
            output_directory = path.join('output_stats', '11-UJIndoorLoc', 'testing')

            curr_df = pd.read_csv(filepath)

            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['PHONEID'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

            grouped_df = curr_df.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-space-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID',
                 'TIMESTAMP'])]
            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != 100][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 3:
            filepath = path.join('datasets', '10-JUIndoorLoc', 'JUIndoorLoc-Test-data.csv')
            output_directory = path.join('output_stats', '10-JUIndoorLoc', 'testing')

            curr_df = pd.read_csv(filepath)
            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['Did'].value_counts().to_csv(path.join(output_directory, 'device-counts.csv'))

            grouped_df = curr_df.groupby('Cid')
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-cell-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(['Cid', 'Rs', 'Hpr', 'Did', 'Ts'])]
            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != -110][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 4:
            for i in range(1, 9):
                filepath = path.join('datasets', '13-IPIN-Tutorial', 'training', f'fingerprints_0{i}.csv')
                output_directory = path.join('output_stats', '13-IPIN-Tutorial', 'training', f'fingerprints_0{i}')

                curr_df = pd.read_csv(filepath)
                curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

                # Categorical Features
                curr_df['PHONEID'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

                grouped_df = curr_df.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])
                grouped_df.size().to_csv(path.join(output_directory, 'grouped-space-counts.csv'))

                # Extract Timestamp Data and Duration
                start_time = datetime.fromtimestamp(curr_df.head(1)['TIMESTAMP'].squeeze())
                end_time = datetime.fromtimestamp(curr_df.tail(1)['TIMESTAMP'].squeeze())
                duration = end_time - start_time
                with open(path.join(output_directory, 'durations.txt'), 'w') as duration_file:
                    duration_file.write(f'Start Time: {start_time}\nEnd Time: {end_time}\nDuration: {duration}\n')

                # Spatiotemporal plot
                fig = plt.figure()
                spatial_temp_plot = fig.add_subplot(projection='3d')
                spatial_temp_plot.scatter(curr_df['TIMESTAMP'], curr_df['lat'], curr_df['lon'])
                spatial_temp_plot.set_xlabel('Timestamp (Unix Epoch)')
                spatial_temp_plot.set_ylabel('Latitude')
                spatial_temp_plot.set_zlabel('Longitude')
                # curr_df[['lat', 'lon']].plot(y='lon')
                plt.savefig(path.join(output_directory, 'space-time-plot.png'))
                plt.close()

                # RSSI Features
                ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                    ['lon', 'lat', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID',
                     'TIMESTAMP'])]
                with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                    for column in ap_df:
                        temp_df = ap_df[ap_df[column] != 100][column]
                        max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                        min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                        ap_file.write(f'{temp_df.describe()}\n\n')

                    # write globals after each column
                    ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 5:
            filepath = path.join('datasets', '13-IPIN-Tutorial', 'testing', test_file_name)
            output_directory = path.join('output_stats', '13-IPIN-Tutorial', 'testing')

            curr_df = pd.read_csv(filepath)
            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['PHONEID'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

            grouped_df = curr_df.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-space-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                ['lon', 'lat', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID',
                 'TIMESTAMP'])]
            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != 100][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 6:
            filepath = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00',
                                 f'merged-{train_file_name[:-4]}-grouped.csv')
            output_directory = path.join('output_stats', '09-UJI_LIB_DB_v2', 'training')

            curr_df = pd.read_csv(filepath)
            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['phone'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

            grouped_df = curr_df.groupby('floor')
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-floor-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                ['id', 'lat', 'lon', 'floor', 'timestamp', 'phone'])]
            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != 100][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')

        case 7:
            filepath = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00',
                                 f'merged-{test_file_name[:-4]}-grouped.csv')
            output_directory = path.join('output_stats', '09-UJI_LIB_DB_v2', 'testing')

            curr_df = pd.read_csv(filepath)
            curr_df.describe().to_csv(path.join(output_directory, 'described.csv'))

            # Categorical Features
            curr_df['phone'].value_counts().to_csv(path.join(output_directory, 'phone-counts.csv'))

            grouped_df = curr_df.groupby('floor')
            grouped_df.size().to_csv(path.join(output_directory, 'grouped-floor-counts.csv'))

            # RSSI Features
            ap_df = curr_df.loc[:, ~curr_df.columns.isin(
                ['id', 'lat', 'lon', 'floor', 'timestamp', 'phone'])]

            with open(path.join(output_directory, 'AP-stats.txt'), 'w') as ap_file:
                for column in ap_df:
                    temp_df = ap_df[ap_df[column] != 100][column]
                    max_rssi = temp_df.max() if temp_df.max() > max_rssi else max_rssi
                    min_rssi = temp_df.min() if temp_df.min() < min_rssi else min_rssi
                    ap_file.write(f'{temp_df.describe()}\n\n')

                # write globals after each column
                ap_file.write(f'Global Max and Min:\nMax:{max_rssi}\nMin:{min_rssi}')
        case _:
            print('[Warning]\tPlease use a valid data exploration number')


if __name__ == '__main__':
    # If output directories don't exist, make them
    dir_lst = [Path(Path(".") / "output_stats" / "09-UJI_LIB_DB_v2" / "training"),
               Path(Path(".") / "output_stats" / "09-UJI_LIB_DB_v2" / "testing"),
               Path(Path(".") / "output_stats" / "10-JUIndoorLoc" / "testing"),
               Path(Path(".") / "output_stats" / "11-UJIndoorLoc" / "training"),
               Path(Path(".") / "output_stats" / "11-UJIndoorLoc" / "testing"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_01"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_02"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_03"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_04"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_05"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_06"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_07"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "training" / "fingerprints_08"),
               Path(Path(".") / "output_stats" / "13-IPIN-Tutorial" / "testing")]
    for curr_dir in dir_lst:
        if not Path(curr_dir).exists():
            Path(curr_dir).mkdir(parents=True, exist_ok=True)

    # allow user to set number of processes when running
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "p:", ["processes="])

    # default to serial execution
    number_of_processes = 1

    try:
        for opt, arg in opts:
            if opt in ('-p', '--processes'):
                number_of_processes = int(arg)

    except getopt.GetoptError:
        print("Unknown argument specified\n")
        sys.exit(0)

    # Perform basic data exploration on each dataset in parallel
    with Pool(number_of_processes) as p:
        p.map(get_dataset_stats, [x for x in range(1, 8)])
