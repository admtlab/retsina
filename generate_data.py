import pandas as pd
from src.ipin_feature_extractor import IPINFeatureExtractor
from src.jui_feature_extractor import JUIFeatureExtractor
from src.uji_loc_feature_extractor import UJILocFeatureExtractor
from src.uji_feature_extractor import UJIFeatureExtractor
from os import path
import sys
import getopt
from multiprocessing import Pool
from pathlib import Path


class DataConfig:
    """
    Due to the use of a multiprocessing queue and a process pool, this object simply
    acts as a wrapper class for encapsulating properties that should be passed to a single
    process when running in parallel
    """

    def __init__(self, dataset, fingerprints_to_use=1000, negative_label='Far', filter_distance_upper_bound=None,
                 filter_max_distance=None, filter_rssi_lower_bound=None, num_processes=1, custom_dataset_name=None,
                 custom_file_name=None):
        self.dataset = dataset
        self.fingerprints_to_use = fingerprints_to_use
        self.negative_label = negative_label
        self.filter_distance_upper_bound = filter_distance_upper_bound
        self.filter_max_distance = filter_max_distance
        self.filter_rssi_lower_bound = filter_rssi_lower_bound
        self.num_processes = num_processes
        self.custom_dataset_name = custom_dataset_name
        self.custom_file_name = custom_file_name


def generate_data(data_obj):
    """
    The main function for performing data generation (i.e., computing the features described
    in the IPIN 2023 paper) that interacts with the corresponding feature extractors under the
    src directory.
    :param data_obj: An instance of the DataConfig class containing specifications for computing new features
    """
    upper_bound_filename = "-filtered-range" if data_obj.filter_distance_upper_bound is not None else ""
    max_distance_filename = "-filtered-max-distance" if data_obj.filter_max_distance is not None else ""
    rssi_lower_bound_filename = "-filtered-rssi" if data_obj.filter_rssi_lower_bound is not None else ""
    train_file_name = 'trainingData.csv'
    test_file_name = 'testingData.csv'
    output_train_path = f'trainingData{upper_bound_filename}{max_distance_filename}{rssi_lower_bound_filename}.csv'
    output_test_path = f'testingData{upper_bound_filename}{max_distance_filename}{rssi_lower_bound_filename}.csv'
    match data_obj.dataset:
        case 9:
            print('Grouping data for UJIndoor')
            # Merge training dataset for UJIIndoor
            input_file_path = path.join('datasets', '09-UJI_LIB_DB_v2', 'db', '01', 'trn01')
            ext = UJIFeatureExtractor(input_file_path, '-g', True)
            ext.merge_files_to_single_csv(False)

            # Merge testing dataset for UJIndoor
            ext.merge_files_to_single_csv(True)

            print('Generating training dataset for UJIndoor')
            # Generate training dataset from merged data
            input_file_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00', 'merged-trainingData')
            output_file_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00', output_train_path)
            ext = UJIFeatureExtractor(input_file_path, '-g', False)
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)

            print('Generating testing dataset for UJIndoor')
            # Generate testing dataset from merged data
            input_file_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00', 'merged-testingData')
            output_file_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00', output_test_path)
            ext = UJIFeatureExtractor(input_file_path, '-g', False)
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)
        case 10:
            print('Generating testing dataset for JUIndoorLoc')
            # Generate Testing dataset for JUIndoorLoc as training is not used
            input_file_path = path.join('datasets', '10-JUIndoorLoc', 'JUIndoorLoc-Test-data.csv')
            output_file_path = path.join('output_datasets', '10-JUIndoorLoc', output_test_path)
            ext = JUIFeatureExtractor(input_file_path, '-g')
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)
        case 11:
            print('Generating training dataset for UJIndoorLoc')
            # Generate Training dataset for UJIndoorLoc
            input_file_path = path.join('datasets', '11-UJIndoorLoc', 'UJIndoorLoc', train_file_name)
            output_file_path = path.join('output_datasets', '11-UJIndoorLoc', output_train_path)
            ext = UJILocFeatureExtractor(input_file_path, '-g')
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)

            print('Generating testing dataset for UJIndoorLoc')
            # Generate Testing dataset for UJIndoorLoc
            input_file_path = path.join('datasets', '11-UJIndoorLoc', 'UJIndoorLoc', 'validationData.csv')
            output_file_path = path.join('output_datasets', '11-UJIndoorLoc', output_test_path)
            ext = UJILocFeatureExtractor(input_file_path, '-g')
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)
        case 13:
            print('Generating training dataset for IPIN Tutorial')
            # Generate Training dataset for IPIN Tutorial
            for i in range(1, 9):
                input_file_path = path.join('datasets', '13-IPIN-Tutorial', 'training', f'fingerprints_0{i}.csv')
                output_file_path = path.join('output_datasets', '13-IPIN-Tutorial', 'training',
                                             f'fingerprints_0{i}{upper_bound_filename}{max_distance_filename}{rssi_lower_bound_filename}.csv')
                ext = IPINFeatureExtractor(input_file_path)
                ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                      filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                      filter_max_distance=data_obj.filter_max_distance,
                                      filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                      num_processes=data_obj.num_processes)

            print('Generating testing dataset for IPIN Tutorial')
            # Generate Testing dataset for IPIN Tutorial
            input_file_path = path.join('datasets', '13-IPIN-Tutorial', 'testing', test_file_name)
            output_file_path = path.join('output_datasets', '13-IPIN-Tutorial', 'testing', output_test_path)
            ext = IPINFeatureExtractor(input_file_path)
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)
        case 8:
            print('Generating custom dataset')
            input_file_path = path.join('datasets', data_obj.custom_dataset_name, data_obj.custom_file_name)
            output_file_path = path.join('output_datasets', data_obj.custom_dataset_name, data_obj.custom_file_name)
            ext = IPINFeatureExtractor(input_file_path)
            ext.write_rows_to_csv(output_file_path, data_obj.fingerprints_to_use,
                                  filter_distance_upper_bound=data_obj.filter_distance_upper_bound,
                                  filter_max_distance=data_obj.filter_max_distance,
                                  filter_rssi_lower_bound=data_obj.filter_rssi_lower_bound,
                                  num_processes=data_obj.num_processes)
        case 0:
            file_path9 = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00')
            file_path10 = path.join('output_datasets', '10-JUIndoorLoc')
            file_path11 = path.join('output_datasets', '11-UJIndoorLoc')
            output_path = path.join('output_datasets', '00-AllDatasets')
            train_path = path.join(output_path, output_train_path)
            test_path = path.join(output_path, output_test_path)

            print('Reading Training CSVs and Combining into a single CSV file')
            df9 = pd.read_csv(path.join(file_path9, train_file_name))
            df11 = pd.read_csv(path.join(file_path11, train_file_name))
            joint_df = pd.concat([df9, df11])
            proximity_col = pd.DataFrame()
            proximity_col['proximity'] = (joint_df.proximity == data_obj.negative_label).astype(int)
            joint_df['proximity'] = proximity_col
            joint_df.to_csv(train_path)

            print('Reading Testing CSVs and combining into a single CSV file')
            df9 = pd.read_csv(path.join(file_path9, test_file_name))
            df10 = pd.read_csv(path.join(file_path10, test_file_name))
            df11 = pd.read_csv(path.join(file_path11, test_file_name))
            joint_df = pd.concat([df9, df10, df11])
            proximity_col = pd.DataFrame()
            proximity_col['proximity'] = (joint_df.proximity == data_obj.negative_label).astype(int)
            joint_df['proximity'] = proximity_col
            joint_df.to_csv(test_path)
        case _:
            print('Invalid dataset number')


if __name__ == '__main__':
    # If output directories don't exist, make them
    uji_dir_lst = [Path(Path(".") / "output_datasets" / "09-UJI_LIB_DB_v2" / f"0{x}") for x in range(0, 10)]
    for x in range(10, 26):
        uji_dir_lst.append(Path(Path(".") / "output_datasets" / "09-UJI_LIB_DB_v2" / f"{x}"))
    uji_dir_lst.extend([Path(Path(".") / "output_datasets" / "00-AllDatasets"),
                        Path(Path(".") / "output_datasets" / "10-JUIndoorLoc"),
                        Path(Path(".") / "output_datasets" / "11-UJIndoorLoc"),
                        Path(Path(".") / "output_datasets" / "13-IPIN-Tutorial" / "training"),
                        Path(Path(".") / "output_datasets" / "13-IPIN-Tutorial" / "testing")])
    for curr_dir in uji_dir_lst:
        if not Path(curr_dir).exists():
            Path(curr_dir).mkdir(parents=True, exist_ok=True)

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "d:f:u:m:r:p:cd:cf:",
                               ["dataset=", "fingerprints=", "upper=", "max=", "rssi=", "processes=", "custom_dataset=",
                                "custom_file="])
    dataset_to_use = 'all'
    fingerprints = 100
    distance_upper_bound = None
    max_distance = None
    rssi_bound = None
    number_of_processes = 1
    custom_dataset = None
    custom_file = None

    try:
        for opt, arg in opts:
            if opt in ('-d', '--dataset'):
                dataset_to_use = arg
            elif opt in ('-f', '--fingerprints'):
                fingerprints = int(arg)
            elif opt in ('-u', '--upper'):
                distance_upper_bound = float(arg)
            elif opt in ('-m', '--max'):
                max_distance = float(arg)
            elif opt in ('-r', '--rssi'):
                rssi_bound = float(arg)
            elif opt in ('-p', '--processes'):
                number_of_processes = int(arg)
            elif opt in ('-cd', '--custom_dataset'):
                custom_dataset = arg
            elif opt in ('-cf', '--custom_file'):
                custom_file = arg
    except getopt.GetoptError:
        print('Unknown argument specified\n')
        sys.exit(0)

    if dataset_to_use == 'all':
        datasets = [x for x in range(9, 14)] + [0]

        # Note that None here means no filter on range or max distance
        upper_bounds = [3.5]
        max_dists = [10.0]
        rssi_bounds = [-90]
        for ub in upper_bounds:
            for m in max_dists:
                for r in rssi_bounds:
                    with Pool(4) as p:
                        p.map(generate_data, [DataConfig(x, num_processes=number_of_processes) for x in datasets])
    elif dataset_to_use == 'custom':
        generate_data(
            DataConfig(int(8), fingerprints_to_use=fingerprints, filter_distance_upper_bound=distance_upper_bound,
                       filter_max_distance=max_distance, filter_rssi_lower_bound=rssi_bound,
                       num_processes=number_of_processes, custom_dataset_name=custom_dataset,
                       custom_file_name=custom_file))
    else:
        generate_data(DataConfig(int(dataset_to_use), fingerprints_to_use=fingerprints,
                                 filter_distance_upper_bound=distance_upper_bound, filter_max_distance=max_distance,
                                 filter_rssi_lower_bound=rssi_bound, num_processes=number_of_processes))
