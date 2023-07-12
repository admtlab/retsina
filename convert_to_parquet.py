import pandas as pd
from multiprocessing import Pool
from os import path
from pathlib import Path
import dask.dataframe as dd
from src.parquet_merger import ParquetMerger

"""
A simple python script for converting the datasets used in the IPIN2023 paper
from CSV files to parquet files. The main reason was to allow for faster file
reading/processing through the use of the Dask Python library rather than 
sklearn's from_csv method.
"""


def convert_to_pq(csv_filepath, output_filepath):
    """
    A simple function for converting a Pandas DataFrame
    to a Dask DataFrame
    """
    df = pd.read_csv(csv_filepath)
    df.to_parquet(output_filepath)


def run_converter(dataset_to_convert, custom_dataset_name=None, custom_file_name=None):
    """
    The main function for providing filepaths and converting a
    CSV file to a parquet file. Note that a custom dataset following the
    format specified in the IPIN 2023 paper can be used by setting the
    dataset to 8 and providing a custom_dataset_name and custom_file_name
    """
    train_file_name = 'trainingData'
    test_file_name = 'testingData'
    base_dir = path.join('/', 'output_datasets')
    match dataset_to_convert:
        case 1:
            dataset_name = path.join(base_dir, '09-UJI_LIB_DB_v2', '00')
            convert_to_pq(path.join(dataset_name, f"{train_file_name}.csv"),
                          path.join(dataset_name, f"{train_file_name}.parquet"))
        case 2:
            dataset_name = path.join(base_dir, '09-UJI_LIB_DB_v2', '00')
            convert_to_pq(path.join(dataset_name, f"{test_file_name}.csv"),
                          path.join(dataset_name, f"{test_file_name}.parquet"))
        case 3:
            dataset_name = path.join(base_dir, '10-JUIndoorLoc')
            convert_to_pq(path.join(dataset_name, f"{test_file_name}.csv"),
                          path.join(dataset_name, f"{test_file_name}.parquet"))
        case 4:
            dataset_name = path.join(base_dir, '11-UJIndoorLoc')
            convert_to_pq(path.join(dataset_name, f"{train_file_name}.csv"),
                          path.join(dataset_name, f"{train_file_name}.parquet"))
        case 5:
            dataset_name = path.join(base_dir, '11-UJIndoorLoc')
            convert_to_pq(path.join(dataset_name, f"{test_file_name}.csv"),
                          path.join(dataset_name, f"{test_file_name}.parquet"))
        case 6:
            dataset_name = path.join(base_dir, '13-IPIN-Tutorial', 'training')
            for i in range(1, 9):
                train_file_name = f"fingerprints_0{i}"
                convert_to_pq(path.join(dataset_name, f"{train_file_name}.csv"),
                              path.join(dataset_name, f"{train_file_name}.parquet"))
        case 7:
            dataset_name = path.join(base_dir, '13-IPIN-Tutorial', 'testing')
            convert_to_pq(path.join(dataset_name, f"{test_file_name}.csv"),
                          path.join(dataset_name, f"{test_file_name}.parquet"))
        case 8:
            convert_to_pq(path.join(custom_dataset_name, f'{custom_file_name}.csv'),
                          path.join(custom_dataset_name, f'{custom_file_name}.parquet'))
        case _:
            print('[Warning]\tPlease use a valid experiment number')


def merge_ipin_training():
    """
    A simple helper function for merging the IPIN dataset files into a single
    training file
    """
    pq = ParquetMerger()
    base_dir = Path("/") / "output_datasets" / "13-IPIN-Tutorial" / "training"
    training_file_lst = []
    for i in range(1, 9):
        training_file_lst.append(base_dir / f"fingerprints_0{i}.parquet")

    training_df_lst = []
    with Pool(8) as process_pool:
        training_df_lst = process_pool.map(pq.compute_parquet_to_df, training_file_lst)

    merged_pq = dd.concat(training_df_lst, axis=0).compute().reset_index(drop=True)
    merged_pq.to_parquet(Path(base_dir / 'trainingData.parquet'))


if __name__ == "__main__":
    # Uncomment the line below to merge the IPIN dataset's training files
    # merge_ipin_training()

    # Uncomment the line below to convert a custom dataset, this example is for the IPIN testing dataset
    # run_converter(8, Path("/") / "output_datasets" / "13-IPIN-Tutorial" / "testing", 'testingData')

    # Convert each of the datasets to a parquet file in parallel
    datasets_to_convert = [x for x in range(1, 8)]
    with Pool(8) as p:
        p.map(run_converter, datasets_to_convert)
