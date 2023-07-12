from multiprocessing import Pool
from pathlib import Path
import dask.dataframe as dd
import pandas as pd

"""
A Python class for merging the parquet files from the datasets used in the
IPIN 2023 paper together into a single parquet file for experiments that
required combining the datasets together.
"""

class ParquetMerger:

    @staticmethod
    def parquet_file_exists(pq_directory, pq_filename):
        """
        Helper function for checking if parquet file existed
        before merging or reading anything. If the file does not exist,
        the path is created.
        :param pq_directory: the path/directory for the output parquet file
        :param pq_filename: the filename for the output parquet file
        :return: True if the parquet file already existed, false if the
        file needed to be created and cannot be reused
        """
        if Path(pq_directory).exists():
            if Path(pq_directory / pq_filename).exists():
                return True
        else:
            # If the path doesn't exist, create it and return false
            Path(pq_directory).mkdir(parents=True, exist_ok=True)
            return False

    @staticmethod
    def compute_parquet_to_df(filepath):
        """
        A helper function for reading a parquet file to a dask DataFrame
        :param filepath: Path for reading a parquet file from
        :return: A dask dataframe that supports parallelized reading and computation
        """
        return dd.read_parquet(filepath).compute()

    @staticmethod
    def random_parquet_sampling(x_df, negative_label="Far", close_samples=9000, far_samples=8000, random_state=29):
        """
        A function that randomly samples the rows of x_df for the given number of close and far samples using
        the provided random_state.
        :param x_df: A dataframe containing the dataset to be randomly sampled
        :param negative_label: The negative label to use for distinguishing close and far (in our case this was 'Far')
        :param close_samples: The number of close samples to sample for
        :param far_samples: The number of far samples to sample for
        :param random_state: The random state/seed to use when randomly sampling
        :return: A dataframe containing the randomly sampled rows based on the provided parameters with a reset index.
        """
        far_df = x_df.query(f"proximity == '{negative_label}'")
        close_df = x_df.query(f"proximity != '{negative_label}'")
        with_replacement = False

        # Verify that each DataFrame has sufficient rows for samples
        if far_df.shape[0] < far_samples:
            print(
                f"[Warning]\tRequested {far_samples} Far samples, but DF only has {far_df.shape[0]} samples")
            print("[Warning]\tUsing replacement of the samples from Far for requested number of samples")
            with_replacement = True

        if close_df.shape[0] < close_samples:
            print(
                f"[Warning]\tRequested {close_samples} Close samples, but DF only has {close_df.shape[0]} samples")
            print("[Warning]\tUsing replacement of the samples from Close for requested number of samples")
            with_replacement = True

        print("Sampling Far Datapoints\n")
        far_df = far_df.sample(n=far_samples, random_state=random_state, replace=with_replacement)
        print("Sampling Close Datapoints\n")
        close_df = close_df.sample(n=close_samples, random_state=random_state, replace=with_replacement)
        print("Combining Sampled Datasets into one DataFrame")
        return (pd.concat([far_df, close_df], axis=0)).drop_duplicates().reset_index(drop=True)

    def write_merged_parquet(self, filepath_lst, pq_dir, pq_filename, number_of_processes=1, negative_label="Far",
                             close_samples=9000,
                             far_samples=8000, random_state=29):
        """
        Main function to be called from this file which sets up a parquet file if needed
        :param filepath_lst: List of separate parquet files to merge
        :param pq_dir: the path/directory for the output parquet file
        :param pq_filename: the filename for the output parquet file
        :param number_of_processes: processes to use for parallelism
        :param negative_label: The label to use for the negative class
        :param close_samples: The number of close samples to sample
        :param far_samples: The number of Far samples to sample
        :param random_state: The random state to use for random sampling
        """
        # If parquet file already exists, simply read and return computed result
        if not self.parquet_file_exists(pq_dir, pq_filename):
            # If parquet file does not exist, read individual parquet files to concatenate together
            sampled_df_lst = []
            with Pool(number_of_processes) as p:
                sampled_df_lst = p.map(self.compute_parquet_to_df, filepath_lst)
                sample_input_lst = []
                for curr_df in sampled_df_lst:
                    sample_input_lst.append((curr_df, negative_label, close_samples, far_samples, random_state))

                sampled_df_lst = p.starmap(self.random_parquet_sampling, sample_input_lst)

            merged_pq = dd.concat(sampled_df_lst, axis=0).compute().reset_index(drop=True)

            # Write to file for future usage
            merged_pq.to_parquet(Path(pq_dir / pq_filename))
