from os import path
import csv
import src.feature_utils as utils
import json
from src.FeatureExtractorInterface import FeatureExtractorInterface

"""
UJI Feature Extractor: Responsible for processing and
parsing the data from each dataset and converting
the data into the features described in the paper by
Zach Van Hyfte and Avideh Zakhor (IPIN2021)

This feature extractor is specifically for handling dataset [9]
<Long-Term Wi-Fi fingerprinting dataset for robust indoor positioning,
G.M. Mendoza-Silva et al., 2017> and available at Zenodo repository, DOI 10.5281/zenodo.1066041
"""


class UJIFeatureExtractor(FeatureExtractorInterface):

    def __init__(self, file_location, mode='-g', grouping_data=False, fingerprint_seed=29, undetected_rssi='100'):
        """
        Note that the expectation is file_location is the base of the
        file path. E.g., trn01 as this is the base path for accessing
        the ID, coordinate, timestamp, and rss files for training file 01.
        The data collection month refers to the month the dataset belongs to
        e.g., 01 (as a string) for the first month.
        """
        # Call Super class with multiple_files set to True for overriding
        super(UJIFeatureExtractor, self).__init__(file_location, undetected_rssi, fingerprint_seed, True)

        self.mode = mode
        self.grouping_data = grouping_data

        if mode == '-g' and grouping_data:
            # Create a file for each of the components: ID, coordinates, RSS, timestamp
            id_file = path.join(file_location + 'ids.csv')
            crd_file = path.join(file_location + 'crd.csv')
            rss_file = path.join(file_location + 'rss.csv')
            tms_file = path.join(file_location + 'tms.csv')

            # Create a CSV reader for each of the above files
            self.id_csvfile = open(id_file)
            self.crd_csvfile = open(crd_file)
            self.rss_csvfile = open(rss_file)
            self.tms_csvfile = open(tms_file)
            self.id_csvreader = csv.reader(self.id_csvfile, delimiter=',')
            self.crd_csvreader = csv.reader(self.crd_csvfile, delimiter=',')
            self.rss_csvreader = csv.reader(self.rss_csvfile, delimiter=',')
            self.tms_csvreader = csv.reader(self.tms_csvfile, delimiter=',')

            self._extract_rows()
        elif mode == '-g' and not grouping_data:
            # Create a file for the grouped csv
            self.filepath = path.join(file_location + '-grouped.csv')
            self.csvfile = open(self.filepath)
            self.csvreader = csv.reader(self.csvfile, delimiter=',')

            # Use separate extract rows that reads from grouped file
            self._extract_rows()
        else:
            # Create a file for the features csv, note this path wants filename.csv
            self.filepath = path.join(file_location)
            self.csvfile = open(self.filepath)
            self.csvreader = csv.reader(self.csvfile, delimiter=',')
            # Note that rows aren't extracted as this uses columns for DataFrame

    def get_data_df(self):
        return utils.get_data_df(self.filepath)

    def _extract_column(self, col_num, rows_to_skip=1):
        """
        A function for extracting the values of a single column
        as a Python List.
        Note that the column system used below is due to the
        separation of data into multiple files by the dataset.
            Column: 0 => ID
            Columns: 1-3 => lat, lon, floor
            Column: 4 => timestamp
            Columns: 5-625 => RSS values
        """
        if self.mode == '-g' and self.grouping_data:
            values = []
            self._reset_csv_readers()
            if col_num == 0:
                for line in self.id_csvreader:
                    values.append(line[0])
            elif 1 <= col_num <= 3:
                for line in self.crd_csvreader:
                    # Note -1 to handle offset
                    values.append(line[col_num - 1])
            elif col_num == 4:
                for line in self.tms_csvreader:
                    values.append(line[0])
            elif 5 <= col_num <= 625:
                # Note that there are 620 APs in dataset, -5 to handle offset
                for line in self.rss_csvreader:
                    values.append(line[col_num - 5])
            else:
                print('Invalid Column number')

            return values
        else:
            return super()._extract_column(col_num, rows_to_skip)

    def _extract_rows(self, num_features=625, rows_to_skip=1):
        self.rows = []
        self._reset_csv_readers()

        if self.mode == '-g' and self.grouping_data:
            # First process the ID file
            for line in self.id_csvreader:
                id_to_add = {'id': line[0]}
                self.rows.append(id_to_add)

            row_counter = 0
            # Next process the Coordinates file
            for line in self.crd_csvreader:
                crds_to_add = []
                for x in range(1, 4):
                    crds_to_add.append(line[x - 1])
                self.rows[row_counter]['coords'] = crds_to_add
                row_counter += 1

            row_counter = 0
            # Next process the timestamp file
            for line in self.tms_csvreader:
                self.rows[row_counter]['timestamp'] = line[0]
                row_counter += 1

            row_counter = 0
            # Finally process the RSS file
            for line in self.rss_csvreader:
                self.rows[row_counter]['rss'] = line
                row_counter += 1
        else:
            super()._extract_rows(num_features, rows_to_skip)

    @staticmethod
    def _extract_merged_rows(file_location):
        """
        Takes in a filename for the merge_to_single_csv method to
        avoid changing the state of the extractor repeatedly. Also
        avoids the problem of updating the extractor's csv reader on
        each iteration.
        """
        # Set CSV files and readers
        id_file = path.join(file_location + 'ids.csv')
        crd_file = path.join(file_location + 'crd.csv')
        rss_file = path.join(file_location + 'rss.csv')
        tms_file = path.join(file_location + 'tms.csv')

        # Create a CSV reader for each of the above files
        id_csvfile = open(id_file)
        crd_csvfile = open(crd_file)
        rss_csvfile = open(rss_file)
        tms_csvfile = open(tms_file)
        id_csvreader = csv.reader(id_csvfile, delimiter=',')
        crd_csvreader = csv.reader(crd_csvfile, delimiter=',')
        rss_csvreader = csv.reader(rss_csvfile, delimiter=',')
        tms_csvreader = csv.reader(tms_csvfile, delimiter=',')

        # Extract the rows from the CSV file
        rows = []
        # First process the ID file
        for line in id_csvreader:
            id_to_add = {'id': line[0]}
            rows.append(id_to_add)

        row_counter = 0
        # Next process the Coordinates file
        for line in crd_csvreader:
            crds_to_add = []
            for x in range(1, 4):
                crds_to_add.append(line[x - 1])
            rows[row_counter]['coords'] = crds_to_add
            row_counter += 1

        row_counter = 0
        # Next process the timestamp file
        for line in tms_csvreader:
            rows[row_counter]['timestamp'] = line[0]
            row_counter += 1

        row_counter = 0
        # Finally process the RSS file
        for line in rss_csvreader:
            rows[row_counter]['rss'] = line
            row_counter += 1

        # return rows rather than storing internally in extractor
        return rows

    def _compute_label(self, fingerprint_x_num, fingerprint_y_num,
                       floor_bld_start=None, floor_bld_end=None, lat_lon_start=1, lat_lon_end=4, ap_start=5,
                       ap_end=625):
        lat_lon_floor_x = ((self.rows[fingerprint_x_num])[lat_lon_start:lat_lon_end])
        lat_lon_floor_y = ((self.rows[fingerprint_y_num])[lat_lon_start:lat_lon_end])

        # Since the floors match, compute Euclidean Distance for label
        lat_lon_x = [float(lat_lon_floor_x[0]), float(lat_lon_floor_x[1])]
        lat_lon_y = [float(lat_lon_floor_y[0]), float(lat_lon_floor_y[1])]
        (label, dist) = utils.euclidean(lat_lon_x, lat_lon_y)

        # Check if on the same floor
        if lat_lon_floor_x[2] != lat_lon_floor_y[2]:
            return 'Far', dist

        return label, dist

    def _same_device_model(self, fingerprint_x_num, fingerprint_y_num, device_index=5):
        x_phone = (self.rows[fingerprint_x_num])[device_index]
        y_phone = (self.rows[fingerprint_y_num])[device_index]
        return 0 if x_phone != y_phone else 1

    def _reset_csv_readers(self):
        """
        A function for resetting the CSV readers to
        the start of the file for future reads.
        """
        # Ensure that CSV reader is at the start of the file
        if self.mode == '-g' and self.grouping_data:
            self.id_csvfile.seek(0)
            self.crd_csvfile.seek(0)
            self.rss_csvfile.seek(0)
            self.tms_csvfile.seek(0)
        else:
            super()._reset_csv_reader()

    def group_features_to_csv(self, file_location):
        """
        A method for writing the original features that are split
        into multiple files into a single grouped csv
        """
        output_path = path.join(file_location + '-grouped.csv')
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # Write a header
            header_row = ['id', 'lat', 'lon', 'floor', 'timestamp']
            for ap_counter in range(1, len(self.rows[0]['rss']) + 1):
                header_row.append(f'AP#{str(ap_counter)}')
            writer.writerow(header_row)

            for list_item in self.rows:
                row_to_write = [list_item['id'], list_item['coords'][0], list_item['coords'][1], list_item['coords'][2],
                                list_item['timestamp']]
                for rss_val in list_item['rss']:
                    row_to_write.append(rss_val)

                writer.writerow(row_to_write)

    def merge_files_to_single_csv(self, testing_data=False):
        # Note: the output file is placed under month 00 to match pathing
        filename = 'merged-trainingData-grouped.csv'
        if testing_data:
            filename = 'merged-testingData-grouped.csv'

        output_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', '00', filename)

        # Open json config file for data regarding collection months and data files
        config_file_path = path.join('conf', 'datasets.json')
        config_file = open(config_file_path, 'r')
        config_data = json.loads(config_file.read())
        dataset_months = config_data['ujiMonths']

        # Create and write csv header row
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            header_row = ['id', 'lat', 'lon', 'floor', 'timestamp', 'phone']
            for ap_counter in range(1, 621):
                header_row.append(f'AP#{str(ap_counter)}')
            writer.writerow(header_row)

        # Close file and reopen in append mode to begin appending all data into single csv file
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for collection_month in dataset_months:
                file_prefixes = dataset_months[collection_month]['trainingFiles']
                if testing_data:
                    file_prefixes = dataset_months[collection_month]['testingFiles']
                for file_prefix in file_prefixes:
                    rows = self._extract_merged_rows(
                        path.join('datasets', '09-UJI_LIB_DB_v2', 'db', collection_month, file_prefix))

                    # get the phone model
                    phone = 'S3'
                    num = int(file_prefix[3:5])
                    if collection_month == 25:
                        if testing_data:
                            if num >= 6 or num <= 10:
                                phone = 'A5'
                        if not testing_data:
                            if num == 2:
                                phone = 'A5'

                    # Actually append the rows to the CSV file
                    for item in rows:
                        row_to_write = [item['id'], item['coords'][0], item['coords'][1], item['coords'][2],
                                        item['timestamp'], phone]
                        for rss_val in item['rss']:
                            row_to_write.append(rss_val)

                        writer.writerow(row_to_write)

    def write_rows_to_csv(self, file_location, fingerprints_to_pair, not_detected_value='100',
                          filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None,
                          ap_start=6, ap_end=625, device_index=5, num_processes=1):

        super().write_rows_to_csv(file_location, fingerprints_to_pair, not_detected_value,
                                  filter_distance_upper_bound, filter_max_distance, filter_rssi_lower_bound,
                                  ap_start=ap_start, ap_end=ap_end, device_index=device_index,
                                  num_processes=num_processes)


if __name__ == '__main__':
    input_file_path = path.join('datasets', '09-UJI_LIB_DB_v2', 'db', '01', 'trn01')
    output_file_path = path.join('output_datasets', '09-UJI_LIB_DB_v2', 'trn01.csv')
    extractor = UJIFeatureExtractor(input_file_path, '-g', True)
    extractor.write_rows_to_csv(output_file_path, 100)
