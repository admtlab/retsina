import random
import csv
import src.feature_utils as utils
from multiprocessing import Manager, Pool


class APConfig:
    """
    Due to the use of a multiprocessing queue and a process pool, this object simply
    acts as a wrapper class for encapsulating properties that should be passed to a single
    process when running in parallel
    """

    def __init__(self, mp_queue, x, y, not_detected_value='100', filter_distance_upper_bound=None,
                 filter_max_distance=None, filter_rssi_lower_bound=None):
        self.mp_queue = mp_queue
        self.x = x
        self.y = y
        self.not_detected_value = not_detected_value
        self.filter_distance_upper_bound = filter_distance_upper_bound
        self.filter_max_distance = filter_max_distance
        self.filter_rssi_lower_bound = filter_rssi_lower_bound


class FeatureExtractorInterface:
    """
    An interface class for the overlapping components of the feature
    extractors, overridden by extractors when needed
    """

    def __init__(self, file_location, undetected_rssi='100', fingerprint_seed=9, multiple_files=False):
        # Set the seed for randomly selecting fingerprints to pair
        random.seed(fingerprint_seed)

        # Note that this value is used for undetected APs
        self.not_detected_value = undetected_rssi

        # For storing data rows for feature calculations
        self.rows = []

        if not multiple_files:
            self.filepath = file_location
            self.csvfile = open(self.filepath)
            self.csvreader = csv.reader(self.csvfile, delimiter=',')

            self._extract_rows()

    def _reset_csv_reader(self):
        """
        A simple method for resetting the csv file reader if multiple
        passes over the same file are needed.
        """
        self.csvfile.seek(0)

    def get_data_df(self):
        return utils.get_data_df(self.filepath)

    def _extract_column(self, col_num, rows_to_skip=1):
        """
        A function for extracting the values of a single column
        as a Python List
        """
        self._reset_csv_reader()
        values = []
        i = 0
        for line in self.csvreader:
            # Skip first line of CSV
            if i < rows_to_skip:
                i += 1
                continue
            values.append(line[col_num])
        return values

    def _extract_rows(self, num_features=178, rows_to_skip=1):
        """
        A function for extracting the rows of the dataset
        as a list into rows attribute of self
        """
        self.rows = []
        i = 0
        for line in self.csvreader:
            # Skip first row of CSV
            if i < rows_to_skip:
                i += 1
                continue
            row = []
            for x in range(0, num_features - 1):
                row.append(line[x])
            self.rows.append(row)

    def _compute_label(self, fingerprint_x_num, fingerprint_y_num,
                       floor_bld_start=170, floor_bld_end=172, lat_lon_start=168, lat_lon_end=170, ap_start=None,
                       ap_end=None):
        """
        A function for labelling two APs, x and y, as Close or Far based on the distance features
        provided in the corresponding dataset. Note that this function can (and is) be overriden in
        feature extractors that implement this interface to utilize other properties for setting the
        target label.
        :return:  A tuple with the first element being the target label (e.g., Close or Far) and the
                    second element being the distance between APs x and y
        """
        floor_building_x = (self.rows[fingerprint_x_num])[floor_bld_start:floor_bld_end]
        lat_lon_x = (self.rows[fingerprint_x_num])[lat_lon_start:lat_lon_end]
        floor_building_y = (self.rows[fingerprint_y_num])[floor_bld_start:floor_bld_end]
        lat_lon_y = (self.rows[fingerprint_y_num])[lat_lon_start:lat_lon_end]

        # Determine if both fingerprints were in close-proximity, assuming same space means Close
        in_proximity = 'Close'
        distance = -1
        for coord in range(0, len(floor_building_x)):
            if floor_building_x[coord] != floor_building_y[coord]:
                in_proximity = 'Far'
                coords_x = [float(lat_lon_x[0]), float(lat_lon_x[1])]
                coords_y = [float(lat_lon_y[0]), float(lat_lon_y[1])]
                distance = utils.euclidean(coords_x, coords_y)[1]

        if in_proximity == 'Close':
            coords_x = [float(lat_lon_x[0]), float(lat_lon_x[1])]
            coords_y = [float(lat_lon_y[0]), float(lat_lon_y[1])]
            (in_proximity, distance) = utils.euclidean(coords_x, coords_y)
        return in_proximity, distance

    def _same_device_model(self, fingerprint_x_num, fingerprint_y_num, device_index=527):
        """
        A simple function for determining if two APs, x and y, were recorded by the same
        device model based on the appropriate features of the corresponding dataset.
        Note that this function can (and is) be overriden in feature extractors that
        implement this interface to utilize other properties for checking for device equality
        :return: 1 if the two recordings were done by the same device model and 0 otherwise.
        """
        x_phone = int((self.rows[fingerprint_x_num])[device_index])
        y_phone = int((self.rows[fingerprint_y_num])[device_index])

        same_model = 0
        if x_phone == y_phone:
            same_model = 1
        # all if statements based on UJILoc readme
        elif x_phone in [1, 2] and y_phone in [1, 2]:
            same_model = 1
        elif x_phone in [8, 9] and y_phone in [8, 9]:
            same_model = 1
        elif x_phone in [11, 12] and y_phone in [11, 12]:
            same_model = 1
        elif x_phone in [14, 15] and y_phone in [14, 15]:
            same_model = 1
        elif x_phone in [19, 20] and y_phone in [19, 20]:
            same_model = 1

        return same_model

    def write_rows_to_csv(self, file_location, fingerprints_to_pair, not_detected_value='100',
                          filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None,
                          ap_start=0, ap_end=168, device_index=527, num_processes=1):
        """
        A function for writing fingerprint AP features to a CSV file.
        The function takes in the filename for the output file as well as the
        number of fingerprints to consider when computing the AP features for each pair.
        This allows for the generation of smaller pairs of features as entire datasets can be
        quite long with respect to runtime.
        """
        num_to_pair = fingerprints_to_pair
        if fingerprints_to_pair > len(self.rows):
            num_to_pair = len(self.rows)

        # Randomly select which fingerprints to use
        sampled_fingerprints = random.sample(range(len(self.rows)), num_to_pair)

        # Map to list of APs
        row_aps = []
        for i in range(0, len(sampled_fingerprints)):
            row_aps.append((self.rows[sampled_fingerprints[i]])[ap_start:ap_end])

        # Compute the features of all pairs before writing to CSV by placing tuples into MP queue
        mng = Manager()
        queue = mng.Queue()

        # Create list of fingerprint pairs to compute features for and gather
        sampled_pairs = []
        for i in range(0, num_to_pair):
            x = sampled_fingerprints[i]
            for j in range(i + 1, num_to_pair):
                y = sampled_fingerprints[j]
                same_model = self._same_device_model(x, y, device_index)

                (pair_label, distance) = self._compute_label(x, y)
                sampled_pairs.append(
                    (queue, x, y, row_aps[i], row_aps[j], pair_label, distance, not_detected_value, same_model,
                     filter_distance_upper_bound, filter_max_distance, filter_rssi_lower_bound))
        with Pool(num_processes) as p:
            """A very complicated line, starmap will split the work of this list for function 
               utils.fingerprint_pair_features() among the 4 processes. The list being evaluated 
               consists of a queue to act as shared memory for the result, the ap pair, and the 
               other four required params"""
            p.starmap(utils.fingerprint_pair_features, sampled_pairs)

        # At this point, the queue should contain all features for the necessary pairs
        with open(file_location, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # Write a header
            header_row = ['x_id', 'y_id', 'x_num_aps', 'y_num_aps', 'shared_count', 'union_count', 'non_shared_count',
                          'diff_count',
                          'jaccard_ratio']
            # Repeat these per transformation
            repeat = ['manhattan', 'euclidean']
            for x in range(1, 16):
                repeat.append(f'top_ap_{str(x)}dB')
            for x in range(1, 16):
                repeat.append(f'percent_shared_{str(x)}dB')
            for x in range(1, 9):
                repeat.append(f'shared_top_{str(x)}')
            repeat = repeat + ['redpin_1', 'redpin_2', 'cos_shared', 'cos_difference', 'cos_ratio', 'cos_rank',
                               'pearson_shared', 'pearson_difference', 'pearson_ratio', 'pearson_rank',
                               'spear_shared', 'spear_difference', 'spear_ratio', 'spear_rank',
                               'kendall_shared', 'kendall_difference', 'kendall_ratio', 'kendall_rank', 'diff_smallest',
                               'diff_largest', 'diff_mean', 'diff_median', 'diff_harmonic_mean', 'diff_stdev',
                               'diff_pstdev',
                               'pair_diff_smallest', 'pair_diff_largest', 'pair_diff_mean', 'pair_diff_median',
                               'pair_diff_harmonic_mean', 'pair_diff_stdev', 'pair_diff_pstdev', 'pair_ratio_smallest',
                               'pair_ratio_largest', 'pair_ratio_mean', 'pair_ratio_median', 'pair_ratio_harmonic_mean',
                               'pair_ratio_stdev', 'pair_ratio_pstdev', 're3']

            # repeat for transformations (0=none, 1=single least squares, 2=50% least squares, 3=double least squares)
            header_row = header_row + [("0 - " + i) for i in repeat]
            header_row = header_row + [("1 - " + i) for i in repeat]
            header_row = header_row + [("2 - " + i) for i in repeat]
            header_row = header_row + [("3 - " + i) for i in repeat]
            header_row = header_row + ['same_model', 'distance', 'proximity']
            writer.writerow(header_row)

            # Rather than computing features on the fly here, utilize the queue
            while not queue.empty():
                (x_id, y_id, base_features, no_transformation, single_transformation, half_transformation,
                 double_transformation, model, dist, in_proximity) = queue.get()

                writer.writerow(
                    [x_id, y_id, *base_features, *no_transformation, *single_transformation, *half_transformation,
                     *double_transformation, model, dist, in_proximity])
