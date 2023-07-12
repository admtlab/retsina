import src.feature_utils as utils
from src.FeatureExtractorInterface import FeatureExtractorInterface

"""
JUI Feature Extractor: Responsible for processing and
parsing the data from each dataset and converting
the data into the features described in the paper by
Zach Van Hyfte and Avideh Zakhor (IPIN2021)

This feature extractor is specifically for handling dataset [10]
<Roy P, Chowdhury C, Ghosh D, Bandyopadhyay S. JUIndoorLoc: A Ubiquitous Framework for Smartphone-
Based Indoor Localization Subject to Context and Device Heterogeneity. Wireless Personal Communications.
2019:1-24, doi:10.1007/s11277-019-06188-2, URL https://doi.org/10.1007/s11277-019-06188-2.>
"""


class JUIFeatureExtractor(FeatureExtractorInterface):
    """
    Note that the file_location in this instance is the filename
    e.g., JUIndoorLoc-Training-data.csv
    """

    def __init__(self, file_location, undetected_rssi='-110', fingerprint_seed=29):
        # call super class init with corrected default rssi
        super().__init__(file_location, undetected_rssi, fingerprint_seed)

    def _extract_column(self, col_num, rows_to_skip=2):
        return super()._extract_column(col_num, rows_to_skip)

    def _extract_rows(self, num_features=178, rows_to_skip=2):
        super()._extract_rows(num_features, rows_to_skip)

    def _compute_label(self, fingerprint_x_num, fingerprint_y_num,
                       floor_bld_start=None, floor_bld_end=None, lat_lon_start=None, lat_lon_end=None, ap_start=1,
                       ap_end=173):
        cell_id_x = ((self.rows[fingerprint_x_num])[ap_start - 1]).split('-')
        cell_id_y = ((self.rows[fingerprint_y_num])[ap_start - 1]).split('-')

        coords_x = [float(cell_id_x[1]), float(cell_id_x[2])]
        coords_y = [float(cell_id_y[1]), float(cell_id_y[2])]
        (label, dist) = utils.euclidean(coords_x, coords_y)

        # If floor numbers differ
        if cell_id_x[0] != cell_id_y[0]:
            return 'Far', dist

        return label, dist

    def _same_device_model(self, fingerprint_x_num, fingerprint_y_num, device_index=175):
        x_phone = (self.rows[fingerprint_x_num])[device_index]
        y_phone = (self.rows[fingerprint_y_num])[device_index]
        return 0 if x_phone != y_phone else 1

    def write_rows_to_csv(self, file_location, fingerprints_to_pair, not_detected_value='-110',
                          filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None,
                          ap_start=1, ap_end=173, device_index=175, num_processes=1):
        super().write_rows_to_csv(file_location, fingerprints_to_pair, not_detected_value, filter_distance_upper_bound,
                                  filter_max_distance, filter_rssi_lower_bound, ap_start, ap_end, device_index,
                                  num_processes)

    @staticmethod
    def usage():
        print('A class to be used for extracting the JUI dataset #10')


if __name__ == '__main__':
    extractor = JUIFeatureExtractor('JUIndoorLoc-Training-small.csv', '-g')
    extractor.write_rows_to_csv('JUIndoorLoc-Test-smaller-feature-data.csv', 49)
