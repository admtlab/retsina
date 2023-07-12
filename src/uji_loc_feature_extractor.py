from src.FeatureExtractorInterface import FeatureExtractorInterface
import src.feature_utils as utils
import random
import csv

"""
UJILoc Feature Extractor: Responsible for processing and
parsing the data from each dataset and converting
the data into the features described in the paper by
Zach Van Hyfte and Avideh Zakhor (IPIN2021)
This feature extractor is specifically for handling dataset [11]
<Joaquín Torres-Sospedra, Raúl Montoliu, Adolfo Martínez-Usó, Tomar J. Arnau, Joan P. Avariento, Mauri Benedito-
Bordonau, Joaquín Huerta
UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization
Problems
In Proceedings of the Fifth International Conference on Indoor Positioning and Indoor Navigation, 2014.>
"""


class UJILocFeatureExtractor(FeatureExtractorInterface):
    """
    Note that the file_location in this instance is the filename
    e.g., trainingData.csv
    """

    def __init__(self, file_location, undetected_rssi='100', fingerprint_seed=29):
        # call super class init with corrected default rssi
        super().__init__(file_location, undetected_rssi=undetected_rssi, fingerprint_seed=fingerprint_seed)

    def _extract_column(self, col_num, rows_to_skip=1):
        return super()._extract_column(col_num, rows_to_skip)

    def _extract_rows(self, num_features=530, rows_to_skip=1):
        super()._extract_rows(num_features, rows_to_skip)

    def _compute_label(self, fingerprint_x_num, fingerprint_y_num, floor_bld_start=522, floor_bld_end=524,
                       lat_lon_start=520, lat_lon_end=522, ap_start=0, ap_end=520):
        return super()._compute_label(fingerprint_x_num, fingerprint_y_num, floor_bld_start, floor_bld_end,
                                      lat_lon_start, lat_lon_end)

    def _same_device_model(self, fingerprint_x_num, fingerprint_y_num, device_index=527):
        return super()._same_device_model(fingerprint_x_num, fingerprint_y_num, device_index)

    def write_rows_to_csv(self, file_location, fingerprints_to_pair, not_detected_value='100',
                          filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None,
                          ap_start=0, ap_end=520, device_index=527, num_processes=1):
        super().write_rows_to_csv(file_location, fingerprints_to_pair, not_detected_value, filter_distance_upper_bound,
                                  filter_max_distance, filter_rssi_lower_bound, ap_start, ap_end, device_index,
                                  num_processes)

    @staticmethod
    def usage():
        print('A class to be used for extracting the UJI-LOC dataset #11')


if __name__ == '__main__':
    extractor = UJILocFeatureExtractor('validationData.csv', '-g')
    extractor.write_rows_to_csv('validationData.csv', 300)
