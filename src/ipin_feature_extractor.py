from src.FeatureExtractorInterface import FeatureExtractorInterface
import random
import csv
import src.feature_utils as utils

"""
IPIN Tutorial Feature Extractor: Responsible for processing and
parsing the data from each dataset and converting
the data into the features described in the paper by
Zach Van Hyfte and Avideh Zakhor (IPIN2021)
This feature extractor is specifically for handling dataset [13]
<Torres-Sospedra, J.; Moreira, A.; Knauth, S.; Berkvens, R.; Montoliu, R.; Belmonte, O.; Trilles, S.; Nicolau, M.J.; 
Meneses, F.; Costa, A.; Koukofikis, A.; Weyn, M.; & Peremans, H. 
A Realistic Evaluation of Indoor Positioning Systems Based on Wi-Fi Fingerprinting: 
The 2015 EvAAL-ETRI Competition Journal of Ambient Intelligence and Smart Environments, pp. xxx-xxx, 2016>
"""


class IPINFeatureExtractor(FeatureExtractorInterface):

    def __init__(self, file_location, undetected_rssi='-110', fingerprint_seed=29):
        # call super class init with corrected default rssi
        super().__init__(file_location, undetected_rssi, fingerprint_seed)

    def _extract_column(self, col_num, rows_to_skip=2):
        return super()._extract_column(col_num, rows_to_skip)

    def _extract_rows(self, num_features=178, rows_to_skip=2):
        super()._extract_rows(num_features, rows_to_skip)

    def _compute_label(self, fingerprint_x_num, fingerprint_y_num,
                       floor_bld_start=170, floor_bld_end=172, lat_lon_start=168, lat_lon_end=170, ap_start=0,
                       ap_end=168):
        return super()._compute_label(fingerprint_x_num, fingerprint_y_num, floor_bld_start, floor_bld_end,
                                      lat_lon_start, lat_lon_end, ap_start, ap_end)

    def _same_device_model(self, fingerprint_x_num, fingerprint_y_num, device_index=175):
        x_phone = (self.rows[fingerprint_x_num])[device_index]
        y_phone = (self.rows[fingerprint_y_num])[device_index]
        return 0 if x_phone != y_phone else 1

    def write_rows_to_csv(self, file_location, fingerprints_to_pair, not_detected_value='100',
                          filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None,
                          ap_start=0, ap_end=168, device_index=175, num_processes=1):
        super().write_rows_to_csv(file_location, fingerprints_to_pair, not_detected_value, filter_distance_upper_bound,
                                  filter_max_distance, filter_rssi_lower_bound, ap_start, ap_end, device_index,
                                  num_processes)

    @staticmethod
    def usage():
        print('A class to be used for extracting the IPIN dataset #13')


if __name__ == '__main__':
    extractor = IPINFeatureExtractor('fingerprints_01.csv', '-g')
    extractor.write_rows_to_csv('fingerprints_01.csv', 100)
