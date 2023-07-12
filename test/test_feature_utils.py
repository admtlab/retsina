import unittest
import src
from src import feature_utils as utils


class TestFeatureUtils(unittest.TestCase):
    def test_ap_val_list(self):
        """
        Preconditions: Set up a list of 20 access points with
        the first 10 being invalid RSSI values where the
        invalid RSSI value is set to -10.
        Execution Steps: Execute ap_val_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_val_list returns a list with 10 entries, whose indexes
        and values are 10...19.
        """
        # Precondition Steps
        ap_list = []
        for x in range(0, 20):
            if x < 10:
                ap_list.append('-10')
            else:
                ap_list.append(str(x))

        # Execution Step
        list_to_check = utils.ap_val_list(ap_list, '-10')
        expected_lst = [{'value': str(x), 'index': x} for x in range(10, 20)]
        self.assertEqual(list_to_check, expected_lst, 'Incorrect APs returned as detected!')

    def test_no_ap_val(self):
        """
        Preconditions: Set up a list of 50 access points with
        all of them being invalid RSSI values of "-100."
        Execution Steps: Execute ap_val_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_val_list is empty.
        """
        ap_list = []
        for _ in range(0, 50):
            ap_list.append('-100')

        list_to_check = utils.ap_val_list(ap_list, '-100')
        self.assertEqual(list_to_check, [], 'Expected an empty list, but APs were reported as detected')

    def test_all_detected_ap_val(self):
        """
        Preconditions: Set up a list of 300 access points with
        all of them having valid RSSI values (i.e., not -50).
        Execution Steps: Execute ap_val_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_val_list has all APs.
        """
        ap_list = []
        for x in range(0, 300):
            ap_list.append(str(x))

        list_to_check = utils.ap_val_list(ap_list, '-50')
        expected_list = [{'value': str(x), 'index': x} for x in range(0, 300)]
        self.assertEqual(list_to_check, expected_list, 'Incorrect number of APs detected')

    def test_ap_list(self):
        """
        Preconditions: Set up a list of 10 valid access points
        Execution Steps: Execute ap_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_list returns a list with 10 entries, 10...19.
        """
        # Precondition Steps
        ap_list = [{'value': str(x), 'index': x} for x in range(10, 20)]

        # Execution Step
        list_to_check = utils.ap_list(ap_list)
        expected_lst = [x for x in range(10, 20)]
        self.assertEqual(list_to_check, expected_lst, 'Incorrect APs extracted from ap_list')

    def test_no_ap(self):
        """
        Preconditions: Set up an empty list to represent zero APs
        Execution Steps: Execute ap_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_list is empty.
        """
        ap_list = []

        list_to_check = utils.ap_list(ap_list)
        self.assertEqual(list_to_check, [], 'Expected an empty list, but APs were somehow extracted')

    def test_all_detected_ap(self):
        """
        Preconditions: Set up a list of 300 valid access points
        Execution Steps: Execute ap_list with the generated ap_list
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        ap_list has all APs, 0...299
        """
        ap_list = [{'value': str(x), 'index': x} for x in range(0, 300)]

        list_to_check = utils.ap_list(ap_list)
        expected_list = [x for x in range(0, 300)]
        self.assertEqual(list_to_check, expected_list, 'Incorrect number of APs detected')

    def test_basic_list_intersect(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 7...20 and the second having APs 1...13.
        Execution Steps: Execute list_intersection with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_intersection has APs 7...13
        """
        ap_lst_one = [x for x in range(7, 20)]
        ap_lst_two = [x for x in range(1, 14)]

        lst_to_check = utils.list_intersection(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(7, 14)]
        self.assertEqual(lst_to_check, expected_lst, 'Incorrect AP intersections detected. Should be 7...13')

    def test_empty_list_intersect(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...49 and the second having APs 60...99.
        Execution Steps: Execute list_intersection with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_intersection is empty, [].
        """
        ap_lst_one = [x for x in range(0, 50)]
        ap_lst_two = [x for x in range(60, 100)]

        lst_to_check = utils.list_intersection(ap_lst_one, ap_lst_two)
        self.assertEqual(lst_to_check, [], 'No APs should intersect, but some were detected. Should be []')

    def test_adv_list_intersect(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...300 and the second having even APs form 0...400,
        i.e., 0, 2, 4, ..., 398, 400.
        Execution Steps: Execute list_intersection with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_intersection has even APs 0, 2, ..., 296, 298.
        """
        ap_lst_one = [x for x in range(0, 300)]
        ap_lst_two = [x for x in range(0, 400) if x % 2 == 0]

        lst_to_check = utils.list_intersection(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(0, 300) if x % 2 == 0]
        self.assertEqual(lst_to_check, expected_lst,
                         'Only even APs between 0...298 should intersect, but other APs were detected')

    def test_basic_mut_excl(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 7...20 and the second having APs 1...13.
        Execution Steps: Execute list_mutually_excl_elems with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_mutually_excl_elems has APs 1...6 and 14...19.
        """
        ap_lst_one = [x for x in range(7, 20)]
        ap_lst_two = [x for x in range(1, 14)]

        lst_to_check = utils.list_mutually_excl_elems(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(14, 20)]
        expected_lst += [x for x in range(1, 7)]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP mutual exclusion detected. Should be 1...6 and 14...19')

    def test_full_mut_excl(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...49 and the second having APs 60...99.
        Execution Steps: Execute list_mutually_excl_elems with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_mutually_excl_elems has APs, 0...49 and 60...99.
        """
        ap_lst_one = [x for x in range(0, 50)]
        ap_lst_two = [x for x in range(60, 100)]

        lst_to_check = utils.list_mutually_excl_elems(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(0, 100) if x < 50 or x > 59]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP mutual exclusion detected. Should be 0...49 and 60...99')

    def test_adv_mut_excl(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...300 and the second having even APs form 0...400,
        i.e., 0, 2, 4, ..., 398, 400.
        Execution Steps: Execute list_mutually_excl_elems with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_mutually_excl_elems has odd APs 1, 3, ..., 299 and even APs 300, 302, ..., 396, 398.
        """
        ap_lst_one = [x for x in range(0, 300)]
        ap_lst_two = [x for x in range(0, 400) if x % 2 == 0]

        lst_to_check = utils.list_mutually_excl_elems(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(0, 300) if x % 2 == 1]
        expected_lst += [x for x in range(300, 400) if x % 2 == 0]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP mutual exclusion detected. Should be odd from 0...299 and even from 300...399')

    def test_basic_list_union(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 7...20 and the second having APs 1...13.
        Execution Steps: Execute list_union with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_union has APs 1...19.
        """
        ap_lst_one = [x for x in range(7, 20)]
        ap_lst_two = [x for x in range(1, 14)]

        lst_to_check = utils.list_union(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(1, 20)]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP union detected. Should be 1...19')

    def test_full_list_union(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...49 and the second having APs 60...99.
        Execution Steps: Execute list_union with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_union has APs, 0...49 and 60...99.
        """
        ap_lst_one = [x for x in range(0, 50)]
        ap_lst_two = [x for x in range(60, 100)]

        lst_to_check = utils.list_union(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(0, 100) if x < 50 or x > 59]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP union detected. Should be 0...49 and 60...99')

    def test_adv_list_union(self):
        """
        Preconditions: Set up two lists of valid access points. The
        first list having APs 0...300 and the second having even APs form 0...400,
        i.e., 0, 2, 4, ..., 398, 400.
        Execution Steps: Execute list_union with the generated AP lists
        as described in the Preconditions.
        Post conditions: Check that the list returned from executing
        list_union has APs 0...299 and even APs from 300...399.
        """
        ap_lst_one = [x for x in range(0, 300)]
        ap_lst_two = [x for x in range(0, 400) if x % 2 == 0]

        lst_to_check = utils.list_union(ap_lst_one, ap_lst_two)
        expected_lst = [x for x in range(0, 300)]
        expected_lst += [x for x in range(300, 400) if x % 2 == 0]
        self.assertEqual(lst_to_check, expected_lst,
                         'Incorrect AP union detected. Should be 0...299 and even values from 300...399')

    def test_close_haversine(self):
        """
        Preconditions: Set up two points. The
        first point having coordinates (10, 10) and the second having
        coordinates (10, 10.00000000001)
        Execution Steps: Execute haversine with the two points as described in the Preconditions.
        Post conditions: Check that the label returned from executing haversine is Close.
        """
        p_x = [10, 10]
        p_y = [10, 10.00000000001]
        label_to_check = utils.haversine(p_x, p_y)
        self.assertEqual(label_to_check, 'Close',
                         f'(10, 10) and (11, 11) labeled as {label_to_check}, but should be Close')

    def test_far_haversine(self):
        """
        Preconditions: Set up two points. The
        first point having coordinates (25, 20) and the second having coordinates (65, 63)
        Execution Steps: Execute haversine with the two points as described in the Preconditions.
        Post conditions: Check that the label returned from executing haversine is Far.
        """
        p_x = [25, 20]
        p_y = [65, 63]
        label_to_check = utils.haversine(p_x, p_y)
        self.assertEqual(label_to_check, 'Far',
                         f'(25, 20) and (65, 63) labeled as {label_to_check}, but should be Far')

    def test_same_point_haversine(self):
        """
        Preconditions: Set up two points that have the same coordinates (50, 50).
        Execution Steps: Execute haversine with the two points as described in the Preconditions.
        Post conditions: Check that the label returned from executing haversine is Close.
        """
        p_x = [50, 50]
        p_y = [50, 50]
        label_to_check = utils.haversine(p_x, p_y)
        self.assertEqual(label_to_check, 'Close', f'Same point labeled as {label_to_check}, but should be Close')

    def test_negative_haversine(self):
        """
        Preconditions: Set up two points. The
        first point having coordinates (-10, -14) and the second having coordinates (21, 31)
        Execution Steps: Execute haversine with the two points as described in the Preconditions.
        Post conditions: Check that the label returned from executing haversine is Far.
        """
        p_x = [-10, 14]
        p_y = [21, 31]
        label_to_check = utils.haversine(p_x, p_y)
        self.assertEqual(label_to_check, 'Far',
                         f'(-10, 14) and (21, 31) labeled as {label_to_check}, but should be Far')


if __name__ == '__main__':
    unittest.main()
