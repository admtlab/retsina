from math import radians, cos, sin, asin, sqrt, pow
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy import linalg
import pandas as pd
import itertools
import numpy as np
import statistics

"""
A Python file containing classes and functions that are utilized by several
other classes in this project such as the various feature extraction classes.
"""


class InternalValidation:
    """
    This is a class for Internal Validation operations such as
    cross-validation, k-folds, and computing the mean accuracy
    and standard deviation of each fold.
    """

    @staticmethod
    def calculate_mean_standard_deviation(scores):
        """
        The function for computing the mean and standard deviation
        of a list of scores, typically from k-folds validation.

        Parameters:
            scores (list): The list of scores to compute the metrics from

        Returns:
            (float, float): A tuple with the first item being the average accuracy
                and the second being the standard deviation.
        """
        tot = 0.0
        for scor in scores:
            tot += scor
        avg = tot / int(len(scores))
        totdev = 0.0
        for scor in scores:
            devz = scor - avg
            totdev += pow(devz, 2)
        variance = totdev / int(len(scores) - 1)
        standarddev = pow(variance, .5)
        return avg, standarddev

    @staticmethod
    def score_split(split, data, labels, model):
        """
        A function that calculates the score on the given train test split.

        Parameters:
            split (list): The list of beginning and end splits, typically obtained
                from the Splitter class. Index 0 represents the training beginning index,
                Index 1 represents the training ending index, and the same format holds for
                Indices 2 and 3 for the testing splits.
            data (DataFrame): The training dataset without the label as a pandas DataFrame.
            labels (DataFrame): The labels from the training dataset as a pandas DataFrame.
            model (Classifier): An sklearn Classifier class instance that will be used for
                classifications during the cross-validation.

        Returns:
            float: The score computed for the current k-fold split.
        """
        train_b = split[0]
        train_e = split[1]
        test_b = split[2]
        test_e = split[3]

        # Remove testing data if there is an overlap
        if train_b <= test_b <= train_e:
            x_train_lst = [data[train_b:test_b - 1], data[test_e + 1:train_e]]
            y_train_lst = [labels[train_b:test_b - 1], labels[test_e + 1:train_e]]
            x_train = pd.concat(x_train_lst)
            x_test = data[test_b:test_e]
            y_train = pd.concat(y_train_lst)
            y_test = labels[test_b:test_e]
            model.fit(x_train, y_train.values.ravel())
            scor = model.score(x_test, y_test.values.ravel())
            print(
                f'Fold: ([{train_b}:{test_b - 1}, {test_e + 1}:{train_e}], [{test_b}, {test_e}])\tscore = {str(scor)}')
            return scor

        # If no overlap, proceed as usual
        x_train = data[train_b:train_e]
        x_test = data[test_b:test_e]
        y_train = labels[train_b:train_e]
        y_test = labels[test_b:test_e]

        model.fit(x_train, y_train.values.ravel())
        scor = model.score(x_test, y_test.values.ravel())
        print(f'Fold: {str(split)}\tscore = {str(scor)}')

        return scor

    @staticmethod
    def cross_validation(splits, data, labels, classifier, bar=None):
        """
        The function for performing cross-validation to evaluate the classifier.

        Parameters:
            splits (list): The list of beginning and end splits, typically obtained
                from the Splitter class. Index 0 represents the training beginning index,
                Index 1 represents the training ending index, and the same format holds for
                Indices 2 and 3 for the testing splits.
            data (DataFrame): The training dataset without the label as a pandas DataFrame.
            labels (DataFrame): The labels from the training dataset as a pandas DataFrame.
            classifier (Classifier): An sklearn Classifier class instance that will be used for
                classifications during the cross-validation.
            bar (list): An optional parameter for storing cross-validation results into a list.
                The argument is named bar as it is very useful for extracting the y_axis for plots.

        Returns:
            (float, float): A tuple of score values from the provided splits. The first value is the
            average accuracy and the second value is the standard deviation of all scores.
        """
        scores = []
        for split in splits:
            split_score = InternalValidation.score_split(split, data, labels, classifier)
            scores.append(split_score)
            if bar is not None:
                bar.append(split_score * 100)

        return InternalValidation.calculate_mean_standard_deviation(scores)


class Splitter:
    """
    This is a class for splitting the data typically used when
    performing cross-validation on the training data as an
    additional metric before testing on the testing set.
    """

    @staticmethod
    def get_k_fold_splits(data, num_splits=10):
        """
        A function for obtaining the k-fold split values of a dataset.

        Parameters:
            data (DataFrame): The training dataset, excluding labels, as a Pandas DataFrame
            num_splits (int): An optional argument to specify the number of splits to create.
                The default number of splits is 10.

        Returns:
            list: The list of tuples that indicate the k-fold split indices. Index 0 represents
                the training beginning index, Index 1 represents the training ending index, and
                the same format holds for Indices 2 and 3 for the testing splits.
        """
        kf = KFold(n_splits=num_splits)
        splits = kf.split(data)
        tuples = []
        for train_index, test_index in splits:
            train_b = train_index[0]
            train_e = train_index[-1]
            test_b = test_index[0]
            test_e = test_index[-1]
            split = (train_b, train_e, test_b, test_e)
            tuples.append(split)
        return tuples


def get_data_df(filepath):
    """
    Reads the csv file at the specified location and
    returns this as a pandas DataFrame object
    :param filepath: the location of the CSV file
    :return: pandas DataFrame of the CSV file
    """
    x_df = pd.read_csv(filepath)

    # Encode target labels as 0 for Close and 1 for Far
    y_df = pd.DataFrame()
    y_df['proximity'] = (x_df.proximity == 'Far').astype(int)

    # Remove labels from x_df
    x_df = x_df.drop('proximity', axis=1)

    return x_df, y_df


def euclidean(p_x, p_y):
    """
    Function for computing Euclidean distance between two
    points on a Euclidean space (x, y)
    :param p_x: Point 1
    :param p_y: Point 2
    :return: the Euclidean Distance
    """
    x1 = p_x[0]
    y1 = p_x[1]
    x2 = p_y[0]
    y2 = p_y[1]

    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if distance <= 2.25:
        return 'Close', distance

    return 'Far', distance


def haversine(p_x, p_y):
    """
    Function for computing great circle distance
    between two points on the earth (specified in decimal degrees).

    Parameters:
        p_x (list): Point x with index 0 containing the latitude
            and index 1 containing the longitude.
        p_y (list): Point y with index 0 containing the latitude
            and index 2 containint the longitude.

    Returns:
        String: The label of whether the two points are in close-proximity
            or not. If the distance is less than 2.25 meters, the result is
            Close. Otherwise, the result is Far.
    """
    lon1 = p_x[0]
    lat1 = p_x[1]
    lon2 = p_y[0]
    lat2 = p_y[1]

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers is 6371
    km = 6371 * c

    # Convert kilometers to meters for comparing to CDC's threshold
    m = km * 1000

    if m <= 2.25:
        return 'Close', m
    return 'Far', m


def list_union(lst1, lst2):
    """
    Function for returning the union of two lists without
    any repetitions
    """
    return list(set(lst1) | set(lst2))


def list_intersection(lst1, lst2):
    """
    Function for returning the intersection of two lists
    """
    return list(set(lst1).intersection(lst2))


def list_mutually_excl_elems(lst1, lst2):
    """
    Function for returning the elements that were only
    present in a single list, but not both.
    """
    mutual_excl_x = [x for x in lst1 if x not in lst2]
    mutual_excl_y = [y for y in lst2 if y not in lst1]
    return mutual_excl_x + mutual_excl_y


def ap_val_list(lst, not_detected_value, filter_rssi_lower_bound=None):
    """
    A function for filtering out APs that were not detected, yielding
    a list of the APs that were detected. index represents the AP number
    and value represents the RSSI value.
    """
    # Filter out weak RSSI signals and undetected APs
    if filter_rssi_lower_bound is not None:
        return list(
            filter(lambda x: (x['value'] != not_detected_value and float(x['value']) > filter_rssi_lower_bound),
                   [{'value': x, 'index': i} for i, x in enumerate(lst)])
        )

    # Else if rssi filter is not provided, only filter undetected APs
    return list(
        filter(lambda x: (x['value'] != not_detected_value), [{'value': x, 'index': i} for i, x in enumerate(lst)]))


def ap_list(lst):
    """
    A function for mapping the list of APs as index and RSSI
    values as value to a list of only the AP IDs. This function
    is useful for obtaining a list of just the AP index.
    """
    return list(map(lambda x: x['index'], lst))


def ap_dict(fingerprint_ap_vals):
    ap_dic = {}
    for x in fingerprint_ap_vals:
        ap_dic[x['index']] = x['value']
    return ap_dic


def list_top(aps_dict):
    '''
    Finds the top value APs of any fingerprint. Returns dict
    with AP and value 
    '''
    top = {}
    top_val = -200
    for x in aps_dict:
        if int(aps_dict[x]) > top_val:
            top_val = int(aps_dict[x])
            top = {}
            top[x] = int(aps_dict[x])
        elif int(aps_dict[x]) == top_val:
            top[x] = int(aps_dict[x])
    return (top)


def RedpinScore(x_aps_dict, y_aps_dict, shared_aps):
    # adapted from Redpin method measurementSimilarityLevel
    ID_POS_CONTRIBUTION = 1
    ID_NEG_CONTRIBUTION = -0.4
    SIGNAL_CONTRIBUTION = 1
    LOCATION_KNOWN = 10
    LOCATION_UNKNOWN = 0
    total_credit = 0
    account = 0
    matches = 0
    readings = 0
    for i in shared_aps:
        matches = matches + 1
        account = account + ID_POS_CONTRIBUTION
        account = account + signalContribution(int(x_aps_dict[i]), int(y_aps_dict[i]))
    readings = max(len(x_aps_dict), len(y_aps_dict))
    account = account + (readings - matches) * ID_NEG_CONTRIBUTION
    total_credit += len(x_aps_dict) * ID_POS_CONTRIBUTION
    total_credit += len(x_aps_dict) * SIGNAL_CONTRIBUTION
    factor = LOCATION_KNOWN - LOCATION_UNKNOWN
    accuracy = 0
    if account > 0:
        a = (account / total_credit) * factor + LOCATION_UNKNOWN
        accuracy = round(a)
    return accuracy


def signalContribution(rssi1, rssi2):
    SIGNAL_CONTRIBUTION = 1
    SIGNAL_PENALTY_THRESHOLD = 10
    SIGNAL_GRAPH_LEVELING = 0.2
    base = rssi1
    diff = abs(rssi1 - rssi2)

    # Handle edge case where base is zero
    if base == 0.0:
        if diff > 0.0:
            # Set to the non-zero rssi value
            base = rssi2
        else:
            # If rssi values are both zero, return 1
            return SIGNAL_CONTRIBUTION

    x = diff / base
    if x > 0.0:
        y = 1 / x
        t = SIGNAL_PENALTY_THRESHOLD / base
        y -= 1 / t
        y = y * SIGNAL_GRAPH_LEVELING
        if (-1 * SIGNAL_CONTRIBUTION <= y) and (y <= SIGNAL_CONTRIBUTION):
            return y
        else:
            return SIGNAL_CONTRIBUTION
    else:
        return SIGNAL_CONTRIBUTION


def fingerprint_pair_features(mp_queue, x_id, y_id, row_x_aps, row_y_aps, pair_label, distance, not_detected_value,
                              same_model,
                              filter_distance_upper_bound=None, filter_max_distance=None, filter_rssi_lower_bound=None):
    """
    A function that, given a list of APs detected by fingerprint x and a list
    of fingerprints detected by fingerprint y, computes all the AP-based features
    between the two fingerprints. All the features are returned in a single
    tuple placed into the provided multiprocessing queue.
    """
    # Find all features not dependent on RSSI value
    # Convert each fingerprint list to the AP ID if detected
    fingerprint_x_ap_vals = ap_val_list(row_x_aps, not_detected_value, filter_rssi_lower_bound)
    fingerprint_y_ap_vals = ap_val_list(row_y_aps, not_detected_value, filter_rssi_lower_bound)
    fingerprint_x_aps = ap_list(fingerprint_x_ap_vals)
    fingerprint_y_aps = ap_list(fingerprint_y_ap_vals)
    x_aps_dict = ap_dict(fingerprint_x_ap_vals)
    y_aps_dict = ap_dict(fingerprint_y_ap_vals)
    ### A. AP Detection-Based Features ###
    fingerprint_x_num_aps = len(fingerprint_x_aps)
    fingerprint_y_num_aps = len(fingerprint_y_aps)
    shared_aps = list_intersection(fingerprint_x_aps, fingerprint_y_aps)
    union_aps = list_union(fingerprint_x_aps, fingerprint_y_aps)
    exclusive_aps = list_mutually_excl_elems(fingerprint_x_aps, fingerprint_y_aps)
    shared_ap_count = len(shared_aps)
    union_ap_count = len(union_aps)
    non_shared_ap_count = len(exclusive_aps)
    detected_ap_count_difference = abs(fingerprint_x_num_aps - fingerprint_y_num_aps)
    jaccard_similarity = 0
    # Note if no APs are shared, the two fingerprints have 0 jaccard similarity score
    if union_ap_count != 0:
        jaccard_similarity = shared_ap_count / union_ap_count

    base_features = [fingerprint_x_num_aps, fingerprint_y_num_aps, shared_ap_count, union_ap_count, non_shared_ap_count,
                     detected_ap_count_difference, jaccard_similarity]

    # LINEAR TRANSFORMATIONS - find all features dependent on RSSI values for each transformation
    # No transformation
    no_transformation = rssi_features(x_aps_dict, y_aps_dict, shared_aps)

    if shared_aps:
        # Single-fingerprint least squares transformation (fit using the shared access points)
        x_shared = [x_aps_dict[i] for i in shared_aps]
        y_shared = [y_aps_dict[i] for i in shared_aps]
        x_var = np.array([int(x) for x in x_shared]).reshape(-1, 1)
        y_var = np.array([int(x) for x in y_shared]).reshape(-1, 1)
        model = LinearRegression().fit(x_var, y_var)
        x_dict_single = {}
        for x in x_aps_dict:
            sample = np.array([int(x_aps_dict[x])]).reshape(1, -1)
            x_dict_single[x] = model.predict(sample)[0][0]
        single_transformation = rssi_features(x_dict_single, y_aps_dict, shared_aps)
        # 50% least squares
        x_dict_half = {}
        for x in x_aps_dict:
            x_dict_half[x] = x_dict_single[x] / 2
        half_transformation = rssi_features(x_dict_half, y_aps_dict, shared_aps)
        # Double fingerprint least squares
        model_y = LinearRegression().fit(y_var, x_var)
        y_dict_single = {}
        for y in y_aps_dict:
            sample = np.array([int(y_aps_dict[y])]).reshape(1, -1)
            y_dict_single[y] = model_y.predict(sample)[0][0]
        double_transformation = rssi_features(x_dict_single, y_dict_single, shared_aps)
    else:
        single_transformation = no_transformation
        half_transformation = no_transformation
        double_transformation = no_transformation
    # Return values
    feature_result = (
        x_id, y_id, base_features, no_transformation, single_transformation, half_transformation, double_transformation,
        same_model, distance, pair_label)

    # If the result should not be pruned out, add to the queue, else ignore result
    # Filter based on Euclidean distance in range of (2.25m, upper_bound m]
    if filter_distance_upper_bound is not None:
        if distance <= 2.25 or distance > filter_distance_upper_bound:
            if filter_max_distance is not None:
                # If both are set
                if distance <= filter_max_distance:
                    mp_queue.put(feature_result)
            else:
                # If only upper bound is set
                mp_queue.put(feature_result)
    elif filter_max_distance is not None:
        # If only max distance is set
        if distance <= filter_max_distance:
            mp_queue.put(feature_result)
    else:
        # If neither are set, always add
        mp_queue.put(feature_result)


def rssi_features(x_aps_dict, y_aps_dict, shared_aps):
    """
    A function for computing all features for a single
    transform given a dictionary of x_aps, y_aps and a list of
    the APs that are shared between x and y. The function returns
    a list of all computed features.
    :param x_aps_dict: a dictionary of x's APs and RSSI values
    :param y_aps_dict: a dictionary of y's APs and RSSI values
    :param shared_aps: a list of the APs that are shared between
                        x and y
    :return: A list of all computed features between x and y
    """
    sorted_x_vals = dict(sorted(x_aps_dict.items(), key=lambda x: x[1]))
    sorted_y_vals = dict(sorted(y_aps_dict.items(), key=lambda x: x[1]))
    ### B. Basic RSSI Value-Based Features ###
    # Manhattan and Euclidean distances
    # Make shared ap lists (just append values)
    shared_x_rssi = []
    shared_y_rssi = []
    for i in shared_aps:
        shared_x_rssi.append(x_aps_dict[i])
        shared_y_rssi.append(y_aps_dict[i])
    # Default of 120 as this is the farthest RSSI distance value
    manhattan = 120
    euclidean = 120
    if (len(shared_aps) > 0):
        manhattan = sum(
            abs(int(val1) - int(val2)) for val1, val2 in zip(shared_x_rssi, shared_y_rssi)) / len(shared_aps)
        euclidean = sqrt(
            sum((int(val1) - int(val2)) ** 2 for val1, val2 in zip(shared_x_rssi, shared_y_rssi))) / len(shared_aps)
    # Get shared top access points
    top_x = list_top(x_aps_dict)
    top_y = list_top(y_aps_dict)
    shared_top = list_intersection(top_x.keys(), top_y.keys())
    # B-1. Shares top APs within Z dBm of each other
    shared_top_z = []
    if (shared_top):
        shared_top_z = []
        for z in range(1, 16):
            cur_shared_top = 0
            for i in shared_top:
                if (abs(int(top_x[i]) - int(top_y[i])) <= z):
                    cur_shared_top = 1
            if cur_shared_top == 1:
                shared_top_z.append(1)
            else:
                shared_top_z.append(0)
    else:
        for z in range(1, 16):
            shared_top_z.append(0)
    # B-2. RSSIs within Z dBm percentage
    shared_percent_z = []
    # calculate the percentage of aps within Z dBm of each other
    for z in range(1, 16):
        num_shared = 0
        if len(shared_aps) == 0:
            shared_percent_z.append(0)
        else:
            for i in shared_aps:
                if abs(int(x_aps_dict[i]) - int(y_aps_dict[i])) <= z:
                    num_shared = num_shared + 1
            percent_shared = num_shared / len(shared_x_rssi)
            shared_percent_z.append(percent_shared)
    # B-3. Top K access points in common
    shares_top_k = []
    for k in range(1, 9):
        if len(sorted_x_vals) < k or len(sorted_y_vals) < k:
            shares_top_k.append(0)
        else:
            top_k_x = list(sorted_x_vals.keys())[0:k]
            top_k_y = list(sorted_y_vals.keys())[0:k]
            if set(top_k_x) == set(top_k_y):
                shares_top_k.append(1)
            else:
                shares_top_k.append(0)
    ### C. Redpin Score-Based Features ###
    redpin_1 = RedpinScore(x_aps_dict, y_aps_dict, shared_aps)
    redpin_2 = RedpinScore(y_aps_dict, x_aps_dict, shared_aps)

    ### D. Correlation-Based Features ###
    # Generate shared AP RSSI value vectors
    shared_x_rssi = [int(i) for i in shared_x_rssi]
    shared_y_rssi = [int(i) for i in shared_y_rssi]
    # Define these vectors as empty because they're needed in E. Difference Vectors
    difference_x = []
    difference_y = []
    ratio_x = []
    ratio_y = []
    # In case there are no shared features
    if len(shared_x_rssi) == 0:
        # Cosine similarity
        cos_shared = cos_difference = cos_ratio = cos_rank = 0
        # Pearson coefficient
        pearson_shared = pearson_difference = pearson_ratio = pearson_rank = 0
        # Spearman coefficient
        spear_shared = spear_difference = spear_ratio = spear_rank = 0
        # Kendall coefficient
        kendall_shared = kendall_difference = kendall_ratio = kendall_rank = 0
    elif len(shared_x_rssi) == 1:
        # if there is only one shared value, the difference and ratio vectors will be empty
        cos_shared = cosine_similarity([shared_x_rssi], [shared_y_rssi])[0][0]
        cos_difference = cos_ratio = 0  # there is no possible similarity between empty vectors
        cos_rank = 1  # the vectors would be the same [1] = [1]
        # The rest will all be 0 because a vector with no variance (or an empty vector) has correlation coefficient of 0
        pearson_shared = pearson_difference = pearson_ratio = pearson_rank = 0
        spear_shared = spear_difference = spear_ratio = spear_rank = 0
        kendall_shared = kendall_difference = kendall_ratio = kendall_rank = 0
    else:
        # Generate shared AP pair difference vectors
        combo_x = list(itertools.combinations(shared_x_rssi, 2))
        combo_y = list(itertools.combinations(shared_y_rssi, 2))
        difference_x = []
        difference_y = []
        for x, y in zip(combo_x, combo_y):
            difference_x.append(abs(x[0] - x[1]))
            difference_y.append(abs(y[0] - y[1]))
        # Generate shared AP pair ratio vectors
        ratio_x = []
        ratio_y = []
        for x, y in zip(combo_x, combo_y):
            if x[1] == 0:
                ratio_x.append(1000)
            else:
                ratio_x.append(abs(x[0] / x[1]))
            if y[1] == 0:
                ratio_y.append(1000)
            else:
                ratio_y.append(abs(y[0] / y[1]))
        # Generate normalized ordered shared AP rank vectors
        # Create rank_x vector (ordered weakest to strongest)
        sorted_shared_x_vals = {key: x_aps_dict[key] for key in shared_aps}
        sorted_shared_x_vals = dict(sorted(sorted_shared_x_vals.items(), key=lambda x: x[1], reverse=True))
        rank_x = stats.rankdata([int(b) for b in list(sorted_shared_x_vals.values())], method='max')
        # Create rank_y vector
        y_vals = []
        for key in sorted_shared_x_vals.keys():  # for each sorted x AP, get the y AP value
            y_vals.append(int(y_aps_dict[key]))
        rank_y = stats.rankdata(y_vals, method='max')
        # normalize the vectors
        rank_x = list(rank_x / linalg.norm(rank_x, ord=1))
        rank_y = list(rank_y / linalg.norm(rank_y, ord=1))
        # For each vector pair, get the cosine similarity and the pearson, spear, and kendall coefficients
        # Shared RSSI vectors
        if statistics.variance(shared_x_rssi) == 0 or statistics.variance(shared_y_rssi) == 0:
            # If the vector has no variance, then there's a divide by 0 error in pearson, spearman, and kendall.
            # No variance means no correlation
            cos_shared = cosine_similarity([shared_x_rssi], [shared_y_rssi])[0][0]
            pearson_shared = 0
            spear_shared = 0
            kendall_shared = 0
        else:
            cos_shared = cosine_similarity([shared_x_rssi], [shared_y_rssi])[0][0]
            pearson_shared = np.corrcoef(shared_x_rssi, shared_y_rssi)[1][0]
            spear_shared = stats.spearmanr(shared_x_rssi, shared_y_rssi).correlation
            kendall_shared = stats.kendalltau(shared_x_rssi, shared_y_rssi).correlation
        # Difference vectors
        if len(difference_x) == 1 or statistics.variance(difference_x) == 0 or statistics.variance(difference_y) == 0:
            # No variance means no correlation
            cos_difference = cosine_similarity([difference_x], [difference_y])[0][0]
            pearson_difference = 0
            spear_difference = 0
            kendall_difference = 0
        else:
            cos_difference = cosine_similarity([difference_x], [difference_y])[0][0]
            pearson_difference = np.corrcoef(difference_x, difference_y)[1][0]
            spear_difference = stats.spearmanr(difference_x, difference_y).correlation
            kendall_difference = stats.kendalltau(difference_x, difference_y).correlation
        # Ratio vectors
        if len(ratio_x) == 1 or statistics.variance(ratio_x) == 0 or statistics.variance(ratio_y) == 0:
            # No variance means no correlation
            cos_ratio = cosine_similarity([ratio_x], [ratio_y])[0][0]
            pearson_ratio = 0
            spear_ratio = 0
            kendall_ratio = 0
        else:
            cos_ratio = cosine_similarity([ratio_x], [ratio_y])[0][0]
            pearson_ratio = np.corrcoef(ratio_x, ratio_y)[1][0]
            spear_ratio = stats.spearmanr(ratio_x, ratio_y).correlation
            kendall_ratio = stats.kendalltau(ratio_x, ratio_y).correlation
        # Rank vectors
        if len(rank_x) == 1 or statistics.variance(rank_x) == 0 or statistics.variance(rank_y) == 0:
            # No variance means no correlation
            cos_rank = cosine_similarity([rank_x], [rank_y])[0][0]
            pearson_rank = 0
            spear_rank = 0
            kendall_rank = 0
        else:
            cos_rank = cosine_similarity([rank_x], [rank_y])[0][0]
            pearson_rank = np.corrcoef(rank_x, rank_y)[1][0]
            spear_rank = stats.spearmanr(rank_x, rank_y).correlation
            kendall_rank = stats.kendalltau(rank_x, rank_y).correlation
    ### E. Difference-Based Features ###
    # Generate difference vector and values
    diff_smallest = 120
    diff_largest = 120
    diff_mean = 120
    diff_median = 120
    diff_harmonic_mean = 120
    diff_stdev = 0
    diff_pstdev = 0
    diff_vector = []
    for i in range(0, len(shared_aps)):
        diff_vector.append(abs(int(shared_x_rssi[i]) - int(shared_y_rssi[i])))
    if (len(diff_vector) > 0):
        diff_smallest = min(diff_vector)
        diff_largest = max(diff_vector)
        diff_mean = statistics.mean(diff_vector)
        diff_median = statistics.median(diff_vector)
        diff_harmonic_mean = statistics.harmonic_mean(diff_vector)
        if len(diff_vector) == 1:
            diff_stdev = 0
            diff_pstdev = 0
        else:
            diff_stdev = statistics.stdev(diff_vector)
            diff_pstdev = statistics.pstdev(diff_vector)
    # Generate pair difference vector
    pair_diff_smallest = 120
    pair_diff_largest = 120
    pair_diff_mean = 120
    pair_diff_median = 120
    pair_diff_harmonic_mean = 120
    pair_diff_stdev = 0
    pair_diff_pstdev = 0
    pair_difference = []
    for i in range(0, len(difference_x)):
        pair_difference.append(abs(difference_x[i] - difference_y[i]))
    if (len(pair_difference) > 0):
        pair_diff_smallest = min(pair_difference)
        pair_diff_largest = max(pair_difference)
        pair_diff_mean = statistics.mean(pair_difference)
        pair_diff_median = statistics.median(pair_difference)
        pair_diff_harmonic_mean = statistics.harmonic_mean(pair_difference)
        if len(pair_difference) == 1:
            pair_diff_stdev = 0
            pair_diff_pstdev = 0
        else:
            pair_diff_stdev = statistics.stdev(pair_difference)
            pair_diff_pstdev = statistics.pstdev(pair_difference)
    # Generate pair ratio vector
    pair_ratio_smallest = 120
    pair_ratio_largest = 120
    pair_ratio_mean = 120
    pair_ratio_median = 120
    pair_ratio_harmonic_mean = 120
    pair_ratio_stdev = 0
    pair_ratio_pstdev = 0
    pair_ratio = []
    for i in range(0, len(ratio_x)):
        pair_ratio.append(abs(ratio_x[i] - ratio_y[i]))
    if (len(pair_ratio) > 0):
        pair_ratio_smallest = min(pair_ratio)
        pair_ratio_largest = max(pair_ratio)
        pair_ratio_mean = statistics.mean(pair_ratio)
        pair_ratio_median = statistics.median(pair_ratio)
        pair_ratio_harmonic_mean = statistics.harmonic_mean(pair_ratio)
        if len(pair_ratio) == 1:
            pair_ratio_stdev = 0
            pair_ratio_pstdev = 0
        else:
            pair_ratio_stdev = statistics.stdev(pair_ratio)
            pair_ratio_pstdev = statistics.pstdev(pair_ratio)
    ####VI. Device Heterogeneity ####
    # RE3 score
    re3 = 0
    for i in shared_aps:
        max_x = int(x_aps_dict[i])
        max_y = int(y_aps_dict[i])
        # make the lower vector
        low_x = [x for x in x_aps_dict.keys() if (int(x_aps_dict[x]) < max_x)]
        low_y = {y for y in y_aps_dict.keys() if (int(y_aps_dict[y]) < max_y)}
        # find all the shared aps between the lower vectors
        shared_low = []
        for t in low_x:
            if t in low_y:
                shared_low.append(t)
        # Use those shared aps to calculate re3
        for j in shared_low:
            diff_x = int(x_aps_dict[j]) - max_x
            diff_y = int(y_aps_dict[j]) - max_y
            re3 = re3 + (1 - (diff_x - diff_y) / (diff_x + diff_y))

    return [manhattan, euclidean, *shared_top_z, *shared_percent_z,
            *shares_top_k, redpin_1, redpin_2, cos_shared, cos_difference, cos_ratio, cos_rank, pearson_shared,
            pearson_difference, pearson_ratio, pearson_rank, spear_shared, spear_difference, spear_ratio,
            spear_rank, kendall_shared, kendall_difference, kendall_ratio, kendall_rank, diff_smallest,
            diff_largest,
            diff_mean, diff_median, diff_harmonic_mean, diff_stdev, diff_pstdev, pair_diff_smallest,
            pair_diff_largest,
            pair_diff_mean, pair_diff_median, pair_diff_harmonic_mean, pair_diff_stdev, pair_diff_pstdev,
            pair_ratio_smallest,
            pair_ratio_largest, pair_ratio_mean, pair_ratio_median, pair_ratio_harmonic_mean, pair_ratio_stdev,
            pair_ratio_pstdev, re3]
