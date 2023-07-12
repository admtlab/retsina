import logging
import logging.handlers
import multiprocessing
import pandas as pd
import sys
from os import path
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, roc_auc_score, \
    classification_report, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_cumulative_gain, plot_lift_curve
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import pymrmr
from multiprocessing import Pool
from pathlib import Path
import dask.dataframe as dd
import traceback
from src.parquet_merger import ParquetMerger

log_file_path = 'output_figs'
log_name = 'experiments_log'
manager = multiprocessing.Manager()
log_queue = manager.Queue()
max_features_to_extract = 16
max_processes_to_use = 60
supported_classifiers = ['KNeighborsClassifier(3)', 'DecisionTreeClassifier(max_depth=5)', 'AdaBoostClassifier()',
                         'MLPClassifier(alpha=1, max_iter=1000)']


def listener_configurer(log_name, log_file_path):
    """
    Configures and returns a log file based on the given name
    
    Args:
        log_name (str): String of the log name to use
        log_file_path (str): String of the log file path
        
    Returns:
        logger: configured logger
    """
    logger = logging.getLogger(log_name)

    fh = logging.FileHandler(path.join(log_file_path, f'{log_name}.log'), encoding='utf-8')
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(log_format)
    logger.setLevel(logging.INFO)
    current_fh_names = [fh.__dict__.get('baseFilename', '') for fh in logger.handlers]

    # In order to prevent multiple logs to the same file and confusing interlaced logs
    if fh.__dict__['baseFilename'] not in current_fh_names:
        logger.addHandler(fh)

    return logger


def listener_process(queue, configurer, log_name):
    """
    Listener process is a target for a multiprocess that runs and listens to 
    a queue for logging events.
    
    Args:
        queue (multiprocessing.manager.Queue): queue to monitor
        configurer (func): function to configure the logger
        log_name (str): name of the log to use
        
    Returns:
        None
    """
    configurer(log_name, log_file_path)

    # Monitor queue for events
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            print('Failure in listener process', file=sys.stderr)
            traceback.print_last(limit=1, file=sys.stderr)


def logging_function(msg_to_log, msg_level=None, logger_name='experiments_log'):
    """
    The main function for conducting logging that will log each classifier to a
    separate logging file, which is useful when running experiments for different
    classifiers in parallel
    :param msg_to_log: The message that should be written to the log file
    :param msg_level: The level of the message
    :param logger_name: The name of the log file, used primarily to separate
                        log files for different classifiers
    """
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger(path.join(log_file_path, logger_name))
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if msg_level is not None and msg_level == 'warning':
        logger.warning(msg_to_log)
    else:
        logger.info(msg_to_log)


def build_and_test(x_tr, x_te, y_tr, y_te, figs_dir, experiment_filedir, estimator=None, num_features=8):
    """
    The main function for building the appropriate estimator/classifier, training the classifier, and
    making predictions on the testing data to observe the performance of the classifiers and output
    the corresponding figures for the results
    :param x_tr: Training data frame without the target label
    :param x_te: Testing data frame without the target label
    :param y_tr: Data frame with only the training target label column
    :param y_te: Data frame with only the testing target label column
    :param figs_dir: The filepath for outputting figures
    :param experiment_filedir: The filepath for outputting experiment results
    :param estimator: The base estimator to use in the ensemble classifier. Note that None is a decision tree
    :param num_features: The number of features that appear in x_tr and x_te, used mostly for separating experiments
                        into separate files and rows
    :return: A tuple containing the results of sklearn's roc and auc curve results
    """
    results_dir = path.join(figs_dir, experiment_filedir)
    clf_name = type(estimator).__name__

    # Build and Plot PCA
    logging_function("Building PCA Components", logger_name=clf_name)
    pca = PCA(n_components=2)
    pca.fit(x_tr)
    x_pca = pca.transform(x_tr)

    logging_function("Building Scatter Plot", logger_name=clf_name)
    fig, ax = plt.subplots()
    ax.scatter(x_pca[:, 0], x_pca[:, 1], c=y_tr, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.savefig(path.join(results_dir, f'{clf_name}-{num_features}-features-scatter.png'))
    plt.close()

    # Build and fit the model, note that estimator of None is a DecisionTree Classifier
    logging_function("Training the Model with training dataset", logger_name=clf_name)
    model = BaggingClassifier(n_estimators=300, max_features=3, estimator=estimator, n_jobs=8)
    model.fit(x_tr, y_tr)

    # Test the model
    logging_function("Making predictions with testing dataset", logger_name=clf_name)
    y_pred = model.predict(x_te)

    y_score = model.predict_proba(x_te)
    fpr0, tpr0, thresholds = roc_curve(y_te, y_score[:, 1])
    roc_auc0 = auc(fpr0, tpr0)

    # Plot metrics
    logging_function("Generating Plots...", logger_name=clf_name)
    plot_roc(y_te, y_score)
    plt.savefig(path.join(results_dir, f'{clf_name}-{num_features}-features-roc.png'))
    plt.close()

    plot_precision_recall(y_te, y_score)
    plt.savefig(path.join(results_dir, f'{clf_name}-{num_features}-features-precision_recall.png'))
    plt.close()

    plot_cumulative_gain(y_te, y_score)
    plt.savefig(path.join(results_dir, f'{clf_name}-{num_features}-features-cumulative_gain.png'))
    plt.close()

    plot_lift_curve(y_te, y_score)
    plt.savefig(path.join(results_dir, f'{clf_name}-{num_features}-features-lift_curve.png'))
    plt.close()
    plt.close('all')

    # Retrieve confusion matrix and split to each metric
    logging_function("Computing Confusion Matrix and writing output stats", logger_name=clf_name)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    tp_ratio = (tp / (tp + fn))
    fp_ratio = (fp / (fp + tn))
    tn_ratio = (tn / (tn + fp))
    fn_ratio = (fn / (fn + tp))
    tp_percent = "%.4f%%" % tp_ratio
    fp_percent = "%.4f%%" % fp_ratio
    tn_percent = "%.4f%%" % tn_ratio
    fn_percent = "%.4f%%" % fn_ratio

    # Write classification report
    with open(path.join(results_dir, f'{clf_name}-{num_features}-features-stats.txt'), 'a') as stat_file:
        stat_file.write(f'Precision score: {precision_score(y_te, y_pred)}\n'
                        f'Recall score: {recall_score(y_te, y_pred)}\n'
                        f'F1-score score: {f1_score(y_te, y_pred)}\n'
                        f'Accuracy score: {accuracy_score(y_te, y_pred)}\n'
                        f'Balanced Accuracy Score: {balanced_accuracy_score(y_te, y_pred)}\n'
                        f'True Positive Percent (Correct Close Predictions): {tp_percent}\n'
                        f'False Positive Percent (Incorrect Close Predictions): {fp_percent}\n'
                        f'True Negative Percent (Correct Far Predictions): {tn_percent}\n'
                        f'False Negative Percent (Incorrect Far Predictions): {fn_percent}\n'
                        f'Classification Report:\n{classification_report(y_te, y_pred)}')
    logging_function("Experiment complete, ending process", logger_name=clf_name)
    return roc_auc0, fpr0, tpr0


def retrieve_dataframes(dataset_name, file, figs_dir, experiment_filedir, testing_data=False,
                        selected_features=None, feature_selection_algorithm='MID',
                        num_features=8, negative_label='Far', close_samples=9000, far_samples=8000,
                        logger_name="experiments_log", filter_distance_upper_bound=3.25, filter_max_distance=20.0):
    """
    A function for retrieving dataframes for the given dataset. This function handles reading
    a dataset into a dask data frame, filtering ranges of RSSI values if needed, encoding the target label,
    performing standard scaling on each of the features, invoking the function for feature selection, and
    calling the function for randomly sampling each class to the specified ratio.
    :return: A Dask Dataframe object after each of the above steps have been performed.
    """
    pq_filepath = path.join('/', 'output_datasets', dataset_name, file)

    # Read in and process the training data
    logging_function("\tReading Data from Parquet File")
    x_df = dd.read_parquet(pq_filepath).compute()
    dist_column = 'distance'

    filtered_center_df = x_df

    if filter_distance_upper_bound is not None:
        # Filter rows based on provided distances
        filtered_center_df = x_df.query(f"`{dist_column}` < 2.25 | `{dist_column}` >= {filter_distance_upper_bound}")

    filtered_max_df = filtered_center_df

    if filter_max_distance is not None:
        filtered_max_df = filtered_center_df.query(f"`{dist_column}` <= {filter_max_distance}")

    # Reset the index (rows) after filtering
    filtered_max_df = filtered_max_df.reset_index(drop=True)

    # Encode target labels as 0 for Close and 1 for Far
    y_df = pd.DataFrame()
    y_df['proximity'] = (filtered_max_df.proximity == negative_label).astype(int)

    # Remove Labels from x_df
    filtered_max_df = filtered_max_df.drop(['proximity', 'distance', 'x_id', 'y_id'], axis=1)

    # Edge Case in All Dataset from DataFrames to CSV
    if negative_label == 1:
        filtered_max_df = filtered_max_df.drop(filtered_max_df.columns[0], axis=1)
        filtered_max_df = filtered_max_df.drop(['distance', 'x_id', 'y_id'], axis=1)

    columns_lst = filtered_max_df.columns

    # Scale the data in x to unit vector for each feature
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(filtered_max_df), columns=columns_lst)

    if not testing_data:
        (feature_x_df, feature_y_df, target_counts, selected_features) = extract_dfs(scaled_df, y_df, figs_dir,
                                                                                     experiment_filedir,
                                                                                     testing_data, selected_features,
                                                                                     feature_selection_algorithm,
                                                                                     num_features)

        # Randomly Sample each class
        sampled_df = random_sample_classes(feature_x_df.join(y_df),
                                           close_samples=close_samples,
                                           far_samples=far_samples, logger_name=logger_name)

        # Encode target labels as 0 for Close and 1 for Far
        y_df = sampled_df[['proximity']].copy()
        sampled_df = sampled_df.drop('proximity', axis=1)

        return sampled_df, y_df, target_counts, selected_features

    # Else if training data
    (feature_x_df, feature_y_df, target_counts) = extract_dfs(scaled_df, y_df, figs_dir, experiment_filedir,
                                                              testing_data, selected_features,
                                                              feature_selection_algorithm, num_features)
    # Note that we do not randomly sample for the testing dataset
    return feature_x_df, feature_y_df, target_counts


def random_sample_classes(x_df, negative_label=1, close_samples=9000, far_samples=8000, random_state=29,
                          logger_name="experiments_log"):
    """
    A function for sampling the rows in x_df to the specified close/far ratio using the
    random state as a seed, if provided.
    :return: A dataframe object with the sampled rows of x_df for both classes
    """
    logging_function(f"Randomly sampling Data for {close_samples} Close Samples and {far_samples} Far Samples...\n")

    far_df = x_df.query(f"proximity == {negative_label}")
    close_df = x_df.query(f"proximity != {negative_label}")
    with_replacement = False

    # Verify that each DataFrame has sufficient rows for samples
    if far_df.shape[0] < far_samples:
        logging_function(
            f"[Warning]\tRequested {far_samples} samples of Far, but DataFrame only has {far_df.shape[0]} samples",
            msg_level='warning', logger_name=logger_name)
        logging_function("[Warning]\tUsing replacement of the samples from Far for requested number of samples",
                         msg_level='warning', logger_name=logger_name)
        with_replacement = True

    if close_df.shape[0] < close_samples:
        logging_function(
            f"[Warning]\tRequested {close_samples} samples of Close, but DataFrame only has {close_df.shape[0]} samples",
            msg_level='warning', logger_name=logger_name)
        logging_function("[Warning]\tUsing replacement of the samples from Close for requested number of samples",
                         msg_level='warning', logger_name=logger_name)
        with_replacement = True

    logging_function("Sampling Far Datapoints\n")
    far_df = far_df.sample(n=far_samples, random_state=random_state, replace=with_replacement)
    logging_function("Sampling Close Datapoints\n")
    close_df = close_df.sample(n=close_samples, random_state=random_state, replace=with_replacement)
    logging_function("Combining Sampled Datasets into one DataFrame")
    return (pd.concat([far_df, close_df], axis=0)).drop_duplicates().reset_index(drop=True)


def under_sample_df(x_df, y_df, random_state=29):
    """
    A simple function for undersampling a dataframe. NOTE that
    this function was replaced by the function random_sample_classes()
    """
    under_sampler = RandomUnderSampler(random_state=random_state)
    under_sample_x, under_sample_y = under_sampler.fit_resample(x_df, y_df)
    under_target_counts = under_sample_y['proximity'].value_counts()
    return under_sample_x, under_sample_y, under_target_counts


def extract_dfs(x_df, y_df, figs_dir, experiment_filedir, testing_data=False,
                selected_features=None, feature_selection_algorithm='MID',
                num_features=8):
    """
    A function that runs the feature selection algorithm on the provided
    dataframe and reduces the dataframe to only contain the features that
    are selected by the algorithm
    :return: A tuple containing Dataframe x_df after feature selection, Dataframe y_df with the
            matching rows to x_df, and the number of rows in the returned Dataframe
    """
    target_counts = y_df['proximity'].value_counts()

    logging_function("\tPerforming Feature Selection using pymRMR Algorithm")
    if not testing_data:
        # Use Feature Selection Reduction Algorithm mRMR with MID algorithm
        # Note that y_df comes first as pymRMR requires target label in first column
        selection_df = y_df.join(x_df)

        selected_features = pymrmr.mRMR(selection_df, feature_selection_algorithm, num_features)

        # Add these back in to test final experiment
        # selected_features.append('x_num_aps')
        # selected_features.append('y_num_aps')

    not_selected_features = [x for x in x_df if x not in selected_features]

    # Write selected features to file
    with open(path.join(figs_dir, experiment_filedir, f'{num_features}-features.txt'), 'a') as feature_file:
        feature_file.write(f'Selected Features: {selected_features}\n')

    # Drop features that were not selected by MRMR algorithm
    for excluded_feature in not_selected_features:
        x_df = x_df.drop(excluded_feature, axis=1)

    if not testing_data:
        return x_df, y_df, target_counts, selected_features

    return x_df, y_df, target_counts


def single_dataset_experiments(exp_num, lst_of_classifiers, close_samples=9000, far_samples=8000, num_processes=8,
                               features_to_use=8, custom_training_dataset=None, custom_testing_dataset=None):
    """
    The main function for running experiments that used a single dataset for the
    training dataset (i.e., not merged). This function will call the necessary
    functions for extracting Dataframes and running experiments.
    """
    logging_function(f"\tBeginning Experiment {exp_num}")
    train_dataset_name: str
    test_dataset_name: str
    experiment_dir: str
    training_file = 'trainingData.parquet'
    testing_file = 'testingData.parquet'
    negative_label = 'Far'
    filter_distance_upper_bound = 3.25
    filter_distance_max_threshold = 20

    match exp_num:
        case 1:
            train_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            test_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            experiment_dir = 'train-uji-test-uji'
        case 2:
            train_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            test_dataset_name = '10-JUIndoorLoc'
            experiment_dir = 'train-uji-test-jui'
        case 3:
            train_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            test_dataset_name = '11-UJIndoorLoc'
            experiment_dir = 'train-uji-test-ujiLoc'
        case 4:
            train_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            test_dataset_name = path.join('13-IPIN-Tutorial', 'testing')
            experiment_dir = 'train-uji-test-ipin'
            filter_distance_upper_bound = None
            filter_distance_max_threshold = None
        case 5:
            train_dataset_name = '11-UJIndoorLoc'
            test_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            experiment_dir = 'train-ujiLoc-test-uji'
        case 6:
            # Train UJI, Test JUI
            train_dataset_name = '11-UJIndoorLoc'
            test_dataset_name = '10-JUIndoorLoc'
            experiment_dir = 'train-ujiLoc-test-jui'
        case 7:
            # Train UJI, Test UJI
            train_dataset_name = '11-UJIndoorLoc'
            test_dataset_name = '11-UJIndoorLoc'
            experiment_dir = 'train-ujiLoc-test-ujiLoc'
        case 8:
            train_dataset_name = '11-UJIndoorLoc'
            test_dataset_name = path.join('13-IPIN-Tutorial', 'testing')
            experiment_dir = 'train-ujiLoc-test-ipin'
            filter_distance_upper_bound = None
            filter_distance_max_threshold = None
        case 9:
            train_dataset_name = path.join('13-IPIN-Tutorial', 'training')
            test_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            experiment_dir = 'train-ipin-test-uji'
        case 10:
            train_dataset_name = path.join('13-IPIN-Tutorial', 'training')
            test_dataset_name = '10-JUIndoorLoc'
            experiment_dir = 'train-ipin-test-jui'
        case 11:
            train_dataset_name = path.join('13-IPIN-Tutorial', 'training')
            test_dataset_name = '11-UJIndoorLoc'
            experiment_dir = 'train-ipin-test-ujiLoc'
        case 12:
            train_dataset_name = path.join('13-IPIN-Tutorial', 'training')
            test_dataset_name = path.join('13-IPIN-Tutorial', 'testing')
            experiment_dir = 'train-ipin-test-ipin'
            filter_distance_upper_bound = None
            filter_distance_max_threshold = None
        case 13:
            # Custom training, testing with UJI dataset
            train_dataset_name = custom_training_dataset
            test_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            experiment_dir = 'train-custom-test-uji'
        case 14:
            # Custom training, testing with JUI dataset
            train_dataset_name = custom_training_dataset
            test_dataset_name = '10-JUIndoorLoc'
            experiment_dir = 'train-custom-test-jui'
        case 15:
            # Custom training, testing with UJILoc
            train_dataset_name = custom_training_dataset
            test_dataset_name = '11-UJIndoorLoc'
            experiment_dir = 'train-custom-test-ujiLoc'
        case 16:
            # Custom training, testing with IPIN
            train_dataset_name = custom_training_dataset
            test_dataset_name = path.join('13-IPIN-Tutorial', 'testing')
            experiment_dir = 'train-custom-test-ipin'
        case 17:
            # UJI training, custom testing
            train_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            test_dataset_name = custom_testing_dataset
            experiment_dir = 'train-uji-test-custom'
        case 18:
            # UJILoc training, custom testing
            train_dataset_name = '11-UJIndoorLoc'
            test_dataset_name = custom_testing_dataset
            experiment_dir = 'train-ujiLoc-test-custom'
        case 19:
            # IPIN training, custom testing
            train_dataset_name = path.join('13-IPIN-Tutorial', 'training')
            test_dataset_name = custom_testing_dataset
            experiment_dir = 'train-ipin-test-custom'
        case 20:
            # custom training, custom testing
            train_dataset_name = custom_training_dataset
            test_dataset_name = custom_testing_dataset
            experiment_dir = 'train-custom-test-custom'
        case _:
            logger.warning('[Warning]\tPlease use a valid experiment number')
            return

    logging_function("\tCollecting Training Data...")
    under_sample_train_x, under_sample_train_y, train_target_counts, selected_features = retrieve_dataframes(
        train_dataset_name, training_file, 'output_figs', experiment_dir, negative_label=negative_label,
        close_samples=close_samples, far_samples=far_samples, num_features=features_to_use)

    logging_function("\tCollecting Testing Data...")
    under_sample_test_x, under_sample_test_y, test_target_counts = retrieve_dataframes(
        test_dataset_name, testing_file, 'output_figs', experiment_dir, testing_data=True,
        selected_features=selected_features,
        close_samples=close_samples, far_samples=far_samples, filter_distance_upper_bound=filter_distance_upper_bound,
        filter_max_distance=filter_distance_max_threshold, num_features=features_to_use)

    # For each classifier, add a new element to experiment_input_lst
    experiment_input_lst = []
    for curr_clf in lst_of_classifiers:
        experiment_input_lst.append((experiment_dir, train_target_counts, test_target_counts, under_sample_train_x,
                                     under_sample_test_x, under_sample_train_y, under_sample_test_y, curr_clf))

    # Run each of the experiment inputs in parallel
    logging_function("\tSwitching over to classifier processes for experiments...")
    with Pool(num_processes) as p:
        logger.info('Test from before splitting to child processes')
        p.starmap(classifier_run_experiments, experiment_input_lst)


def classifier_run_experiments(experiment_dir, train_target_counts, test_target_counts, under_sample_train_x,
                               under_sample_test_x, under_sample_train_y, under_sample_test_y, estimator=None,
                               num_features=8):
    """
    An intermediary helper function for running experiments that acts as a target function
    for the multiprocessing library to facilitate the parallel execution of experiments.
    """
    clf_name = type(estimator).__name__
    logging_function(f"{clf_name}:\tRunning the Experiment...", logger_name=f"{clf_name}")
    filepath = path.join('output_figs', experiment_dir, f'{experiment_dir}-{num_features}-features-stats.txt')
    with open(filepath, 'w') as stats_file:
        stats_file.write(f'Training Target Label Counts (0 = Close, 1 = Far)\n{train_target_counts}\n'
                         f'Test Target Label Counts (0 = Close, 1 = Far)\n{test_target_counts}\n')

    # Run build and test to see results
    roc_auc_imb, fpr_imb, tpr_imb = build_and_test(under_sample_train_x.to_numpy(),
                                                   under_sample_test_x.to_numpy(),
                                                   under_sample_train_y.values.ravel(),
                                                   under_sample_test_y.values.ravel(),
                                                   'output_figs',
                                                   experiment_dir, estimator=estimator, num_features=num_features)


def multi_dataset_experiments(lst_of_classifiers, dataset_to_test, close_samples=9000, far_samples=8000,
                              num_processes=8, num_features=8, custom_testing_dataset=None):
    """
    A function for running experiments where each dataset is merged into a single parquet file
    for the training dataset and a testing dataset is specified
    :param lst_of_classifiers: The list of classifiers to run the experiments with
    :param close_samples: the number of samples labelled Close in the training data
    :param far_samples: the number of samples labelled Far in the training data
    :param num_processes: the number of processes for parallelization
    :param dataset_to_test: the dataset to use for testing data
    :param num_features: the number of features to extract from pymRMR
    """
    logging_function(f"\tRunning experiments for {num_features} features")
    training_filename = "trainingData.parquet"
    testing_filename = "testingData.parquet"
    train_dataset_name = "merged_datasets"
    test_dataset_name: str
    experiment_dir: str
    negative_label = "Far"
    filter_distance_upper_bound = 3.25
    filter_distance_max_threshold = 20

    # Set up a list of filepaths to parquet files
    pq_train_lst = [Path("/") / "output_datasets" / "09-UJI_LIB_DB_v2" / "00" / training_filename,
                    Path("/") / "output_datasets" / "11-UJIndoorLoc" / training_filename,
                    Path("/") / "output_datasets" / "13-IPIN-Tutorial" / "training" / training_filename]

    # filepath for discovering or creating a merged parquet file
    merged_training_filepath = Path("/") / "output_datasets" / train_dataset_name

    # Ensure the merged parquet file is set up as needed for training
    logging_function("\tSetting up merged parquet file, if needed")
    pq_merger = ParquetMerger()
    pq_merger.write_merged_parquet(pq_train_lst, merged_training_filepath, training_filename,
                                   number_of_processes=num_processes,
                                   negative_label=negative_label, close_samples=close_samples, far_samples=far_samples)

    match dataset_to_test:
        case 9:
            test_dataset_name = path.join('09-UJI_LIB_DB_v2', '00')
            experiment_dir = 'test-uji'
            filter_distance_upper_bound = None
        case 10:
            test_dataset_name = '10-JUIndoorLoc'
            experiment_dir = 'test-jui'
            filter_distance_upper_bound = None
        case 11:
            test_dataset_name = '11-UJIndoorLoc'
            experiment_dir = 'test-ujiLoc'
            filter_distance_upper_bound = None
        case 13:
            test_dataset_name = path.join('13-IPIN-Tutorial', 'testing')
            experiment_dir = 'test-ipin'
            filter_distance_upper_bound = None
            filter_distance_max_threshold = None
        case 14:
            test_dataset_name = custom_testing_dataset
            experiment_dir = 'test-custom'
            filter_distance_upper_bound = None
            filter_distance_max_threshold = None
        case _:
            logger.warning('[Warning]\tPlease use a valid dataset number')
            return

    logging_function("\tCollecting Training Data...")
    under_sample_train_x, under_sample_train_y, train_target_counts, selected_features = retrieve_dataframes(
        train_dataset_name, training_filename, 'output_figs', experiment_dir, negative_label=negative_label,
        close_samples=close_samples, far_samples=far_samples, num_features=num_features,
        filter_distance_upper_bound=filter_distance_upper_bound)

    logging_function("\tCollecting Testing Data...")
    under_sample_test_x, under_sample_test_y, test_target_counts = retrieve_dataframes(
        test_dataset_name, testing_filename, 'output_figs', experiment_dir, testing_data=True,
        selected_features=selected_features,
        close_samples=close_samples, far_samples=far_samples, filter_distance_upper_bound=filter_distance_upper_bound,
        filter_max_distance=filter_distance_max_threshold, num_features=num_features)

    # For each classifier, add a new element to experiment_input_lst
    experiment_input_lst = []
    for curr_clf in lst_of_classifiers:
        experiment_input_lst.append((experiment_dir, train_target_counts, test_target_counts, under_sample_train_x,
                                     under_sample_test_x, under_sample_train_y, under_sample_test_y, curr_clf,
                                     num_features))

    # Run each of the experiment inputs in parallel
    logging_function("\tSwitching over to classifier processes for experiments...")
    with Pool(num_processes) as p:
        logger.info('Test from before splitting to child processes')
        p.starmap(classifier_run_experiments, experiment_input_lst)


def get_user_input():
    """
    A helper function for prompting the user for input to run experiments
    and checking for valid inputs to the system.
    :return: A tuple containing all parameters that the user is prompted for
    """
    valid_input = False
    num_training_datasets: str
    training_dir_name = None
    testing_dir_name = None
    close_ratio: int
    far_ratio: int
    num_user_processes: int

    # How many datasets should be used to train and which
    while not valid_input:
        num_training_datasets = input("How many datasets should the classifier be trained on? [all, one] ")
        match num_training_datasets.lower():
            case "all":
                valid_input = True
                print("Using all of the provided datasets as training data\n")
            case "one":
                print("Using only a single dataset for training data\n")
                valid_input = True
            case _:
                print("Please enter a valid option of either all or one.\n")

    # Which datasets should be used if only a single dataset is to be used
    if num_training_datasets == 'one':
        valid_input = False
        while not valid_input:
            training_dir_name = input(
                "What is the directory name for the dataset? (It is assumed that the directory exists within the datasets directory) ")
            # Check that the provided directory exists
            if Path(Path(".") / "datasets" / training_dir_name).exists():
                valid_input = True
                print(f"Using the {training_dir_name} dataset under the datasets/ directory.\n")
            else:
                print(
                    "Please enter a valid dataset that exists within the datasets/ directory. Note that if you are trying to use a new dataset, it must appear in the datasets/ directory.\n")

    # Which dataset should be used for testing
    valid_input = False
    while not valid_input:
        testing_dir_name = input(
            "What is the directory name for the testing dataset? (It is assumed that the directory exists within the datasets directory) ")
        # Check that the provided directory exists
        if Path(Path(".") / "datasets" / testing_dir_name).exists():
            valid_input = True
            print(f"Using the {testing_dir_name} dataset under the datasets/ directory.\n")
        else:
            print(
                "Please enter a valid dataset that exists within the datasets/ directory. Note that if you are trying to use a new dataset, it must appear in the datasets/ directory.\n")

    # How many features to extract with pymrmr
    valid_input = False
    while not valid_input:
        try:
            features_to_use = int(input("How many features would you like extracted using pymRMR? "))
            if 0 < features_to_use < max_features_to_extract:
                valid_input = True
                print(f"pyMRMR will extract {features_to_use} features\n")
            elif features_to_use < 0:
                print("The number of features to be extracted by pyMRMR must be greater than 0\n")
            else:
                print(f"The number of features to be extracted by pyMRMR cannot exceed {max_features_to_extract}\n")
        except ValueError:
            print("The number of features to extract must be an integer value. Please try again.\n")

    # Extract the balance between close/far samples during training
    valid_input = False
    while not valid_input:
        try:
            balance_ratio_str = input(
                "What is the balance of Close and Far Samples? Please enter the balance as Close:Far ")
            ratio_lst = balance_ratio_str.split(":")
            if len(ratio_lst) != 2:
                print("Did not detect two values, please try again.\n")
            else:
                close_ratio = int(ratio_lst[0])
                far_ratio = int(ratio_lst[1])
                if (far_ratio + close_ratio) != 100:
                    print("The two balanced ratios should sum to 100%, please try again.\n")
                else:
                    valid_input = True
        except ValueError:
            print("The ratios to use for Close and Far samples must be integer values. Please try again.\n")

    # Extract which classifiers should be used
    # note supported classifiers list
    print("The system currently supports the following classifiers:\n")
    display_counter = 0
    menu_choice = 0
    for supported_clf in supported_classifiers:
        print(f"\t{display_counter}. {supported_clf}\n")
        display_counter += 1
    selected_classifiers = []
    unique_selected_classifiers = {}
    while menu_choice != -1:
        try:
            menu_choice = int(input(
                "Please enter the number of the classifier you wish to include or -1 to finish choosing classifiers "))
            if menu_choice > len(supported_classifiers) or (menu_choice < 1 and menu_choice != -1):
                print('Please select a valid classifier choice\n')
            elif menu_choice not in unique_selected_classifiers:
                unique_selected_classifiers[menu_choice] = True
                chosen_clf = str_to_clf(supported_classifiers[menu_choice])
                if chosen_clf is not None:
                    selected_classifiers.append(chosen_clf)
        except ValueError:
            print("The menu choice must be an integer. Please try again.\n")

    # How many processes to use for the classifiers
    valid_input = False
    while not valid_input:
        try:
            num_user_processes = int(input("Please enter the number of processes you wish to use as an integer "))
            if not 0 < num_user_processes < max_processes_to_use:
                print(f"Please enter a number of processes that is greater than 0 and less than {max_processes_to_use}")
            else:
                valid_input = True
        except ValueError:
            print("The number of processes must be an integer. Please try again.\n")

    return num_training_datasets, training_dir_name, testing_dir_name, features_to_use, close_ratio, far_ratio, selected_classifiers, num_user_processes


def str_to_clf(clf_name):
    """
    A helper function for matching string classifier names to the corresponding
    constructor in Python used by the system. This function is used to allow
    users to enter classifiers as strings, then map to the correct object.
    :param clf_name: The name of the desired and supported classifier.
    :return: The mapped classifier object or None for invalid options
    """
    match clf_name:
        case 'DecisionTreeClassifier':
            return DecisionTreeClassifier(max_depth=5)
        case 'AdaBoostClassifier':
            return AdaBoostClassifier()
        case 'KNeighborsClassifier':
            return KNeighborsClassifier(3)
        case 'MLPClassifier':
            return MLPClassifier(alpha=1, max_iter=1000)
        case 'SVC':
            return SVC(kernel="linear", C=0.025)
        case _:
            return None


if __name__ == '__main__':
    # If output directories don't exist, make them
    dir_lst = [Path(Path(".") / "output_figs" / "train-uji-test-uji"),
               Path(Path(".") / "output_figs" / "train-uji-test-jui"),
               Path(Path(".") / "output_figs" / "train-uji-test-ujiLoc"),
               Path(Path(".") / "output_figs" / "train-uji-test-ipin"),
               Path(Path(".") / "output_figs" / "train-ujiLoc-test-uji"),
               Path(Path(".") / "output_figs" / "train-ujiLoc-test-jui"),
               Path(Path(".") / "output_figs" / "train-ujiLoc-test-ujiLoc"),
               Path(Path(".") / "output_figs" / "train-ujiLoc-test-ipin"),
               Path(Path(".") / "output_figs" / "train-ipin-test-uji"),
               Path(Path(".") / "output_figs" / "train-ipin-test-jui"),
               Path(Path(".") / "output_figs" / "train-ipin-test-ujiLoc"),
               Path(Path(".") / "output_figs" / "train-ipin-test-ipin"),
               Path(Path(".") / "output_figs" / "train-custom-test-uji"),
               Path(Path(".") / "output_figs" / "train-custom-test-jui"),
               Path(Path(".") / "output_figs" / "train-custom-test-ujiLoc"),
               Path(Path(".") / "output_figs" / "train-custom-test-ipin"),
               Path(Path(".") / "output_figs" / "train-uji-test-custom"),
               Path(Path(".") / "output_figs" / "train-ujiLoc-test-custom"),
               Path(Path(".") / "output_figs" / "train-ipin-test-custom"),
               Path(Path(".") / "output_figs" / "train-custom-test-custom"),
               Path(Path(".") / "output_figs" / "test-uji"),
               Path(Path(".") / "output_figs" / "test-jui"),
               Path(Path(".") / "output_figs" / "test-ujiLoc"),
               Path(Path(".") / "output_figs" / "test-ipin"),
               Path(Path(".") / "output_figs" / "test-custom")]
    for curr_dir in dir_lst:
        if not Path(curr_dir).exists():
            Path(curr_dir).mkdir(parents=True, exist_ok=True)

    (num_training_datasets, training_dir_name, testing_dir_name, features_to_use, close_ratio, far_ratio,
     selected_classifiers, num_user_processes) = get_user_input()

    if num_training_datasets == 'one':
        print(
            f"Input Summary:\nTraining with the {training_dir_name} dataset\nTesting with the {testing_dir_name} dataset\nSelecting {features_to_use} features\nUsing a {close_ratio}/{far_ratio} split\nUsing the following classifiers: {selected_classifiers}\nUsing {num_user_processes} processes\n")
    else:
        print(
            f"Input Summary\nTraining with all datasets\nTesting with the {testing_dir_name} dataset\nSelecting {features_to_use} features\nUsing a {close_ratio}/{far_ratio} split\nUsing the following classifiers: {selected_classifiers}\nUsing {num_user_processes} processes\n")

    # Add root listener for logging
    logger = listener_configurer(log_name, log_file_path)

    listener = multiprocessing.Process(target=listener_process, args=(log_queue, listener_configurer, log_name))

    listener.start()

    # Setup logging, one process for each classifier file
    classifier_listeners = []
    for clf in selected_classifiers:
        classifier_name = type(clf).__name__
        logger = listener_configurer(classifier_name, log_file_path)

        clf_listener = multiprocessing.Process(target=listener_process,
                                               args=(log_queue, listener_configurer, classifier_name))
        classifier_listeners.append(clf_listener)

        # Start the listener logging process
        clf_listener.start()

    # Run the start of the experiments for data retrieval
    print('***Writing RETSINA updates to log files under /output_figs/*.log***\n')
    if num_training_datasets == 'one':
        if training_dir_name == '09-UJI_LIB_DB_v2':
            if testing_dir_name == '09-UJI_LIB_DB_v2':
                single_dataset_experiments(1, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '10-JUIndoorLoc':
                single_dataset_experiments(2, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '11-UJIndoorLoc':
                single_dataset_experiments(3, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '13-IPIN-Tutorial':
                single_dataset_experiments(4, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            else:
                # If using a custom dataset to test
                single_dataset_experiments(17, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_testing_dataset=testing_dir_name)
        elif training_dir_name == '11-UJIndoorLoc':
            if testing_dir_name == '09-UJI_LIB_DB_v2':
                single_dataset_experiments(5, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '10-JUIndoorLoc':
                single_dataset_experiments(6, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '11-UJIndoorLoc':
                single_dataset_experiments(7, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '13-IPIN-Tutorial':
                single_dataset_experiments(8, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            else:
                # If using a custom dataset to test
                single_dataset_experiments(18, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_testing_dataset=testing_dir_name)
        elif training_dir_name == '13-IPIN-Tutorial':
            if testing_dir_name == '09-UJI_LIB_DB_v2':
                single_dataset_experiments(9, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '10-JUIndoorLoc':
                single_dataset_experiments(10, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '11-UJIndoorLoc':
                single_dataset_experiments(11, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            elif testing_dir_name == '13-IPIN-Tutorial':
                single_dataset_experiments(12, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes)
            else:
                # If using a custom dataset to test
                single_dataset_experiments(19, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_testing_dataset=testing_dir_name)
        else:
            # If using a custom dataset to train
            if testing_dir_name == '09-UJI_LIB_DB_v2':
                single_dataset_experiments(13, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_training_dataset=training_dir_name)
            elif testing_dir_name == '10-JUIndoorLoc':
                single_dataset_experiments(14, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_training_dataset=training_dir_name)
            elif testing_dir_name == '11-UJIndoorLoc':
                single_dataset_experiments(15, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_training_dataset=training_dir_name)
            elif testing_dir_name == '13-IPIN-Tutorial':
                single_dataset_experiments(16, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_training_dataset=training_dir_name)
            else:
                # If using a custom dataset to train and test
                single_dataset_experiments(20, selected_classifiers, close_samples=close_ratio, far_samples=far_ratio,
                                           features_to_use=features_to_use, num_processes=num_user_processes,
                                           custom_training_dataset=training_dir_name,
                                           custom_testing_dataset=testing_dir_name)
    else:
        if testing_dir_name == '09-UJI_LIB_DB_v2':
            multi_dataset_experiments(selected_classifiers, 9, close_samples=close_ratio, far_samples=far_ratio,
                                      num_features=features_to_use, num_processes=num_user_processes)
        elif testing_dir_name == '10-JUIndoorLoc':
            multi_dataset_experiments(selected_classifiers, 10, close_samples=close_ratio, far_samples=far_ratio,
                                      num_features=features_to_use, num_processes=num_user_processes)
        elif testing_dir_name == '11-UJIndoorLoc':
            multi_dataset_experiments(selected_classifiers, 11, close_samples=close_ratio, far_samples=far_ratio,
                                      num_features=features_to_use, num_processes=num_user_processes)
        elif testing_dir_name == '13-IPIN-Tutorial':
            multi_dataset_experiments(selected_classifiers, 13, close_samples=close_ratio, far_samples=far_ratio,
                                      num_features=features_to_use, num_processes=num_user_processes)
        else:
            # If testing with a custom dataset
            multi_dataset_experiments(selected_classifiers, 14, close_samples=close_ratio, far_samples=far_ratio,
                                      num_features=features_to_use, num_processes=num_user_processes,
                                      custom_testing_dataset=testing_dir_name)

    # Uncomment below lines for experiments from IPIN 2023 paper
    # classifiers = [KNeighborsClassifier(3, n_jobs=20), AdaBoostClassifier(), MLPClassifier(alpha=1, max_iter=1000)]
    # for i in [5, 8, 9, 10, 11, 12]:
    #     single_dataset_experiments(i, classifiers, close_samples=11000, far_samples=8000)

    # for i in [9, 10, 11, 13]:
    #     multi_dataset_experiments(classifiers, i, close_samples=6800, far_samples=10200, num_features=12)

    # Cleanup and end the logger listening process
    log_queue.put_nowait(None)  # Ends the queue
    listener.join()  # joins the listener process back in

    # Join the listeners for each classifier
    for clf_listener in classifier_listeners:
        clf_listener.join()
