import random

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.stats import stats
from sklearn.decomposition import PCA

from constants import CLASS_LABEL_TO_AVERAGE_REP_DURATION, EXERCISE_NAME_TO_CLASS_LABEL
from data_loading import get_exercise_readings, calculate_longest_and_shortest_rep_per_exercise
#### Helper functions
from utils import print_sequence_array_tight

global fs, lowcut, highcut


## Helper Functions
def butter_band(x):
    return butter_bandpass_filter(x, lowcut, highcut, fs, order=6)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_right_side_zeros(preds, i):
    zeros = 0
    for j in range(i + 1, len(preds)):
        if 0 == preds[j]:
            zeros += 1
        else:
            break
    return zeros


def get_left_side_zeros(preds, i):
    zeros = 0
    for j in range(i, -1, -1):
        if 0 == preds[j]:
            zeros += 1
        elif preds[j] and zeros > 0:
            break
    return zeros


def get_map_mode(map):
    max = -1
    max_key = -1
    for key, value in map.items():
        if value >= max and key > max_key:
            max = value
            max_key = key
    return max_key


def get_stats_for_k(a, k):
    if k not in a:
        return {"min": 0, "mode": 0, "max": 0}
    consecutives_ks = 0
    map = {}
    i = -1
    for point in a:
        i += 1
        if point == k:
            consecutives_ks += 1
            if i == len(a) - 1:
                # we reach the end
                if consecutives_ks in map.keys():
                    map[consecutives_ks] += 1
                else:
                    map[consecutives_ks] = 1
        else:
            if consecutives_ks != 0:
                if consecutives_ks in map.keys():
                    map[consecutives_ks] += 1
                else:
                    map[consecutives_ks] = 1
                consecutives_ks = 0

    mode = get_map_mode(map)
    # mode = (max(map.iteritems(), key=operator.itemgetter(1)))

    min_k = min(map.iterkeys())
    max_k = max(map.iterkeys())
    return {"min": min_k, "mode": mode, "max": max_k}


def smoothing(sequence, mode, factor):
    sequence = np.pad(sequence, (4, 4), 'constant', constant_values=(sequence[0], sequence[-1]))
    min_sequence_length = max(int(mode * factor), 1)
    window_length = min_sequence_length * 2 - 1
    if sequence.shape < window_length:
        return sequence
    new_sequence = np.asarray([])
    for i in range(0, sequence.shape[0] - window_length):
        prevailing_number = stats.mode(sequence[i:i + window_length])[0][0]
        new_sequence = np.append(new_sequence, prevailing_number)
    return new_sequence.astype(np.int32)



def rep_counting_model_pre_processing(X, padding=0):
    max_length = -1
    for sequence in X:
        if len(sequence) > max_length:
            max_length = len(sequence)
    shape = (len(X), max_length)
    if padding == 0:
        padded_X = np.zeros((shape))
    else:
        padded_X = np.ones((shape)) * padding
    for i in range(0, len(X)):
        padded_X[i] = X[i]
    return padded_X


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)).astype(np.float32) / y_true.astype(np.float)) * 100


def get_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))


def get_max_absolute_error(y_true, y_pred):
    return np.max(np.abs((y_true - y_pred)))


def get_error_mode(y_true, y_pred):
    return stats.mode((np.abs((y_true - y_pred))))


def get_0_error_accuracy(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    zero_errors = errors[errors == 0]
    return float(zero_errors.shape[0]) / float(y_true.shape[0])


def get_0_errors(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    zero_errors = errors[errors == 0]
    return len(zero_errors)


def get_1_error_accuracy(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    one_errors = errors[errors == 1]
    return float(one_errors.shape[0]) / float(y_true.shape[0])


def get_1_errors(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    one_errors = errors[errors == 1]
    return len(one_errors)


def get_2_error_accuracy(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    two_errors = errors[errors == 2]
    return float(two_errors.shape[0]) / float(y_true.shape[0])


def get_2_errors(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    two_errors = errors[errors == 2]
    return len(two_errors)


def more_than_two_errors_accuracy(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    two_errors = errors[errors > 2]
    return float(two_errors.shape[0]) / float(y_true.shape[0])


def more_than_two_errors(y_true, y_pred):
    errors = np.abs((y_true - y_pred))
    more_errors = errors[errors > 2]
    return len(more_errors)


def generate_fake_data(n_samples, length):
    X = np.ones((n_samples, length)) * (-1)
    X_noisy = np.ones((n_samples, length)) * (-1)
    labels = np.zeros(n_samples)
    ones_seq_length = [3, 4, 5, 6, 7, 8]
    for i in range(0, X.shape[0]):
        ones_index = random.randint(0, len(ones_seq_length) - 1)
        ones = ones_seq_length[ones_index]
        zeros = ones + int(np.random.uniform(0.2, 1) * ones)
        one_rep = np.concatenate((np.ones(ones), np.zeros(zeros)))
        stop = int(length * np.random.uniform(0.5, 1))
        j = 0
        reps = 0
        # add various kinds of length of the rep, always +-1
        while j + one_rep.shape[0] <= stop:
            X[i, j:j + one_rep.shape[0]] = one_rep
            j = j + one_rep.shape[0]
            reps += 1
        indexes = random.sample(range(0, stop), int(length * 0.02))
        noisy_version = np.copy(X[i, :])
        noisy_version[indexes] = 1 - noisy_version[indexes]
        X_noisy[i, :] = noisy_version
        labels[i] = reps
        i += 1
    return (X, X_noisy, labels)


## reps counting functions

def count_by_contraction(sequence, console_log=False):
    if console_log:
        print("contraction")
    sum = 0
    previous = -1
    for s in sequence:
        if s == 1 and previous != 1:
            sum += 1
        previous = s
    return sum


def split_per_exercise_type(sequence_labels, X, y_truth, preds, GT_of_loo_test):
    map = {}
    for label in EXERCISE_NAME_TO_CLASS_LABEL.keys():
        if label == "Null":
            continue
        label_indexes = np.argwhere(sequence_labels == label)
        entry = {"X": X[label_indexes], "y_truth": y_truth[label_indexes].squeeze(),
                 "preds": preds[label_indexes].squeeze(), "GT_of_loo_test": GT_of_loo_test[label_indexes]}
        map[label] = entry
    return map


def get_predictions_by_method(X, y, GT_of_loo_test, method, sequence_labels):
    preds = []
    for x in X:
        preds.append(method(x))
    preds = np.asarray(preds)
    print_sequence_prediction_results(GT_of_loo_test, X, preds, sequence_labels, y)


def print_sequence_prediction_results(GT_of_loo_test, X, preds, sequence_labels, y):
    map = split_per_exercise_type(sequence_labels, X, y, preds, GT_of_loo_test)
    zero_errors = []
    one_errors = []
    two_errors = []
    more_errors = []
    for label in map.keys():
        print(label)
        # print_sequences_with_error(map[label]["X"], map[label]["y_truth"], map[label]["preds"],
        #                            map[label]["GT_of_loo_test"])
        errors = get_errors_portions(map[label]["preds"], map[label]["y_truth"])
        print_stats(map[label]["preds"], map[label]["y_truth"])
        zero_errors.append(errors[0])
        one_errors.append(errors[1])
        two_errors.append(errors[2])
        more_errors.append(errors[3])

    from print_and_plot_final_results import plot_stacked_barchart
    plot_stacked_barchart(map.keys(), zero_errors, one_errors, two_errors, more_errors)


def print_sequences_with_error(X, y_true, y_preds, GT_of_loo_test):
    errors = np.abs((y_true - y_preds))
    errors_indexes = np.argwhere(errors > 0)
    # plot_error_distribution(errors)

    np.set_printoptions(linewidth=np.inf)
    for id in errors_indexes:
        print_sequence_array_tight(X[id][X[id] >= 0])
        print("smoothed")
        ones_mode = get_stats_for_k(X[id][X[id] >= 0], 1)["mode"]
        print_sequence_array_tight(smoothing(X[id][X[id] >= 0], mode=ones_mode, factor=0.5))
        print_sequence_array_tight(GT_of_loo_test[id][X[id] >= 0])
        print(y_preds[id])
        print(y_true[id])
    # print(np.extract(X[errors_indexes][0] >= 0, X[errors_indexes][0]))


def print_stats(y, preds):
    rel_mean_err = mean_absolute_percentage_error(y, preds)
    abs_err = get_mean_absolute_error(y, preds)
    max_err = get_max_absolute_error(y, preds)
    # max_err = 0
    mode_err = get_error_mode(y, preds)
    zero_errors = get_0_error_accuracy(y, preds)
    one_errors = get_1_error_accuracy(y, preds)
    two_errors = get_2_error_accuracy(y, preds)
    more_than_2_errors = more_than_two_errors_accuracy(y, preds)
    print(
        "MAE: {}, MRE: {}, Max {}, mode {} out of {}, e=0:{}, e==1:{}, e==2:{}, e>2:{}".format(abs_err, rel_mean_err,
                                                                                                max_err,
                                                                                                mode_err,
                                                                                                preds.shape[0],
                                                                                                zero_errors, one_errors,
                                                                                                two_errors,
                                                                                                more_than_2_errors))
    return (zero_errors, one_errors, two_errors, more_than_2_errors)


def get_errors_portions(y, preds):
    zero_errors = get_0_error_accuracy(y, preds)
    one_errors = get_1_error_accuracy(y, preds)
    two_errors = get_2_error_accuracy(y, preds)
    more_than_2_errors = more_than_two_errors_accuracy(y, preds)
    return (zero_errors, one_errors, two_errors, more_than_2_errors)


def count_by_smoothing_and_contraction(seq, console_log=False):
    if console_log:
        print("smooting + contraction")
    if not np.any(seq == 1):
        return 0
    sequence = np.copy(seq)
    ones_mode = get_stats_for_k(sequence, 1)["mode"]
    if ones_mode == 1:
        ones_mode = 2
    smoothed_sequence = smoothing(sequence, int(ones_mode / 2))
    return count_by_contraction(smoothed_sequence)


def count_predicted_reps(sequence, smooth=False, console_log=False):
    if console_log:
        if smooth:
            print("Two stages run through countings")
        else:
            print("Two stages run through countings with smoothing")
    seq = np.copy(sequence)
    certain_rep = 2
    if not 1 in seq:
        return 0
    slim = []
    consecutive_ones = 0
    isrep = False
    mode_1 = get_stats_for_k(seq, 1)["mode"]

    # Test Remove 1-2 zeros between ones
    for i in range(0, len(seq) - 3):
        if seq[i] == 1 and seq[i + 2] == 1:
            seq[i + 1] = 1

    for i in range(0, len(seq) - 4):
        if seq[i] == 1 and seq[i + 3] == 1:
            seq[i + 1] = 1
            seq[i + 2] = 1

    for i in range(0, len(seq) - 4):
        if seq[i] == 1 and seq[i + 4] == 1:
            seq[i + 1] = 1
            seq[i + 2] = 1
            seq[i + 3] = 1

    ## count really likely rep
    for i in range(0, len(seq)):
        if seq[i] == 0:
            isrep = False
            consecutive_ones = 0
        if seq[i] == 1:
            if isrep:
                seq[i] = certain_rep
                continue
            consecutive_ones += 1
            if consecutive_ones >= int(mode_1 * 1 / 2):
                for j in range(0, consecutive_ones):
                    seq[i - j] = certain_rep
                isrep = True
                slim.append(1)
    intrarep_mode = get_stats_for_k(seq, 0)["mode"]

    consecutive_ones = 0
    for i in range(0, len(seq)):
        if seq[i] == 0:
            consecutive_ones = 0
        if seq[i] == 1:
            if i == (len(seq) - 1) or seq[i + 1] == 0:
                left = get_left_side_zeros(seq, i)
                right = get_right_side_zeros(seq, i)
                if (intrarep_mode - left > intrarep_mode / 2) or (
                        (len(seq) - i) > intrarep_mode / 2 and intrarep_mode - right > intrarep_mode / 2):
                    seq[i] = 0
                    if i > 0:
                        seq[i - 1] = 0
                    continue
                else:
                    # if consecutive_ones >= mode_1 / 3:
                    slim.append(1)
            consecutive_ones += 1
    return np.sum(slim)


def count_pred_reps_with_smoothing(seq, smoothing_factor=0.4):
    # print_sequence_array_tight(seq)
    ones_mode = get_stats_for_k(seq, 1)["mode"]
    seq = smoothing(seq, ones_mode, smoothing_factor)
    return count_predicted_reps(seq, smooth=True)


def count_real_reps(truth):
    slim = []
    for t in truth:
        if len(slim) == 0 or slim[-1] != t:
            slim.append(t)
    return np.sum(slim)


if __name__ == "__main__":
    a = 1
    # test()
    # train_rep_counting_algorithm(with_smoothing=False)
    # print_unconstrained_workout_results(with_smoothing=True)

