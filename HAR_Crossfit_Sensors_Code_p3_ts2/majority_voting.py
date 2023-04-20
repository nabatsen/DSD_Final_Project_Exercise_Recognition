from sklearn.metrics import accuracy_score

from constants import EXERCISE_CLASS_LABEL_TO_MIN_DURATION_MS
from utils import np, get_performance_values


def get_exercise_length_ms(x, window, step_percentage):
    if len(x.shape) == 0:
        return window
    step = step_percentage * window
    return window * x.shape[0] - (x.shape[0] - 1) * (window - step)


def find_majority(k):
    freq_map = {}
    maximums = ('', 0)  # (occurring element, occurrences)
    for n in k:
        if n in freq_map:
            freq_map[n] += 1
        else:
            freq_map[n] = 1
        if freq_map[n] > maximums[1]:
            maximums = ([n], freq_map[n])
        elif freq_map[n] == maximums[1]:
            maximums[0].append(n)
    return maximums


def major_voting_at_time(y, time, window_length, step_percentage, with_ties=True):
    if time > get_exercise_length_ms(y, window_length, step_percentage * window_length):
        print("exercise is shorter than time")
    labels_at_time = []
    step = step_percentage * window_length
    for windows_i in range(0, y.shape[0]):
        if step * windows_i < time < step * windows_i + window_length:
            labels_at_time.append(y[windows_i])
    most_common = find_majority(labels_at_time)
    if with_ties:
        if len(most_common[0]) > 1:
            return -1
    return most_common[0][0]


def convert_to_major_voting_labels(y_preds, window_length, step_percentage, with_min_rep_duration_filtering=True,
                                   with_null_class=False, with_ties=True):
    y_mv = []
    step = step_percentage * window_length
    if 0 in np.unique(y_preds):
        y_preds = y_preds+1

    for time in range(int(step / 2), int(get_exercise_length_ms(y_preds, window_length, step_percentage)),
                      int(step)):
        y_mv.append(
            major_voting_at_time(np.asarray(y_preds), time, window_length, step_percentage, with_ties=with_ties))

    previous = y_mv[0]
    streak = 1
    if with_min_rep_duration_filtering:
        for j in range(1, len(y_mv)):
            if j == len(y_mv)-1 and previous != 11 and streak * step_percentage * window_length < \
                        EXERCISE_CLASS_LABEL_TO_MIN_DURATION_MS[previous] and with_null_class:
                y_mv[j - streak:j] = np.ones(streak) * 11
            elif y_mv[j] == previous:
                streak += 1
            else:
                if previous != 11 and streak * step_percentage * window_length < \
                        EXERCISE_CLASS_LABEL_TO_MIN_DURATION_MS[previous]:
                    if y_mv[j - streak - 1] == y_mv[j]:
                        y_mv[j - streak:j] = np.ones(streak) * y_mv[j]
                    elif with_null_class and ((j - streak - 1 >= 0 and y_mv[j - streak - 1] == 11) or y_mv[j] == 11):
                        y_mv[j - streak:j] = np.ones(streak) * 11
                previous = y_mv[j]
                streak = 1
    return np.asarray(y_mv)


def get_majority_voting_performance_values(y_truth, y_preds, exercises_ids, window_length, step_percentage,
                                           with_ties=False):
    y_truth_majority = []
    y_preds_majority = []
    z = 0

    for ex in np.unique(exercises_ids):

        indeces = np.argwhere(exercises_ids == ex)
        if len(indeces)<2:
            print("one window ex")
            continue
        y_mv_truth = (convert_to_major_voting_labels(np.squeeze(y_truth[indeces]), window_length, step_percentage,
                                                     with_ties=with_ties))
        y_mv_preds = (convert_to_major_voting_labels(np.squeeze(y_preds[indeces]), window_length, step_percentage,
                                                     with_ties=with_ties))
        y_truth_majority = np.concatenate((y_truth_majority, y_mv_truth))
        y_preds_majority = np.concatenate((y_preds_majority, y_mv_preds))
        z += 1

    return get_performance_values(y_truth_majority, y_preds_majority)


def convert_split_to_majority_voting(y_truth, y_preds, exercises_ids, window_length=4000, step_percentage=0.005):
    y_truth_majority = []
    y_preds_majority = []
    for ex in np.unique(exercises_ids):
        print(ex)
        indeces = np.argwhere(exercises_ids == ex)
        if len(indeces)==1:
            continue
        y_mv_truth = (convert_to_major_voting_labels(np.squeeze(y_truth[indeces]), window_length, step_percentage,
                                                     with_ties=False))
        y_mv_preds = (convert_to_major_voting_labels(np.squeeze(y_preds[indeces]), window_length, step_percentage,
                                                     with_ties=False))
        y_truth_majority = np.concatenate((y_truth_majority, y_mv_truth))
        y_preds_majority = np.concatenate((y_preds_majority, y_mv_preds))

    return (y_truth_majority, y_preds_majority)
