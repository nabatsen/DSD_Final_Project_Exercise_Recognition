import datetime
import json
import logging as log
import pickle
import random
import time

# matplotlib.use("Agg")
import numpy as np
import yaml
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score

from constants import RANDOMNESS_SEED


def yaml_loader(filepath):
    with open(filepath, 'r') as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


def yaml_dump(filepath, data):
    with open(filepath, 'w') as file_descriptor:
        yaml.dump(data, file_descriptor)


def get_performance_values(y_truth, y_predict):
    return {"accuracy": accuracy_score(y_truth, y_predict, normalize=True),
            "recall": recall_score(y_truth, y_predict, average='micro'),
            "precison": precision_score(y_truth, y_predict, average='micro'),
            "f1": 0}


def get_random_params(params_grid):
    random_params = {}
    for param in params_grid.keys():
        random_params[param] = random.choice (params_grid[param])
    return random_params

def log_metrics_and_params(results, model_savepath):
    # log results and save path
    to_write = {}
    to_write['results'] = results
    to_write['model_savepath'] = model_savepath
    log.info('%s', json.dumps(to_write))


def save_model(clf):
    # save model with timestamp
    timestring = "".join(str(time.time()).split("."))
    model_savepath = 'model_' + timestring + '.pk'
    with open(model_savepath, 'wb') as ofile:
        pickle.dump(clf, ofile)
    return model_savepath


def get_train_metrics():
    # currently impossible
    # X_train and y_train are in higher scopes
    pass


def get_val_metrics(y_pred, y_true):
    return get_metrics(y_pred, y_true)


def get_metrics(y_pred, y_true):
    # compute more than just one metrics

    chosen_metrics = {
        # 'conf_mat': metrics.confusion_matrix,
        'accuracy': metrics.accuracy_score,
        'auc': metrics.roc_auc_score,
    }

    results = {}
    for metric_name, metric_func in chosen_metrics.items():
        try:
            inter_res = metric_func(y_pred, y_true)
        except Exception as ex:
            inter_res = None
            log.error("Couldn't evaluate %s because of %s", metric_name, ex)
        results[metric_name] = inter_res

    # results['conf_mat'] = results['conf_mat'].tolist()

    return results


def _my_scorer(clf, X_val, y_true_val):
    # do all the work and return some of the metrics

    y_pred_val = clf.predict(X_val)

    results = get_val_metrics(y_pred_val, y_true_val)
    timestring = "".join(str(time.time()).split("."))
    model_savepath = 'model_' + timestring + '.pk'
    log_metrics_and_params(results, model_savepath)
    return results['accuracy']


def plot_learning_curves(history, file_name):
    # summarize history for accuracy
    now = datetime.datetime.now()
    start_time = now.strftime("%Y-%m-%d %H:%M")
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    path = "reports/learning_curves_" + file_name + str(start_time) + ".png"
    plt.savefig(path)
    return path


def exercise_time_portion(times_list):
    total_seconds = sum_times(times_list)
    for ex_time in times_list:
        ex_time_seconds = float(int(ex_time[0:2]) * 60 * 60 + int(ex_time[2:4]) * 60 + int(ex_time[4:6]))
        print(ex_time_seconds / total_seconds)


def sum_times(time_lists):
    tot = 0
    for t in time_lists:
        tot += float(int(t[0:2]) * 60 * 60 + int(t[2:4]) * 60 + int(t[4:6]))
    return tot


def print_empty_lines(num_lines):
    for i in range(num_lines):
        print(" ")


def print_asterisks_lines(num_lines):
    for i in range(num_lines):
        print("*************************************")


def print_sequence_array_tight(seq):
    seq = seq.squeeze()
    np.set_printoptions(threshold='nan')
    seq = seq.astype(np.int32)
    stri = ""
    for digit in seq:
        stri += str(digit)
    print(stri)


def seconds_to_time_string(seconds):
    return str(int(seconds))
    # if seconds < 60:
    # else:
    #     min = int(seconds) / 60
    #     seconds = int(seconds) % 60
    #     return str(min) + "m" + str(seconds) + "s"

def seconds_to_time_string_for_array(seconds_arr):
    strs = []
    for seconds in seconds_arr:
        strs.append(seconds_to_time_string(seconds))
    return strs