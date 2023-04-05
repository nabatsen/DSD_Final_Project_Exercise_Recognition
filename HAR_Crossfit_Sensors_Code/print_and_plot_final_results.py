import itertools
import os
import pickle
import sqlite3

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib2tikz import save as tikz_save

from constants import exercise_colors, path_plots, numpy_exercises_data_path, \
    ACCELEROMETER_CODE, ORIENTATION_CODE, READING_VALUES, READINGS_TABLE_NAME, SENSOR_TO_VAR_COUNT, path, \
    GYROSCOPE_CODE, \
    EXERCISE_CLASS_LABEL_TO_FOR_PLOTS, rep_counting_constrained_results_path, \
    constrained_workout_rep_counting_loo_results
from preprocessing import fix_orientation_jumps, addRepSeparators
from rep_counting import count_pred_reps_with_smoothing, count_predicted_reps, print_stats, get_mean_absolute_error, \
    mean_absolute_percentage_error
from utils import seconds_to_time_string_for_array, print_sequence_array_tight, print_asterisks_lines, yaml_loader

config = yaml_loader("./config_cnn.yaml")
window_length = config.get("data_params")["window_length"]

def plot_confusion_matrix(cm,
                          classes=None,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    if classes is None:
        classes = ["Push ups", "Pull ups", "Burpees", "KB Deadlifts", "Box Jumps", "Squats", "Situps",
                   "Wallballs", "Kb Thrusters", "KB Press"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.rcParams.update({'font.size': 8})
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tick_params(axis='both', labelsize=8)
    plt.ylabel('True label')
    plt.xlabel(path_plots + 'Predicted label')
    plt.savefig(path_plots + title + ".png")
    tikz_save(path_plots + title+ ".tex")
    plt.clf()


def plot_boxplot(data, labels=None, title=""):
    plt.clf()
    numOfClasses = 10
    classes_names = [EXERCISE_CLASS_LABEL_TO_FOR_PLOTS[l + 1] for l in labels]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    # ax1.set_xlabel('Exercise')
    ax1.set_ylabel('Test Accuracy', size=10)
    # plt.rcParams.update({'font.size': 12})

    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue']
    medians = list(range(numOfClasses))
    for i in range(numOfClasses):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = np.column_stack([boxX, boxY])
        boxPolygon = Polygon(boxCoords, facecolor='royalblue')
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numOfClasses + 0.5)
    top = 1.1
    bottom = 0.80
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(classes_names,
                        rotation=45, fontsize=12)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numOfClasses) + 1
    weights = ['bold', 'semibold']
    for tick, label in zip(range(numOfClasses), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], top - (top * 0.05), str(np.round(np.average(data[tick]), 3)),
                 horizontalalignment='center', size=10, weight=weights[k],
                 color='royalblue')

    fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman')
    tikz_save(path_plots + 'recognition_box_plots.tex')
    plt.savefig(path_plots + 'recognition_box_plots.png')
    plt.clf()


def plot_barchart():
    import numpy as np
    import matplotlib.pyplot as plt

    n_groups = 10

    zero_errors = (0.83, 0.84, 0.28, 1, 0.82, 0.78, 0.91, 0.88, 0.48, 0.94)
    less_than_two_errors = (0.94, 0.91, 0.85, 1, 0.89, 0.89, 0.95, 0.95, 0.74, 0.96)
    less_than_three_errors = (0.95, 0.91, 0.87, 1, 0.89, 0.89, 0.98, 0.95, 0.82, 0.98)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.20

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, zero_errors, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='0 errors')

    rects2 = ax.bar(index + bar_width, less_than_two_errors, bar_width,
                    alpha=opacity, color='r', error_kw=error_config,
                    label='<=1 errors')

    rects3 = ax.bar(index + 2 * bar_width, less_than_three_errors, bar_width,
                    alpha=opacity, color='g', error_kw=error_config,
                    label='<=2 errors')

    ax.set_xlabel('Exercise')
    ax.set_ylabel('Error rate')
    ax.set_title('Errors per exercise class')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Burpees', 'Squats', "Kb squats press", 'Crunches', 'Kb Presses', 'Box Jumps', 'Pull ups',
                        'Wall balls', 'Push ups', 'Dead lifts'))
    ax.legend()
    plt.hlines(y=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), xmin=-1, xmax=10, linestyles='dashed')
    plt.grid(axis='y', alpha=0.75)
    fig.tight_layout()
    tikz_save(path_plots + "rep_counting_barcharts"+".tex")
    plt.savefig(path_plots + "rep_counting_barcharts"+".png")
    plt.show()


def plot_stacked_barchart(labels=None, zero_errors=None, one_errors=None, two_errors=None,
                          more_than_two_errors=None):
    if labels is None:
        labels = ["Push ups", "Pull ups", "Burpees", "Kb deadlifts", "Box Jumps", "Squats", "Situps", "Wall balls",
                  "Kb presses", "Kb Thrusters"]
    if more_than_two_errors is None:
        more_than_two_errors = [10, 11.6, 4.3, 0, 8.51, 18.18, 0, 4.8 , 2.27, 10.26]
    if two_errors is None:
        two_errors = [4,0 ,2.1, 6.25, 0, 2.27, 0, 0, 4.54, 2.56]
    if one_errors is None:
        one_errors = [28, 23.3 , 8.5, 6.25, 12.77, 25.0, 2.38, 0, 38.64, 28.21]
    if zero_errors is None:
        zero_errors = [58,65.1,85.1, 87.5, 78.72, 54.54, 97.62, 95.2, 54.54, 58.97]
    N = len(labels)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars: can also be len(x) sequence

    zeros_bar = plt.bar(ind, zero_errors, width)
    ones_bar = plt.bar(ind, one_errors, width,
                       bottom=zero_errors)
    twos_bar = plt.bar(ind, two_errors, width,
                       bottom=np.array(one_errors) + np.array(zero_errors))
    more_bar = plt.bar(ind, more_than_two_errors, width,
                       bottom=np.array(one_errors) + np.array(zero_errors) + np.array(two_errors))
    plt.ylim(0, 110)
    plt.yticks(np.linspace(0, 100, 11))

    plt.ylabel('Errors Portion')
    plt.title('Errors by class')
    plt.xticks(ind, labels)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((zeros_bar[0], ones_bar[0], twos_bar[0], more_bar[0]), ('0 errors', '1 errros', '2 errors', '>2 errors'))
    tikz_save(path_plots + "rep_counting_barcharts"+".tex")
    plt.savefig(path_plots + "rep_counting_barcharts"+".png")
    plt.show()


def plot_pie_chart():
    labels = '0 errors', '1 error', '2 errors', '>2 errors'
    sizes = [0.73529, 0.17305, 0.02172, 0.06992]
    explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    tikz_save(path_plots + "rep_counting_pie"+".tex")
    plt.savefig(path_plots + "rep_counting_pie"+".png")
    plt.show()


def plot_over_lap_accuracies(x=None,
                             y=None):
    if y is None:
        y = [0.9964, 0.9967, 0.9968, 0.9968, 0.9989, 0.9989,
             0.9989, 0.9989]
        y = np.asarray(y) * 100
    if x is None:
        x = [0, 0.1, 0.25, 0.50, 0.8, 0.9, 0.95, 0.99]
    plt.plot(x, y, 'ro')
    plt.plot(x, y)
    plt.xlabel("Overlap")
    plt.ylabel("Accuracy")
    plt.title("Accuracies vs windows overlap")
    plt.xticks(x)
    plt.yticks(y)
    plt.tight_layout()
    plt.savefig(path_plots + 'overlap_test_accuracy.png')
    tikz_save(path_plots + 'overlap_test_accuracy.tex')
    plt.show()


def plot_window_length_accuracy(x=None,
                                y=None):
    if x is None:
        x = [1, 1.5, 2, 3, 4, 5, 7, 10]
    if y is None:
        y = [0.9776, 0.9952, 0.9937,  0.9981, 0.9994, 0.9998, 0.9999, 0.9982]
    plt.plot(x, y, 'ro')
    plt.plot(x, y)
    plt.title("5-fold crossvalidation test accuracies for various window lengths")
    plt.xlabel("Window length [s]")
    plt.ylabel("Accuracy")
    plt.xticks(x)
    plt.yticks([0.9776, 0.9952, 0.9937,  0.9980, 0.9990, 0.9999])
    plt.tight_layout()
    plt.savefig(path_plots + 'window_length_test_accuracy.png')
    tikz_save(path_plots + 'window_length_test_accuracy.tex')
    plt.show()


def plot_free_workout(preds, step_size=0.5, with_null_class=True, participant_name=""):
    labels = np.unique(preds)
    print(labels)
    separators = []
    for i in range(1, len(preds)):
        if preds[i - 1] != preds[i]:
            separators.append((i - 1) * step_size)
    separators = np.asarray(separators)

    x = np.arange(0, len(preds)) * step_size
    coef = 100000
    preds = preds + coef
    y = preds
    print(y)
    plt.plot(x, y, color="white")
    # for s in separators:
    #     plt.axvline(x=s, color='grey', linestyle='--')
    plt.gca().axes.get_yaxis().set_visible(False)

    ax = plt.gca()
    x_axis_ticks = np.asarray(range(0, int(len(preds) * step_size), 10000))
    plt.xticks(x_axis_ticks, seconds_to_time_string_for_array((x_axis_ticks / 1000)))
    for i in labels:
        if i == 0: continue
        if i == 11:
            ax.fill_between(x, 0, y, where=(y == i + coef), facecolor=exercise_colors[i])
        else:
            ax.fill_between(x, 0, y, where=(y == i + coef), facecolor=exercise_colors[i],
                            label=EXERCISE_CLASS_LABEL_TO_FOR_PLOTS[i])

        # plt.text(0.5 * (step_size * 5 + step_size * 15), 50, "test",
        #          horizontalalignment='center', fontsize=20)
    plt.ylim((-coef / 2, coef * 2))
    plt.xlabel("Time [s]")
    ax.legend(loc="upper center", ncol=6)
    plt.tight_layout()
    if (with_null_class):
        tikz_save(path_plots + "free_workoutout_with_null_" + participant_name + ".tex")
        plt.savefig(path_plots + "free_workoutout_with_null_" + participant_name + ".png")
    else:
        tikz_save(path_plots + "free_workoutout_" + participant_name + ".tex")
        plt.savefig(path_plots + "free_workoutout_" + participant_name + ".png")
    plt.show()


def print_free_workout1_stats_per_ex():
    push_ups = ([10, 10, 10, 10, 10], [1, 11, 3, 12, 5])
    pull_ups = ([10, 10, 10, 7, 10], [2, 9, 2, 6, 10])
    burpees = ([10, 10, 10, 10, 10], [12, 11, 10, 11, 9])
    dead_lifts = ([10, 10, 10, 10, 10], [10, 10, 10, 10, 13])
    box_jumps = ([10, 10, 10, 10], [11, 10, 8, 9])
    squats = ([10, 10, 10, 10], [12, 12, 11, 10])
    situps = ([10, 10, 10, 10, 10], [12, 10, 10, 10, 10])
    wallballs = ([10, 10, 10, 10, 10], [11, 5, 11, 3, 10])
    presses = ([10, 10, 10, 10, 10], [11, 9, 9, 11, 10])
    thrusters = ([10, 10, 10, 10, 10], [10, 11, 9, 8, 11])

    arr = [push_ups, pull_ups, burpees, dead_lifts, box_jumps, squats, situps, wallballs, presses, thrusters]
    i = 1
    for ex in arr:
        print i
        print "MAE"
        print get_mean_absolute_error(np.array(ex[0]), np.array(ex[1]))
        print "MRE"
        print mean_absolute_percentage_error(np.array(ex[0]), np.array(ex[1]))
        i += 1


def print_free_workout1_stats_per_participant():
    P1 = ([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [10, 1, 11, 9, 8, 11, 11, 10, 10, 10])
    P2 = ([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [10, 9, 10, 10, 10, 12, 10, 8, 9, 10])
    P3 = ([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [1, 1, 10, 10, 10, 9, 10, 9, 10, 8])
    P4 = ([10, 10, 10, 10, 10, 10, 10, 10], [9, 4, 10, 10, 10, 10, 10, 8])
    P5 = ([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [10, 1, 10, 10, 9, 10, 10, 10, 8, 10])

    arr = [P1, P2, P3, P4, P5]
    i = 1
    for ex in arr:
        print i


        print "MAE"
        print get_mean_absolute_error(np.array(ex[0]), np.array(ex[1]))
        # print "MRE"
        # print mean_absolute_percentage_error(np.array(ex[0]), np.array(ex[1]))
        i += 1


def box_plots():
    file = open("./constrained_workout_results/all_sensor_training", 'rb')
    result = pickle.load(file)
    file.close()
    values = []
    labels = []
    per_class_performance_values = result.get_performance_values_per_class(window_length)
    for key, value in per_class_performance_values.items():
        split_values = []
        for split in value:
            split_values.append(split["accuracy"])
        values.append(split_values)
        labels.append(key)
    plot_boxplot(values, labels)


def print_over_lap_results():
    files = os.listdir("constrained_workout_results/")

    for f in files:
        if "over_lap_grid_search" not in f:
            continue
        print(f)
        file = open("./constrained_workout_results/" + f, 'rb')
        result = pickle.load(file)
        file.close()
        print(result.get_grid_search_parameter_test_accuracy(with_majority_voting=True,
                                                             window_length=window_length))

def plot_various_sensor_results_as_bars():
    labels = ["All", "Hand", "Foot", "Hand acc.", "Hand gyro", "Hand orient.", "Hand acc + gyro"]
    N = len(labels)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars: can also be len(x) sequence

    bars = plt.bar(ind, [99.9, 96.6, 85.0, 97.7, 30.9, 14, 98.6], width)
    plt.ylim(0, 1.10)
    plt.yticks([99.9, 96.6, 85.0, 97.7, 30.9, 14, 98.6])

    plt.ylabel('Errors Portion')
    plt.title('Errors by class')
    plt.xticks(ind, labels)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((bars), ('0 errors', '1 errros', '2 errors', '>2 errors'))
    plt.show()

def plot_10_secs_of_each_exercises(ids, sensor_type=ACCELEROMETER_CODE):
    plt.figure()

    column_index = 0
    y_labels = {ACCELEROMETER_CODE: "Acceleration [m/s^2]", GYROSCOPE_CODE: "Rotation rate [rad/s]",
                ORIENTATION_CODE: "Orientation [unitless]"}
    y_label = y_labels[sensor_type]
    graph_name = {ACCELEROMETER_CODE: "Accelerometer", GYROSCOPE_CODE: "Gyroscope", ORIENTATION_CODE: "Orientation"}
    fig, axs = plt.subplots(5, 2)
    index = 0
    for ex_id in ids:
        column_index = index / 5
        row_index = index % 5
        conn = sqlite3.connect(path + "merged_" + "wrist")
        c = conn.cursor()
        c.execute(
            'SELECT * FROM {tn} WHERE exercise_id={exid} AND sensor_type={st}'.format(tn=READINGS_TABLE_NAME,
                                                                                      st=sensor_type,
                                                                                      exid=ex_id))
        table = np.array(c.fetchall())
        if table.size == 0:
            return None

        values = table[:, READING_VALUES]

        sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensor_type]])

        i = 0
        for reading in values:
            vals = np.array(reading.split(" "))[0:3]
            vals = vals.astype(np.float)
            sensorReadingData[i] = vals
            i = i + 1

        if sensor_type is ORIENTATION_CODE:
            sensorReadingData = fix_orientation_jumps(sensorReadingData)

        reps = table[:, 6]
        ten_seconds_in_data_points = 1000
        rep_starts = np.zeros([reps.shape[0], 1])
        for i in range(0, ten_seconds_in_data_points - 1):
            if reps[i] != reps[i + 1] or i == 0:
                rep_starts[i] = True

        timestamps = table[:, 4].astype("int64")
        timestamps = timestamps - timestamps[0]
        timestamps = timestamps[:ten_seconds_in_data_points]

        plt.sca(axs[row_index, column_index])
        plt.xticks(np.arange(min(timestamps), max(timestamps) + 10000, 1000),
                   np.arange(min(timestamps), max(timestamps) + 10000, 1000) / 1000)
        if (row_index == 4):
            plt.xlabel("Time [s]")
        if (column_index == 0) and row_index == 2:
            plt.ylabel(y_label)

        addRepSeparators(axs[row_index, column_index], rep_starts, timestamps)
        axs[row_index, column_index].set_title(EXERCISE_CLASS_LABEL_TO_FOR_PLOTS[index + 1])
        axs[row_index, column_index].plot(timestamps, sensorReadingData[:ten_seconds_in_data_points, 0], 'r-')
        axs[row_index, column_index].plot(timestamps, sensorReadingData[:ten_seconds_in_data_points, 1], 'b-')
        axs[row_index, column_index].plot(timestamps, sensorReadingData[:ten_seconds_in_data_points, 2], 'g-')
        index += 1
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.6)
    plt.savefig(path_plots + "readings_10_exercises" + str(sensor_type)+ ".png")
    tikz_save(path_plots + "readings_10_exercises" + str(sensor_type)+ ".tex")
    plt.show()
    return plt


def print_different_sensors_results():
    files = os.listdir("constrained_workout_results/")
    for f in files:
        if "training" not in f:
            continue
        print(f)
        file = open("./constrained_workout_results/" + f, 'rb')
        result = pickle.load(file)
        file.close()
        print(result.get_total_test_performance_values(with_majority_voting=True, window_length=window_length))


def print_different_different_window_length_results():
    files = os.listdir("./constrained_workout_results/")

    for f in files:
        if "xxxxx" not in f:
            continue
        print(f)
        file = open("./constrained_workout_results/" + f, 'rb')
        result = pickle.load(file)
        file.close()
        wl = int(f.split("_")[-3])
        print(result.get_total_test_performance_values(with_majority_voting=True, window_length=wl)['accuracy'])



def plot_all_sensors_confusion_matrix():
    file = open("./constrained_workout_results/all_sensor_training", 'rb')
    # result = pickle.load(file, encoding='latin1')#pyton 3
    result = pickle.load(file)
    file.close()
    conf_matrix = result.get_confusion_matrix()
    cm = np.copy(conf_matrix)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.savetxt(path_plots + "confusion_matrix_all.csv", cm, delimiter=",")
    plot_confusion_matrix(conf_matrix, normalize=True, title="Confusion matrix all readings")
    print(conf_matrix)


def plot_hand_confusion_matrix():
    file = open("./constrained_workout_results/hand_training", 'rb')
    result = pickle.load(file)
    file.close()
    conf_matrix = result.get_confusion_matrix()
    cm = np.copy(conf_matrix)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.savetxt(path_plots + "confusion_matrix_hand.csv", cm, delimiter=",")
    plot_confusion_matrix(conf_matrix, title="Confusion matrix hand readings")
    print(conf_matrix)


def plot_foot_confusion_matrix():
    file = open("./constrained_workout_results/foot_training", 'rb')
    result = pickle.load(file)
    file.close()
    conf_matrix = result.get_confusion_matrix()
    cm = np.copy(conf_matrix)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.savetxt(path_plots + "confusion_matrix_foot.csv", cm, delimiter=",")
    plot_confusion_matrix(conf_matrix, title="Confusion matrix foot readings")
    print(conf_matrix)


def print_constrained_workout_repcounting_truth_and_predicted_for_comparison():
    X = np.load("X_sequences.npy")
    X = np.pad(X, (4, 0), 'constant', constant_values=(1,))[4:, :]
    y = np.load("rep_count_per_sequence.npy")
    ex_ids = np.load("exercises_ids.npy")
    y_LOO = y[::2]
    for i in range(0, X.shape[0]-1,2):
        pred_rep_count = count_pred_reps_with_smoothing(X[i][X[i]>=0])
        actual_rep_count = y_LOO[i/2]
        if abs(pred_rep_count - actual_rep_count)>0:

            print("id: {}".format(ex_ids[i/2][0]))


            print("truth")
            print_sequence_array_tight(X[i+1][X[i+1]>=0])
            print("pred")
            print_sequence_array_tight(X[i][X[i]>=0])
            differences = (X[i+1][X[i+1]>=0] != X[i][X[i]>=0]).astype(np.int)
            # print_sequence_array_tight(differences)
            print(pred_rep_count)
            print(actual_rep_count)

def generate_all_plots():
    plot_10_secs_of_each_exercises([596, 597, 598, 599, 600, 601, 602, 603, 604, 605], ACCELEROMETER_CODE)
    plot_10_secs_of_each_exercises([596, 597, 598, 599, 600, 601, 602, 603, 604, 605], GYROSCOPE_CODE)
    plot_10_secs_of_each_exercises([596, 597, 598, 599, 600, 601, 602, 603, 604, 605], ROTATION_MOTION)
    box_plots()
    plot_window_length_accuracy()
    plot_all_sensors_confusion_matrix()
    plot_foot_confusion_matrix()
    plot_hand_confusion_matrix()
    plot_over_lap_accuracies()

def save_exercises_to_cv(ids):
    ex_folders = os.listdir(numpy_exercises_data_path)
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        if ex_folder == "Null":
            continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + ex_folder)
        for exercise_file in exericse_readings_list:
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))
            if exercise_code not in ids:
                continue
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            np.savetxt(path_plots + "ex_readings_"+ex_folder+".csv", exercise[:, :1000], delimiter=",")


def plot_free_workout_recognition_result(file, with_null_class=True):
    preds = np.load(file)
    window = window_length
    step = 0.05
    plot_free_workout((preds), window * step, with_null_class=with_null_class)


def print_rep_counting_constrained_workout_results():
    import os
    ex_folders = os.listdir(numpy_exercises_data_path)
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        if ex_folder == "Null":
            continue
        print ex_folder
        X = np.load(constrained_workout_rep_counting_loo_results + "X_sequences_" + ex_folder+ ".npy")
        X = np.pad(X, (4, 0), 'constant', constant_values=(1,))[4:, :]
        y = np.load(constrained_workout_rep_counting_loo_results + "rep_count_per_sequence_" + ex_folder+ ".npy")
        X_GT = X[1::2]
        X_LOO = X[::2]
        y_LOO = y[::2]
        labels_x_loo = []
        for i in range(0,y_LOO.shape[0]):
            labels_x_loo.append(ex_folder)

        print_asterisks_lines(2)
        print_asterisks_lines(2)
        preds = []
        for x in X_LOO:
            preds.append(count_pred_reps_with_smoothing(x[x >= 0]))
        preds = np.asarray(preds)
        print_stats(y_LOO, preds)
    return



if __name__ == "__main__":  #
    plot_window_length_accuracy()
    # print_different_different_window_length_results()
    # print_over_lap_results()
    # print_different_sensors_results()
    # plot_all_sensors_confusion_matrix()
    # print_rep_counting_constrained_workout_results()
    exit()