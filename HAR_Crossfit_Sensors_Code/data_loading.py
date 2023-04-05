import os
import re
import sqlite3

from keras.engine.saving import load_model
from scipy.stats import mode

from constants import numpy_exercises_data_path, EXERCISE_NAME_TO_CLASS_LABEL, numpy_reps_data_path, unconstrained_workout_data_path, \
    READINGS_TABLE_NAME, EXERCISE_CLASS_LABEL_TO_NAME, FREE_WORKOUT_10_REPS, EXERCISES_TABLE_NAME, \
    EXERCISE_ID, WRIST_ACCEL_X, WRIST_GYRO_X, WRIST_ROT_X, ANKLE_ACCEL_X, ANKLE_GYRO_X, \
    WRIST_ROT_Y, WRIST_GYRO_Y, WRIST_ACCEL_Y, WRIST_ACCEL_Z, WRIST_GYRO_Z, WRIST_ROT_Z, ANKLE_ACCEL_Z, ANKLE_GYRO_Z, \
    ANKLE_ROT_Z, ANKLE_GYRO_Y, ANKLE_ACCEL_Y, ANKLE_ROT_X
from preprocessing import interpolate_readings
from data_features_extraction import extract_features_for_single_reading, extract_features
from utils import *




def get_group_for_id(id, config):
    part_to_ex = config.get("participant_to_ex_code_map_anonymous")
    i = 0
    for name in list(part_to_ex.keys()):
        if int(id) in part_to_ex[name]:
            return i, name
        i += 1
    return None, None


def get_exercise_ids_groups_and_persons(config, with_null_class=False):
    ex_folders = os.listdir(numpy_exercises_data_path)
    groups = []
    persons = []
    exercise_ids = []
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        if not with_null_class:
            if ex_folder == "Null":
                continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + ex_folder)
        for exercise_file in exericse_readings_list:
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))
            current_group, person = get_group_for_id(exercise_code, config)
            groups.append(current_group)
            persons.append(person)
            exercise_ids.append(exercise_code)
    groups = np.asarray(groups)
    persons = np.asarray(persons, dtype=object)
    exercise_ids = np.asarray(exercise_ids)
    return exercise_ids, groups, persons


def get_grouped_windows_for_exerices(with_feature_extraction,
                                     config=None, window_length=None,
                                     with_null_class=True,
                                     ids=None,
                                     window_step=0.05,
                                     smart_watches=None):
    ex_folders = os.listdir(numpy_exercises_data_path)
    if window_length is None:
        window_length = config.get("data_params")["window_length"]
    groups = []
    windows = []
    labels = []
    persons = []
    exercise_ids = []
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        if not with_null_class:
            if ex_folder == "Null":
                continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + ex_folder)
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex_folder]
        for exercise_file in exericse_readings_list:
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))
            if ids is not None and exercise_code not in ids:
                continue
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            windows_for_exercise = extract_windows(exercise, window_length, step=window_step)
            windows += windows_for_exercise
            current_group, person = get_group_for_id(exercise_code, config)
            for i in range(0, len(windows_for_exercise)):
                labels.append(label)
                groups.append(current_group)
                persons.append(person)
                exercise_ids.append(exercise_code)

    X = np.asarray(windows)
    print(X.shape)
    Y = np.asarray(labels)
    groups = np.asarray(groups)
    persons = np.asarray(persons, dtype=object)
    exercise_ids = np.asarray(exercise_ids)
    if smart_watches is not None:
        sensor_mask = np.ones(18).astype(np.bool)
        if "wrist" not in smart_watches:
            sensor_mask[0:9] = False
        elif "foot" not in smart_watches:
            sensor_mask[9:] = False
        X = X[:, sensor_mask, :]
    print(X.shape)
    if with_feature_extraction:
        X_features = extract_features(X)
        X_features = np.nan_to_num(X_features)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.nan_to_num(X)
        return [np.transpose(X, (0, 2, 1, 3)), X_features, Y, groups]
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.nan_to_num(X)
        return [np.transpose(X, (0, 2, 1, 3)), Y, groups, persons, exercise_ids]


def get_rep_count_for_exericse_id(ex_id):
    reps_ex_names = os.listdir(numpy_reps_data_path)
    reps = 0
    for ex in reps_ex_names:
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps:
            if ex_id in r_name:
                reps += 1
    return reps


def get_exercise_readings():
    ex_folders = os.listdir(numpy_exercises_data_path)
    exercises = []
    labels = []
    rep_counts = []
    a = True
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex_folder]
        for exercise_file in exericse_readings_list:
            if a:
                print(exercise_file)
                a = False
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            exercises.append(exercise[1:, :])
            exercise_id = (re.sub("[^0-9]", "", exercise_file))
            rep_count = get_rep_count_for_exericse_id(exercise_id)
            labels.append(label)
            rep_counts.append(rep_count)
    Y = np.asarray(labels)
    return exercises, Y, rep_counts


def extract_windows(exercise_reading, window_length_in_ms, step=0.20):
    windows = []
    for i in range(0, exercise_reading.shape[1] - int(window_length_in_ms / 10), int(window_length_in_ms / 10 * step)):
        windows.append(exercise_reading[:, i:i + int(window_length_in_ms / 10)])
    return windows



def get_reps_duration_map():
    reps_folders = os.listdir(numpy_reps_data_path)
    ex_code_to_rep_count_map = {}
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        for rep_readings_file_name in reps_readings_list:
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", rep_readings_file_name)
            rep_ex_id = int(rep_ex_id_plus_rep_num[0:3])
            if rep_ex_id not in ex_code_to_rep_count_map.keys():
                rep = np.load(numpy_reps_data_path + rep_folder + "/" + rep_readings_file_name)
                length = rep.shape[1]
                if length < 100:
                    continue
                # print(length)
                len_rounded = int(50 * round(float(length) / 50))
                ex_code_to_rep_count_map[rep_ex_id] = len_rounded
                if rep_ex_id == 645 or rep_ex_id == 434:
                    ex_code_to_rep_count_map[rep_ex_id] = 250
    return ex_code_to_rep_count_map


def get_min_rep_duration_map(augmentation=False):
    reps_folders = os.listdir(numpy_reps_data_path)
    seen_ex_ids = []
    ex_code_to_rep_count_map = {}
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        for rep_readings_file_name in reps_readings_list:
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", rep_readings_file_name)
            rep_ex_id = int(rep_ex_id_plus_rep_num[0:3])
            if rep_ex_id not in seen_ex_ids:
                seen_ex_ids.append(rep_ex_id)
                rep = np.load(numpy_reps_data_path + rep_folder + "/" + rep_readings_file_name)
                length = rep.shape[1]
                len_rounded = int(50 * round(float(length) / 50))
                if len_rounded <= 100:
                    continue
                if rep_folder not in ex_code_to_rep_count_map.keys() or ex_code_to_rep_count_map[
                    rep_folder] > len_rounded:
                    if augmentation:
                        len_rounded = len_rounded / 2
                    ex_code_to_rep_count_map[rep_folder] = len_rounded
    print(ex_code_to_rep_count_map)
    return ex_code_to_rep_count_map


def does_window_contain_rep_transition(window_endtime, rep_duration, transition_duration, window_length):
    if window_endtime % rep_duration <= window_length - transition_duration / 2 and window_endtime % rep_duration >= transition_duration / 2:
        return True
    return False


def does_window_contain_rep_start(window_endtime, rep_duration, rep_start_duration, window_length):
    if window_length >= window_endtime % rep_duration >= rep_start_duration:
        return True
    return False


def get_exercise_labeled_start_windows(exercise_reading, window_length_in_ms, exercise_code, transition_duration,
                                       ex_code_class=None, with_ex_code_as_feature=False, slide_step_perc=0.10):
    reps_duration_map = get_reps_duration_map()
    rep_duration = reps_duration_map[exercise_code]
    windows = []
    labels = []
    for start in range(0, exercise_reading.shape[1] - int(window_length_in_ms / 10),
                       int(window_length_in_ms / 10 * slide_step_perc)):
        stop = start + int(window_length_in_ms / 10)
        if with_ex_code_as_feature:
            with_ex_code_feature = np.append(exercise_reading[:, start:stop], np.full((1, 100), ex_code_class), axis=0)
            windows.append(with_ex_code_feature)
        else:
            windows.append(exercise_reading[:, start:stop])
        if does_window_contain_rep_start(stop, rep_duration, transition_duration, window_length_in_ms / 10):
            labels.append(True)
        else:
            labels.append(False)
    return (windows, labels)


def calculate_longest_and_shortest_rep_per_exercise():
    reps_folders = os.listdir(numpy_reps_data_path)
    ex_code_to_rep_durations = [[], [], [], [], [], [], [], [], [], []]
    ex_code_seen = []
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        for rep_readings_file_name in reps_readings_list:
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", rep_readings_file_name)
            rep_ex_id = int(rep_ex_id_plus_rep_num[0:3])
            rep = np.load(numpy_reps_data_path + rep_folder + "/" + rep_readings_file_name)
            length = rep.shape[1]
            len_rounded = int(50 * round(float(length) / 50))
            if rep_ex_id in ex_code_seen or len_rounded < 150:
                continue
            else:
                ex_code_seen += [rep_ex_id]
            ex_code_to_rep_durations[EXERCISE_NAME_TO_CLASS_LABEL[rep_folder] - 1] += [len_rounded]
            # print(len_rounded)
            # print()
    for i in range(0, len(ex_code_to_rep_durations)):
        ex_code_to_rep_durations[i] = (
            min(ex_code_to_rep_durations[i]), mode(ex_code_to_rep_durations[i])[0][0], max(ex_code_to_rep_durations[i]))
    return ex_code_to_rep_durations





def get_grouped_windows_for_rep_transistion_per_exercise(training_params, config=None, use_exercise_code_as_group=False,
                                                         exercises=None):
    ex_folders = os.listdir(numpy_exercises_data_path)

    tot = {}
    for ex_folder in ex_folders:
        if ex_folder == "Null":
            continue
        if exercises is not None and ex_folder not in exercises:
            continue
        groups = []
        windows = []
        transition_labels = []
        classes = []
        window_length_in_ms = int(training_params[ex_folder].window_length * 0.90) * 10
        rep_start_duration = int(window_length_in_ms * training_params[ex_folder].rep_start_portion / 10)
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in exericse_readings_list:
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))

            (windows_for_exercise, transition_labels_for_ex) = get_exercise_labeled_start_windows(exercise,
                                                                                                  window_length_in_ms,
                                                                                                  exercise_code,
                                                                                                  rep_start_duration,
                                                                                                  EXERCISE_NAME_TO_CLASS_LABEL[
                                                                                                      ex_folder],
                                                                                                  slide_step_perc=
                                                                                                  training_params[
                                                                                                      ex_folder].window_step_slide,
                                                                                                  with_ex_code_as_feature=False)

            windows += windows_for_exercise
            transition_labels += transition_labels_for_ex
            if use_exercise_code_as_group:
                current_group = exercise_code
            else:
                current_group = get_group_for_id(exercise_code, config)
            for i in range(0, len(windows_for_exercise)):
                groups.append(current_group)
                classes.append(EXERCISE_NAME_TO_CLASS_LABEL[ex_folder])

        X = np.asarray(windows)
        Y = np.asarray(transition_labels)
        classes = np.asarray(classes).reshape((len(classes), 1))
        groups = np.asarray(groups)

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.nan_to_num(X)
        tot[ex_folder] = [np.transpose(X, (0, 2, 1, 3)), classes, Y, groups]
    return tot


def extract_test_data(wrist_file, ankle_file, ex_code=FREE_WORKOUT_10_REPS, window=5000, step=0.20):
    db_wrist = sqlite3.connect(wrist_file)
    db_ankle = sqlite3.connect(ankle_file)
    cursor_w = db_wrist.cursor()
    cursor_w.execute(
        'SELECT * FROM {tn} WHERE exercise_code={code}'.format(tn=EXERCISES_TABLE_NAME, code=ex_code))
    exercise_id_w = np.array(cursor_w.fetchall())[0][EXERCISE_ID]
    cursor_w.execute(
        'SELECT * FROM {tn} WHERE exercise_id={id}'.format(tn=READINGS_TABLE_NAME, id=exercise_id_w))
    cursor_a = db_ankle.cursor()
    cursor_a.execute(
        'SELECT * FROM {tn} WHERE exercise_code={code}'.format(tn=EXERCISES_TABLE_NAME, code=ex_code))
    exercise_id_a = np.array(cursor_a.fetchall())[0][EXERCISE_ID]
    cursor_a.execute(
        'SELECT * FROM {tn} WHERE exercise_id={code}'.format(tn=READINGS_TABLE_NAME, code=exercise_id_a))
    readings_w = np.array(cursor_w.fetchall())
    readings_a = np.array(cursor_a.fetchall())
    interpolated = interpolate_readings(readings_w, readings_a)
    windows = np.asarray(extract_windows(interpolated, window, step=step))
    windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
    windows = np.nan_to_num(windows)
    windows = np.transpose(windows, (0, 2, 1, 3))
    return windows


def extract_test_rep_data(wrist_file, ankle_file, recognized_exercises, ex_code=FREE_WORKOUT_10_REPS, model_params=None,
                          step=0.05):
    db_wrist = sqlite3.connect(wrist_file)
    db_ankle = sqlite3.connect(ankle_file)
    cursor_w = db_wrist.cursor()
    cursor_w.execute(
        'SELECT * FROM {tn} WHERE exercise_code={code}'.format(tn=EXERCISES_TABLE_NAME, code=ex_code))
    exercise_id_w = np.array(cursor_w.fetchall())[0][EXERCISE_ID]
    cursor_w.execute(
        'SELECT * FROM {tn} WHERE exercise_id={id}'.format(tn=READINGS_TABLE_NAME, id=exercise_id_w))
    cursor_a = db_ankle.cursor()
    cursor_a.execute(
        'SELECT * FROM {tn} WHERE exercise_code={code}'.format(tn=EXERCISES_TABLE_NAME, code=ex_code))
    exercise_id_a = np.array(cursor_a.fetchall())[0][EXERCISE_ID]
    cursor_a.execute(
        'SELECT * FROM {tn} WHERE exercise_id={code}'.format(tn=READINGS_TABLE_NAME, code=exercise_id_a))
    readings_w = np.array(cursor_w.fetchall())
    readings_a = np.array(cursor_a.fetchall())
    interpolated = interpolate_readings(readings_w, readings_a)
    for rec_ex in recognized_exercises:
        window_length_in_ms = int(model_params[EXERCISE_CLASS_LABEL_TO_NAME[rec_ex.ex_code]].window_length * 0.90) * 10
        cut_out_data = interpolated[:,
                       max(0, rec_ex.start_time - int(window_length_in_ms / 50)):rec_ex.end_time]
        windows = np.asarray(extract_windows(cut_out_data, window_length_in_ms, step=step))
        windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
        windows = np.nan_to_num(windows)
        windows = np.transpose(windows, (0, 2, 1, 3))
        if model_params[EXERCISE_CLASS_LABEL_TO_NAME[rec_ex.ex_code]].sensors_as_channels:
            windows = windows[:, :, [WRIST_ACCEL_X, WRIST_GYRO_X, WRIST_ROT_X, ANKLE_ACCEL_X, ANKLE_GYRO_X, ANKLE_ROT_X,
                                     WRIST_ACCEL_Y, WRIST_GYRO_Y, WRIST_ROT_Y, ANKLE_ACCEL_Y, ANKLE_GYRO_Y, ANKLE_ROT_Z,
                                     WRIST_ACCEL_Z, WRIST_GYRO_Z, WRIST_ROT_Z, ANKLE_ACCEL_Z, ANKLE_GYRO_Z,
                                     ANKLE_ROT_Z], :]
            windows = np.reshape(windows, (windows.shape[0], windows.shape[1], 3, 6))

        rec_ex.set_windows(windows)
    return recognized_exercises


def load_rep_counting_models():
    rep_counting_models = {}
    for key, value in EXERCISE_NAME_TO_CLASS_LABEL.iteritems():
        if key == "Null":
            continue
        rep_counting_models[value] = load_model("models/best_rep_counting_model_" + key + ".h5")
    return rep_counting_models
