import os

import numpy as np

from constants import numpy_reps_data_path, numpy_exercises_data_path
from utils import yaml_loader


def print_workout_info2():
    print_number_of_reps_per_exercise()


def print_number_of_reps_per_exercise():
    print("********* REPS **********")
    reps_folders = os.listdir(numpy_reps_data_path)
    name_to_rep_count = {}
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        name_to_rep_count[rep_folder] = len(reps_readings_list)
    for key, value in sorted(iter(name_to_rep_count.items()), key=lambda k_v: (k_v[1], k_v[0])):
        print("%s: %s" % (key, value))


def print_number_of_exercise():
    print("********* Exercise Count **********")
    ex_folders = os.listdir(numpy_exercises_data_path)
    name_to_rep_count = {}
    for ex_folder in ex_folders:
        ex_reading_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        name_to_rep_count[ex_folder] = len(ex_reading_list)
    for key, value in sorted(iter(name_to_rep_count.items()), key=lambda k_v1: (k_v1[1], k_v1[0])):
        print("%s: %s" % (key, value))

def seconds_to_min_and_seconds(seconds):
    return str(int(seconds/60))+":"+str((seconds%60))


def print_recording_time_exercises():
    print("********* Exercise Time **********")
    ex_folders = os.listdir(numpy_exercises_data_path)
    ex_to_duration= {}
    for ex_folder in ex_folders:
        ex_reading_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in ex_reading_list:
            readings = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            exercise_duration= (readings.shape[1])/100
            if ex_folder in list(ex_to_duration.keys()):
                ex_to_duration[ex_folder]+=exercise_duration
            else:
                ex_to_duration[ex_folder] =exercise_duration
    for key, value in sorted(iter(ex_to_duration.items()), key=lambda k_v2: (k_v2[1], k_v2[0])):
        print("%s: %s" % (key, seconds_to_min_and_seconds(value)))



