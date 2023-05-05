import collections
import json
import os
import shutil
import sqlite3

# from matplotlib import pyplot as plt
# from matplotlib2tikz import save as tikz_save

from constants import path, READING_VALUES, READING_SENSOR_TYPE, ACCELEROMETER_CODE, GYROSCOPE_CODE, ORIENTATION_CODE, \
    READING_TIMESTAMP, numpy_exercises_data_path, WORKOUT, READING_REP, READING_ID, numpy_reps_data_path, \
    SENSOR_TO_VAR_COUNT, SENSOR_TO_NAME, SMARTWATCH_POSITIONS, DELAYS, NULL_CLASS, path_plots
from db_functions import *

TABLES = [WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME]
def anonymize_dbs():
    db = sqlite3.connect(path + "merged_ankle copy")
    c = db.cursor()
    participants = np.array(c.execute(
        "SELECT * FROM {tn}".format(tn=WORKOUTS_TABLE_NAME)).fetchall())
    num = 1
    for p in participants:
        print(str(p[READING_ID]) + " " + str(p[1]) +  " " + str(num))
        c.execute(
            "UPDATE {tn} SET [participant]='{anonymous_name}' WHERE id={id} ".format(tn=WORKOUTS_TABLE_NAME,
                                                                                 id=p[READING_ID],
                                                                                      anonymous_name="P"+str(num)))
        num+=1

    db.commit()
    c.close()


def addRepSeparators(plot, repsDelimiters, timestamps):
    for i in range(0, timestamps.shape[0]):
        if repsDelimiters[i] == 1:
            plot.axvline(x=timestamps[i])


def check_exercises_for_participant(db, participant, position):
    ex_codes = get_exercise_codes_for_participants(db, participant)
    for code in WORKOUT:
        if code not in ex_codes:
            print((participant + " did not perform " + EXERCISE_CODES_TO_NAME[code] + " on " + position))
    check_readings_for_participant(db, participant, position)
    # check that readings are there


def check_readings_for_participant(db, participant, position):
    ex_ids = get_exercises_ids_for_participants(db, participant)
    for id in ex_ids:
        readings = get_readings_for_exercise(db, id)
        if readings.size == 0:
            print((participant + " " + position + "  readings are missing for exercise " + get_exercise_name_for_id(db,
                                                                                                                   id)))
        else:
            check_all_sensors_were_recorded(readings, participant, position, id[0])


def check_all_sensors_were_recorded(readings, participant, position, ex_id):
    sensor_codes = readings[:, READING_SENSOR_TYPE]
    occurences = collections.Counter(sensor_codes)
    if occurences[str(ACCELEROMETER_CODE)] == 0:
        print((participant + " " + position + " acc numpy_data_01 is missing for ex_id " + str(ex_id)))
    if occurences[str(GYROSCOPE_CODE)] == 0:
        print((participant + " " + position + " gyro numpy_data_01 is missing for ex_id " + str(ex_id)))
    if occurences[str(ORIENTATION_CODE)] == 0:
        print((participant + " " + position + " rot numpy_data_01 is missing for ex_id " + str(ex_id)))


def were_all_sensors_recorded(readings):
    sensor_codes = readings[:, READING_SENSOR_TYPE]
    occurences = collections.Counter(sensor_codes)
    if occurences[str(ACCELEROMETER_CODE)] == 0:
        return False
    if occurences[str(GYROSCOPE_CODE)] == 0:
        return False
    if occurences[str(ORIENTATION_CODE)] == 0:
        return False
    return True




def interpolate_data_and_generate_numpy_arrays():
    shutil.rmtree(numpy_exercises_data_path, ignore_errors=True)
    shutil.rmtree(numpy_reps_data_path, ignore_errors=True)
    db_wrist = sqlite3.connect(path + "/merged_wrist")
    db_ankle = sqlite3.connect(path + "/merged_ankle")
    partipant_to_exercises_codes_map = {}
    WORKOUT.append(NULL_CLASS)
    for code in WORKOUT:
        partipants = get_participants(db_wrist)
        for p in partipants:
            wrist_readings = get_participant_readings_for_exercise(db_wrist, p[0], code)
            if (wrist_readings is None or wrist_readings.size == 0):
                continue
            ankle_readings = get_participant_readings_for_exercise(db_ankle, p[0], code)
            if (ankle_readings is None or ankle_readings.size == 0):
                continue
            if (not were_all_sensors_recorded(ankle_readings) or not were_all_sensors_recorded(wrist_readings)):
                continue
            interpolated_exercise_readings = interpolate_readings(wrist_readings, ankle_readings)
            print(code)
            print(('shape: ' + str(interpolated_exercise_readings.shape)))
            ex_id = get_exercises_id_for_participant_and_code(db_wrist, p[0], code)
            print((str(code) + str(p) + " ex id " + str(ex_id)))
            save_exercise_npy(interpolated_exercise_readings, code, ex_id[0])
            single_reps_readings_wrist = get_sub_readings_from_readings_for_wrist(wrist_readings)
            single_reps_readings_ankle = derive_sub_readings_for_ankle_from_wrist(single_reps_readings_wrist,
                                                                                  ankle_readings)
            for i in range(0, min(len(single_reps_readings_wrist), len(single_reps_readings_ankle))):
                interpolated_rep_reading = interpolate_readings(single_reps_readings_wrist[i],
                                                                single_reps_readings_ankle[i])
                save_rep_npy(interpolated_rep_reading, code, ex_id[0], i)
            if p[0] not in list(partipant_to_exercises_codes_map.keys()):
                partipant_to_exercises_codes_map[p[0]] = [ex_id[0]]
            else:
                partipant_to_exercises_codes_map[p[0]].append(int(ex_id[0]))
    with open('participant_ex_code_map.txt', 'w+') as file:
        file.write(json.dumps(partipant_to_exercises_codes_map))  # use `json.loads` to do the reverse


def save_rep_npy(rep_readings, exercise_code, exercise_id, rep):
    exercise_name = EXERCISE_CODES_TO_NAME[exercise_code]
    if not os.path.exists(numpy_reps_data_path):
        os.makedirs(numpy_reps_data_path)
    if not os.path.exists(numpy_reps_data_path + (exercise_name)):
        os.makedirs(numpy_reps_data_path + (exercise_name))
    np.save(numpy_reps_data_path + (exercise_name) + "/" + exercise_name + "_" + str(exercise_id) + "_" + str(rep),
            rep_readings)


def save_exercise_npy(exercise_readings, exercise_code, exercise_id):
    exercise_name = EXERCISE_CODES_TO_NAME[exercise_code]
    if not os.path.exists(numpy_exercises_data_path):
        os.makedirs(numpy_exercises_data_path)
    if not os.path.exists(numpy_exercises_data_path + exercise_name):
        os.makedirs(numpy_exercises_data_path + exercise_name)
    np.save(numpy_exercises_data_path + exercise_name + "/" + exercise_name + "_" + str(exercise_id),
            exercise_readings)


def interpolate_readings(wrist_readings, ankle_readings):
    acc_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(ACCELEROMETER_CODE)]
    gyro_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(GYROSCOPE_CODE)]
    rot_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(ORIENTATION_CODE)]

    acc_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(ACCELEROMETER_CODE)]
    gyro_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(GYROSCOPE_CODE)]
    rot_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(ORIENTATION_CODE)]

    acc_readings_values_w = extract_sensor_readings_values(acc_readings_w[:, READING_VALUES])
    gyro_readings_values_w = extract_sensor_readings_values(gyro_readings_w[:, READING_VALUES])
    rot_readings_values_w = extract_sensor_readings_values(rot_readings_w[:, READING_VALUES])

    acc_readings_values_a = extract_sensor_readings_values(acc_readings_a[:, READING_VALUES])
    gyro_readings_values_a = extract_sensor_readings_values(gyro_readings_a[:, READING_VALUES])
    rot_readings_values_a = extract_sensor_readings_values(rot_readings_a[:, READING_VALUES])

    rot_readings_values_w = fix_orientation_jumps(rot_readings_values_w)
    rot_readings_values_a = fix_orientation_jumps(rot_readings_values_a)

    timestamps_wrist = extract_timestamps(wrist_readings)
    timestamps_ankle = extract_timestamps(ankle_readings)
    start_timestamp = 0

    end_timestamp = min(np.max(timestamps_wrist), np.max(timestamps_ankle))

    step = 10
    equaly_spaced_apart_timestamps = np.array(list(range(start_timestamp, end_timestamp + 1, step)))
    interpolated_readings = np.zeros((3 * 3 * 2 + 1, equaly_spaced_apart_timestamps.shape[0]))

    values_list = [acc_readings_values_w, gyro_readings_values_w, rot_readings_values_w, acc_readings_values_a,
                   gyro_readings_values_a, rot_readings_values_a]
    time_stamp_list = [extract_timestamps(acc_readings_w), extract_timestamps(gyro_readings_w), extract_timestamps(
        rot_readings_w), extract_timestamps(acc_readings_a), extract_timestamps(gyro_readings_a),
                       extract_timestamps(rot_readings_a)]
    current_indexs = 0
    for i in range(0, len(values_list)):
        original_timestamps = time_stamp_list[i]
        interpolated_x = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 0])
        interpolated_y = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 1])
        interpolated_z = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 2])
        interpolated_x = interpolated_x.reshape(interpolated_x.shape[0], 1)

        interpolated_y = interpolated_y.reshape(interpolated_y.shape[0], 1)
        interpolated_z = interpolated_z.reshape(interpolated_z.shape[0], 1)
        concatenate = np.concatenate(
            (interpolated_x.transpose(), interpolated_y.transpose(), interpolated_z.transpose()))
        interpolated_readings[1 + current_indexs * 3: 1 + current_indexs * 3 + 3, :] = concatenate
        current_indexs += 1


    return interpolated_readings[1:, :]


def extract_sensor_readings_values(readings):
    sensor_reading_data = np.zeros([np.shape(readings)[0], 3])
    i = 0
    for reading in readings:
        vals = np.array(reading.split(" ")[0:3])
        vals = vals.astype(np.float)
        sensor_reading_data[i] = vals
        i = i + 1
    return sensor_reading_data


def get_sub_readings_from_readings_for_wrist(readings):
    reps = np.unique(readings[:, READING_REP])
    sub_readings = []
    for rep in reps:
        # select all rows where the rep number equals r
        a = readings[readings[:, READING_REP] == rep]
        sub_readings.append(a)
    return sub_readings


def derive_sub_readings_for_ankle_from_wrist(wrist_rep_readings, ankle_readings):
    ankle_fist_timestamp = ankle_readings[0, READING_TIMESTAMP].astype(np.int64)
    last = 0
    reps = []
    for rep in wrist_rep_readings:
        timestamps = extract_timestamps(rep)
        start = last
        end = last + timestamps[timestamps.shape[0] - 1]
        filtered = ankle_readings[
            start <= ankle_readings[:, READING_TIMESTAMP].astype(np.int64) - ankle_fist_timestamp]
        filtered = filtered[filtered[:, READING_TIMESTAMP].astype(np.int64) - ankle_fist_timestamp <= end]
        if (filtered.size == 0):
            continue
        reps.append(filtered)
        last = end + 1
    return reps


# def plot_all_exercises_same_type(db, ex_code):
#     cursor_ankle = db.cursor()
#     cursor_ankle.execute('SELECT id FROM {tn} WHERE exercise_code = {ec} '.format(tn=EXERCISES_TABLE_NAME, ec=ex_code))
#     exs = np.array(cursor_ankle.fetchall())
#     plot = None
#     for e in exs:
#         plot = plot_exercise_from_db(e[0], ex_code)
#     plot.show()


# def plot_exercise_from_db(exId, reps_to_show=5):
#     plt.figure()
#
#     column_index = 0
#     y_labels = {ACCELEROMETER_CODE: "Acceleration [m/s^2]", GYROSCOPE_CODE: "Rotation rate [rad/s]",
#                 ORIENTATION_CODE: "Orientation"}
#     graph_name = {ACCELEROMETER_CODE: "Accelerometer", GYROSCOPE_CODE: "Gyroscope", ORIENTATION_CODE: "Orientation"}
#     fig, axs = plt.subplots(3, 2)
#     plt.suptitle("Push up readings")
#     for watch in ["wrist", "ankle"]:
#         row_index = 0
#         for st in [ACCELEROMETER_CODE, GYROSCOPE_CODE, ORIENTATION_CODE]:
#             if column_index == 0:
#                 axs[row_index, column_index].title.set_text(graph_name[st] + " wrist")
#             elif column_index == 1:
#                 axs[row_index, column_index].title.set_text(graph_name[st] + " ankle")
#
#             conn = sqlite3.connect(path + "merged_" + watch)
#             c = conn.cursor()
#             c.execute(
#                 'SELECT * FROM {tn} WHERE exercise_id={exid} AND sensor_type={st}'.format(tn=READINGS_TABLE_NAME,
#                                                                                           st=st,
#                                                                                           exid=exId))
#             table = np.array(c.fetchall())
#             if table.size == 0:
#                 return None
#
#             values = table[:, READING_VALUES]
#             # extract reps
#             reps = table[:, 6]
#             if reps_to_show is None:
#                 length = reps.shape[0]
#             rep_starts = np.zeros([reps.shape[0], 1])
#             for i in range(0, reps.shape[0] - 1):
#                 if reps[i] != reps[i + 1] or i == 0:
#                     rep_starts[i] = True
#                     if reps_to_show is not None and int(np.sum(rep_starts.squeeze())) == reps_to_show + 1:
#                         length = i
#
#             sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[st]])
#
#             i = 0
#             for reading in values:
#                 vals = np.array(reading.split(" "))[0:3]
#                 vals = vals.astype(np.float)
#                 sensorReadingData[i] = vals
#                 i = i + 1
#
#             timestamps = table[:, 4].astype("int64")
#             timestamps = timestamps - timestamps[0]
#             timestamps = timestamps[:length]
#
#             # plt.suptitle(EXERCISE_CODES_TO_NAME[exerciseCode] + " " + SENSOR_TO_NAME[st] + " " + str(exId), fontsize=13)
#             plt.sca(axs[row_index, column_index])
#             plt.xticks(np.arange(min(timestamps), max(timestamps) + 10000, 1000),
#                        np.arange(min(timestamps), max(timestamps) + 10000, 1000) / 1000)
#
#             if (row_index == 2):
#                 plt.xlabel("Time [s]")
#
#             addRepSeparators(axs[row_index, column_index], rep_starts, timestamps)
#             axs[row_index, column_index].plot(timestamps, sensorReadingData[:length, 0], 'r-')
#             axs[row_index, column_index].plot(timestamps, sensorReadingData[:length, 1], 'b-')
#             axs[row_index, column_index].plot(timestamps, sensorReadingData[:length, 2], 'g-')
#             row_index += 1
#         column_index += 1
#     plt.tight_layout()
#     plt.savefig(path_plots + "readings_raw.png")
#     # tikz_save(path_plots+"raw_readings" +  ".tex")
#     return plt


def fix_orientation_jumps(orientation_readings):
    for axis in range(0,orientation_readings.shape[1]):
        readings_for_axis = orientation_readings[:,axis]
        for point_i in range(1, readings_for_axis.shape[0] - 1):
            if abs(readings_for_axis[point_i - 1] - readings_for_axis[point_i]) > 0.5:
                readings_for_axis[ point_i] *= -1
    return orientation_readings



def extract_timestamps(readings_entries):
    timestamps = readings_entries[:, 4].astype("int64")
    timestamps = timestamps - timestamps[0]
    return timestamps


def extract_readings_floats(values, sensorType):
    sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensorType]])
    i = 0
    for reading in values:
        vals = np.array(reading.split(" "))[0:3]
        vals = vals.astype(np.float)
        sensorReadingData[i] = vals
        i = i + 1
    return sensorReadingData


def extract_reps_start_timestamps(readings_wrist):
    reps = readings_wrist[:, READING_REP]
    rep_starts = np.zeros([reps.shape[0], 1])
    for i in range(0, reps.shape[0] - 1):
        if reps[i] != reps[i + 1] or i == 0:
            rep_starts[i] = True
    return rep_starts


if __name__ == "__main__":  #
    interpolate_data_and_generate_numpy_arrays()


