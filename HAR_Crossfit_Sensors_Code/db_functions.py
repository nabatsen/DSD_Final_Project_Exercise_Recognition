import numpy as np

from constants import WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME, EXERCISE_CODES_TO_NAME


def get_exercises_ids_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                     id=workout_id[0, 0]))
    exercises_ids = cursor.fetchall()
    return np.array(exercises_ids)


def get_exercises_id_for_participant_and_code(db, participant, code):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id} AND exercise_code={ex_code}'.format(
            tn=EXERCISES_TABLE_NAME,
            id=workout_id[0, 0], ex_code=code))
    exercises_ids = cursor.fetchall()
    return np.array(exercises_ids)[0]


def did_participant_perform_exercise(db, participant, ex_code):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    if (workout_id.size == 0):
        return False

    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id} AND exercise_code={ex_code}'.format(
            tn=EXERCISES_TABLE_NAME,
            id=workout_id[0, 0], ex_code=ex_code))
    exercises_ids = cursor.fetchall()
    return len(exercises_ids) > 0


def get_exercises_ids_for_workout_id(workout_id, db):
    c = db.cursor()
    c.execute('SELECT id FROM {tn} WHERE workout_session_id={wid}'.format(tn=EXERCISES_TABLE_NAME, wid=workout_id))
    ids = np.array(c.fetchall())
    return ids


def get_participant_for_exercise_id(db, exercise_id):
    c = db.cursor()
    c.execute('SELECT workout_session_id FROM {tn} WHERE id={eid}'.format(tn=EXERCISES_TABLE_NAME, eid=exercise_id))
    workout_id = np.array(c.fetchall())[0]
    c.execute('SELECT participant FROM {tn} WHERE id={wid}'.format(tn=WORKOUTS_TABLE_NAME, wid=workout_id[0]))
    name = np.array(c.fetchall())
    return name[0, 0]


def get_exercise_codes_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    if workout_id.size == 0:
        return None
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                                id=workout_id[0, 0]))
    exercise_codes = cursor.fetchall()
    return np.array(exercise_codes)


def get_readings_for_exercise(db, id, sensor_type=None):
    cursor = db.cursor()
    if sensor_type is not None:
        cursor.execute(
            'SELECT * FROM {tn} WHERE exercise_id = {exid} AND sensor_type ={st}'.format(tn=READINGS_TABLE_NAME,
                                                                                         exid=id[0], st=sensor_type))
    else:
        cursor.execute(
            'SELECT * FROM {tn} WHERE exercise_id = {exid}'.format(tn=READINGS_TABLE_NAME,
                                                                   exid=id[0]))
    return np.array(cursor.fetchall())


def get_exercise_name_for_id(db, id):
    cursor = db.cursor()
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                id=id[0]))
    return EXERCISE_CODES_TO_NAME[np.array(cursor.fetchall())[0, 0]]


def get_participant_readings_for_exercise(db, participant, exercise_code):
    codes = get_exercise_codes_for_participants(db, participant)
    if codes is None:
        return None
    found = False
    for code in codes:
        if code[0] == exercise_code:
            found = True
    if not found:
        return None

    exericse_id = get_exercises_id_for_participant_and_code(db, participant, exercise_code)
    readings = get_readings_for_exercise(db, exericse_id)
    if readings.size == 0:
        return None
    return readings


def get_participants(db):
    b_cursor = db.cursor()
    b_cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(b_cursor.fetchall())
    return participants


def delete_reps_for_participant(db, participant, reps, exercise_code):
    id = get_exercises_id_for_participant_and_code(db, participant, exercise_code)
    c = db.cursor()
    for r in reps:
        c.execute("DELETE FROM {tn} WHERE id={id} AND rep_count={rep}".format(tn=READINGS_TABLE_NAME, id=id[0], rep = r))

    db.commit()
    c.close()


