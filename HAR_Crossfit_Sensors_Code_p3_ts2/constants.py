#
RANDOMNESS_SEED = 29102018

# CONSTANTS
PUSH_UPS = 1
KETTLE_BELL_SWINGS = 2
PULL_UPS = 3
BOX_JUMPS = 4
BURPEES = 5
SQUATS = 6
DEAD_LIFT = 7
KETTLEBELL_SQUAT_PRESS = 8
KETTLEBELL_PRESS = 9
CRUNCHES = 11
WALL_BALLS = 12
NULL_CLASS = 16
FREE_WORKOUT_123_SCHEME = 14
FREE_WORKOUT_10_REPS = 15

WORKOUT = [PUSH_UPS, PULL_UPS, BURPEES, DEAD_LIFT, BOX_JUMPS, SQUATS, CRUNCHES, WALL_BALLS,
           KETTLEBELL_PRESS, KETTLEBELL_SQUAT_PRESS]

DELAYS = {"P24": 1000, "Daniel": 1000, "Matteo": 1000, "Tobi bro": 1000, "Renata farkas": 1000,
          "Karol Wojtas": 1000, "Riccardo Rigamonti": 1000, "Damian ga": 4000,
          "Donato pari": 4500}

EXERCISE_CODES_TO_NAME = {
    PUSH_UPS: "Push ups",
    PULL_UPS: "Pull ups",
    BURPEES: "Burpees",
    SQUATS: "Squats",
    BOX_JUMPS: "Box jumps",
    KETTLE_BELL_SWINGS: "Kettle B swings",
    DEAD_LIFT: "Dead lifts",
    KETTLEBELL_SQUAT_PRESS: "KB Squat press",
    KETTLEBELL_PRESS: "KB Press",
    CRUNCHES: "Crunches",
    WALL_BALLS: "Wall balls",
    NULL_CLASS: "Null"
}

EXERCISE_NAME_TO_CLASS_LABEL = {
    "Push ups": 1,
    "Pull ups": 2,
    "Burpees": 3,
    "Dead lifts": 4,
    "Box jumps": 5,
    "Squats": 6,
    "Crunches": 7,
    "Wall balls": 8,
    "KB Squat press": 9,
    "KB Press": 10,
    "Null": 11,
}

EXERCISE_CLASS_LABEL_TO_NAME = {
    1: "Push ups",
    2: "Pull ups",
    3: "Burpees",
    4: "Dead lifts",
    5: "Box jumps",
    6: "Squats",
    7: "Crunches",
    8: "Wall balls",
    9: "KB Squat press",
    10: "KB Press",
    11: "Null"
}

EXERCISE_CLASS_LABEL_TO_FOR_PLOTS = {
    1: "Push-up",
    2: "Pull-up",
    3: "Burpee",
    4: "KB deadlift",
    5: "Box jump",
    6: "Air squat",
    7: "Sit-up",
    8: "Wall ball",
    9: "KB thruster",
    10: "KB press",
    11: "Null"
}

EXERCISE_CLASS_LABEL_TO_MIN_DURATION_MS = {
    1: 1200,
    2: 2000,
    3: 2500,
    4: 2000,
    5: 2000,
    6: 1500,
    7: 2000,
    8: 2000,
    9: 1500,
    10: 1500
}

CLASS_LABEL_TO_AVERAGE_REP_DURATION = [250, 300, 300, 200, 300, 200, 250, 250, 250, 300]
MIN_REP_DURATION_MAP = {'Burpees': 200, 'Push ups': 150, 'Squats': 150, 'KB Squat press': 200, 'Crunches': 200,
                        'KB Press': 150, 'Pull ups': 200, 'Wall balls': 200, 'Dead lifts': 150, 'Box jumps': 200}

ACCELEROMETER_CODE = 1
GYROSCOPE_CODE = 4
ORIENTATION_CODE = 11

SENSOR_POSITION_ANKLE = 0
SENSOR_POSITION_WRIST = 1

SENSOR_TO_VAR_COUNT = {
    ACCELEROMETER_CODE: 3,
    GYROSCOPE_CODE: 3,
    ORIENTATION_CODE: 3}

SENSOR_TO_NAME = {
    ACCELEROMETER_CODE: "Accel",
    GYROSCOPE_CODE: "Gyro",
    ORIENTATION_CODE: "Rot Motion"}

# TABLE COLUMNS
# READINGS
READING_ID = 0
READING_SENSOR_TYPE = 1
READING_VALUES = 2
READING_EXERCISE_ID = 3
READING_TIMESTAMP = 4
READING_PLACEMENT = 5
READING_REP = 6

EXERCISE_ID = 0

READINGS_TABLE_NAME = "sensor_readings"
EXERCISES_TABLE_NAME = "exercises"
WORKOUTS_TABLE_NAME = "workout_sessions"

POSITION_WRIST = "wrist"
POSITION_ANKLE = "ankle"
SMARTWATCH_POSITIONS = [POSITION_WRIST, POSITION_ANKLE]

# Numpy numpy_data_01
WRIST_ACCEL_X = 0
WRIST_ACCEL_Y = 1
WRIST_ACCEL_Z = 2
WRIST_GYRO_X = 3
WRIST_GYRO_Y = 4
WRIST_GYRO_Z = 5
WRIST_ROT_X = 6
WRIST_ROT_Y = 7
WRIST_ROT_Z = 8

ANKLE_ACCEL_X = 9
ANKLE_ACCEL_Y = 10
ANKLE_ACCEL_Z = 11
ANKLE_GYRO_X = 12
ANKLE_GYRO_Y = 13
ANKLE_GYRO_Z = 14
ANKLE_ROT_X = 15
ANKLE_ROT_Y = 16
ANKLE_ROT_Z = 17

SENSOR_CHANNELS = [WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z,
                   WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z,
                   WRIST_ROT_X, WRIST_ROT_Y, WRIST_ROT_Z,
                   # ANKLE
                   ANKLE_ACCEL_X, ANKLE_ACCEL_Y, ANKLE_ACCEL_Z,
                   ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z,
                   ANKLE_ROT_X, ANKLE_ROT_Y, ANKLE_ROT_Z]

EXPERIENCE_LEVEL_MAP = {
    "Viviane des": 2,
    "Ada def": 3,
    "Adrian stetter": 3,
    "Agustin diaz": 2,
    "Alberto sanchez": 3,
    "Ale forino": 1,
    "Alex mil": 1,
    "Alex turicum": 3,
    "Andrea Soro": 2,
    "Anja vont": 2,
    "Anna fertig": 1,
    "Beni fueg": 1,
    "Camilla cav": 2,
    "Conan obri": 2,
    "Corneel van": 2,
    "Damian ga": 2,
    "Daniel luetolf": 1,
    "David geiter": 1,
    "Denis kara": 1,
    "Denis karatwo": 1,
    "Desiree Heller": 3,
    "Donato pari": 2,
    "Fra lam": 2,
    "Georg poll": 1,
    "Karl dei": 3,
    "Karol Wojtas": 1,
    "Lara riparip": 1,
    "Lisa bra": 3,
    "Llorenc mon": 2,
    "Lukas hofm": 2,
    "Marcel feh": 2,
    "Martin butt": 1,
    "Matt senn": 3,
    "Matteo": 1,
    "Max abe": 2,
    "Mike jiang": 1,
    "Muriel haug": 1,
    "Nick lad": 3,
    "Ramon fan": 2,
    "Raphael riedo": 2,
    "Renata farkas": 3,
    "Riccardo rigamonti": 3,
    "Seba curi": 2,
    "Simon bod": 2,
    "Simone pira": 3,
    "Starkaor Hrobjartsson": 2,
    "Tobi bro": 1,
    "Virginia Storni": 2
}

exercise_colors = {1: "#b82c2c",  # push up
                   2: "#f49f59",  # pull up
                   3: "#fffa8b",  # burpees
                   4: "#386b30",  # dead liftc
                   5: "#ec2d4a",  # box jumps
                   6: "#324e4a",  # squats
                   7: "#7abcc9",  # sit ups
                   8: "#b16559",  # wall balls
                   9: "#6b6f7d",  # kb press
                   10: "#fad764",  # thrusters
                   11: "white"  # null class
                   }

## Paths

copy_from_path = "controlled_workout_data/individual_dbs/"
path = "./data/constrained_workout/raw_db_data/"
path_plots = "./plots/"
unconstrained_workout_data_path = "./data/unconstrained_workout/raw_db_data/"
numpy_reps_data_path = "./data/constrained_workout/preprocessed_numpy_data/np_reps_data/"
numpy_exercises_data_path = "./data/constrained_workout/preprocessed_numpy_data/np_exercise_data/"
best_rep_counting_models_params_file_name = "best_counting_models_params.npy"
rep_counting_constrained_results_path = "./rep_counting_constrained_results/"
uncontrained_workout_data = "./data/unconstrained_workout/raw_db_data/"
constrained_workout_rep_counting_loo_results = "constrained_workout_results/constrained_workout_loo_results/"