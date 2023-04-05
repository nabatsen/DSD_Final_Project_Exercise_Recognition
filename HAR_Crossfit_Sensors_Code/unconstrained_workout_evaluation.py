import numpy as np
from keras.engine.saving import load_model

from cnn_training import TrainingRepCountingParameters
from constants import EXERCISE_NAME_TO_CLASS_LABEL, EXERCISE_CLASS_LABEL_TO_NAME, FREE_WORKOUT_10_REPS, FREE_WORKOUT_123_SCHEME, \
    uncontrained_workout_data
from data_loading import extract_test_data, yaml_loader, load_rep_counting_models, extract_test_rep_data
from majority_voting import convert_to_major_voting_labels
from rep_counting import count_predicted_reps

config = yaml_loader("./config_cnn.yaml")
window_length = config.get("data_params")["window_length"]

class RecognizedExercise:
    reps = 0

    def __init__(self, ex_code, start_time, end_time):
        self.ex_code = ex_code
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return EXERCISE_CLASS_LABEL_TO_NAME[self.ex_code]

    def __repr__(self):
        return "RecognizedExercise({}, {}, {}, {})".format(self.ex_code, self.start_time, self.end_time,
                                                           self.get_duration())

    def get_duration(self):
        return self.end_time - self.start_time

    def set_windows(self, windows):
        self.windows = windows




def print_exercise_segments(preds):
    segmented_workout = []
    start = 0
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1] or i == len(preds) - 1:
            segmented_workout.append(
                {"start": start, "end": i,
                 "label": preds[i - 1]})
            start = i
    return segmented_workout


def init_best_rep_counting_models_params():
    ex_to_rep_traning_model_params = {}
    ex_to_rep_traning_model_params["Push ups"] = TrainingRepCountingParameters("Push ups", window_length=120,
                                                                               window_step_slide=0.05,
                                                                               rep_start_portion=0.5,
                                                                               strides=(1, 1),
                                                                               sensors_as_channels_model=True,
                                                                               with_dropout=True)
    ex_to_rep_traning_model_params["Pull ups"] = TrainingRepCountingParameters("Pull ups", window_length=200,
                                                                               sensors_as_channels_model=True,
                                                                               normalize_input=True,
                                                                               window_step_slide=0.05)
    ex_to_rep_traning_model_params["Burpees"] = TrainingRepCountingParameters("Burpees", window_length=250,
                                                                              window_step_slide=0.05,
                                                                              activation_fun="relu",
                                                                              normalize_input=False,
                                                                              batch_normalization=False)

    ex_to_rep_traning_model_params["Dead lifts"] = TrainingRepCountingParameters("Dead lifts", window_length=200,
                                                                                 sensors_as_channels_model=True,
                                                                                 window_step_slide=0.05, layers=3,
                                                                                 strides=(1, 1),
                                                                                 with_dropout=False)
    ex_to_rep_traning_model_params["Box jumps"] = TrainingRepCountingParameters("Box Jumps", window_length=200,
                                                                                sensors_as_channels_model=True,
                                                                                window_step_slide=0.05,
                                                                                activation_fun="relu")
    ex_to_rep_traning_model_params["Squats"] = TrainingRepCountingParameters("Squats", window_length=150,
                                                                             window_step_slide=0.05,
                                                                             with_extra_layers=True,
                                                                             activation_fun="elu"
                                                                             , with_dropout=True,
                                                                             normalize_input=True)
    ex_to_rep_traning_model_params["Crunches"] = TrainingRepCountingParameters(exercise="Crunches",
                                                                               window_length=200,
                                                                               window_step_slide=0.05,
                                                                               with_dropout=True,
                                                                               sensors_as_channels_model=True)
    ex_to_rep_traning_model_params["Wall balls"] = TrainingRepCountingParameters(exercise="Wall balls",
                                                                                 window_length=200,
                                                                                 with_dropout=False,
                                                                                 window_step_slide=0.05,
                                                                                 activation_fun="elu")
    ex_to_rep_traning_model_params["KB Press"] = TrainingRepCountingParameters(exercise="Kb Press",
                                                                               window_length=150,
                                                                               window_step_slide=0.05,
                                                                               sensors_as_channels_model=True,
                                                                               batch_normalization=True)
    ex_to_rep_traning_model_params["KB Squat press"] = TrainingRepCountingParameters("Kb Squat press",
                                                                                     window_length=150,
                                                                                     with_dropout=False,
                                                                                     window_step_slide=0.05)

    return ex_to_rep_traning_model_params


def segment_workout(preds, window_length, step):
    current_ex_code = -1
    current_start = 0
    streak = 0
    recognized_exercises = []
    for i in range(0, len(preds)):
        if preds[i] != current_ex_code or i == len(preds) - 1:
            if current_ex_code != -1 and current_ex_code != 11:
                start_time = int((window_length + (current_start - 1) * step * window_length) / 10)
                end_time = int((window_length + (i - 1) * step * window_length) / 10)
                print(current_ex_code)
                min_one_rep_length = model_params[EXERCISE_CLASS_LABEL_TO_NAME[current_ex_code]].window_length
                if (end_time - start_time) >= min_one_rep_length:
                    recognized_exercises.append(
                        RecognizedExercise(current_ex_code, start_time=start_time, end_time=end_time))
            streak = 0
            current_start = i
            current_ex_code = preds[i]
        else:
            streak += 1
    return recognized_exercises


def print_results_for_free_workout(free_workout_code= FREE_WORKOUT_10_REPS):
    if free_workout_code==FREE_WORKOUT_123_SCHEME:
        print("******** 123 REPS SCHEME *********")
    elif free_workout_code==FREE_WORKOUT_10_REPS:
        print("******** 10 REPS SCHEME *********")

    global config, model_params
    with_majority_voting = True
    rep_counting_models = load_rep_counting_models()
    results = {}
    if  free_workout_code ==FREE_WORKOUT_10_REPS:
        parts = ["p1", "p2", "p3", "p4", "p5"]
    elif free_workout_code ==FREE_WORKOUT_123_SCHEME:
        parts = ["p1", "p3", "p4", "p5"]

    for participant in parts:
        print("")
        print("")
        print(participant)
        print("")
        print("")
        config = yaml_loader("./config_cnn.yaml")

        wrist_db_file = uncontrained_workout_data + "uw_" + participant + "_wrist"
        ankle_db_file = uncontrained_workout_data + "uw_" + participant + "_ankle"
        step = 0.05
        test_windows = extract_test_data(wrist_db_file, ankle_db_file,
                                         ex_code=free_workout_code, window=window_length, step=step)

        for with_null_class in [True]:

            if with_null_class:
                model = load_model('./models/recognition_model_with_null.h5')
            else:
                model = load_model('./models/recognition_model.h5')

            preds = model.predict_classes(test_windows) + 1
            preds_majority = convert_to_major_voting_labels(preds, window_length, step, with_null_class=with_null_class,
                                                            with_ties=False, with_min_rep_duration_filtering=True)
            preds_majority = preds_majority[preds_majority > 0].astype(np.int)

            save_results_in_files = False
            if save_results_in_files:
                np.save("free_workout_preds_with_null_filtering_" + str(free_workout_code) + "_" + participant, preds_majority)
                np.savetxt("free_workout_preds_with_null_filtering_" + str(free_workout_code) + "_" + participant + ".csv",
                       preds_majority,
                       delimiter=",")
            ## PLOT WORKOUT - uncomment the following line
            # plot_free_workout((preds_majority), window * step, with_null_class=with_null_class, participant_name=name)

            for seg in print_exercise_segments(preds_majority):
                print(str(seg["label"]) + ": " + str((seg["end"] - seg["start"]) * window_length * step))
        model_params = init_best_rep_counting_models_params()
        if with_majority_voting:
            recognized_exercises = segment_workout(preds_majority, window_length * step, 1)
        else:
            recognized_exercises1 = segment_workout(preds, window_length, step)

        recognized_exercises_readings = extract_test_rep_data(wrist_db_file, ankle_db_file,
                                                              recognized_exercises,
                                                              model_params=model_params,
                                                              ex_code=free_workout_code, step=0.05)
        results[participant] = []
        for rec_ex in recognized_exercises_readings:
            model = rep_counting_models[rec_ex.ex_code]
            preds = model.predict([rec_ex.windows])
            preds_rounded = 1 - preds.argmax(axis=1)
            results[participant].append({EXERCISE_CLASS_LABEL_TO_NAME[rec_ex.ex_code]: preds_rounded})
            print(EXERCISE_CLASS_LABEL_TO_NAME[rec_ex.ex_code])
            print(str(rec_ex.get_duration()))
            print(preds_rounded)
            print(count_predicted_reps(np.copy(preds_rounded)))


if __name__ == "__main__":
    print_results_for_free_workout(FREE_WORKOUT_10_REPS)
    print_results_for_free_workout(FREE_WORKOUT_123_SCHEME)