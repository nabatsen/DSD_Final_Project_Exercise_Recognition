import collections
import datetime
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from majority_voting import get_majority_voting_performance_values, convert_split_to_majority_voting
from utils import get_performance_values


class Result:
    def __init__(self, name):
        self.name = name
        self.directory_name = "constrained_workout_results"

    def save_result_object(self):
        file = open("./" + self.directory_name + "/" + self.name, "w+")
        now = datetime.datetime.now()
        self.last_modified = now.strftime("%Y-%m-%d %H:%M")
        pickle.dump(self, file)
        file.close()


class CVResult(Result):

    def __init__(self, name):
        Result.__init__(self, name)
        self.testing_accuracies = []
        self.truth_predicted_values_tuples = []
        self.test_exercise_ids = []

    def set_results(self, truth_values, predicted_values, testing_accuracy, test_exercise_ids):
        print(testing_accuracy)
        print(self.name)
        self.truth_predicted_values_tuples.append((truth_values, predicted_values))
        self.testing_accuracies.append(testing_accuracy)
        self.test_exercise_ids.append(test_exercise_ids)
        self.save_result_object()

    def get_number_of_cv_splits(self):
        return len(self.testing_accuracies)

    def get_performance_values_per_class(self,window_length ,with_majority_voting=True):
        performance_values_per_class = {}
        f1_per_class = {}

        for split in range(0, len(self.truth_predicted_values_tuples)):
            truth_values = self.truth_predicted_values_tuples[split][0]
            pred_values = self.truth_predicted_values_tuples[split][1]
            ex_ids = self.test_exercise_ids[split]
            classes = np.unique(truth_values)
            for cl in classes:
                indexes = np.where(truth_values == cl)
                if with_majority_voting:
                    performance_values = \
                        get_majority_voting_performance_values(truth_values[indexes], pred_values[indexes],
                                                               ex_ids[indexes], window_length, 0.05)
                else:
                    performance_values = get_performance_values(truth_values[indexes], pred_values[indexes])
                if cl not in performance_values_per_class.keys():
                    performance_values_per_class[cl] = []
                    f1_per_class[cl] = []
                performance_values_per_class[cl].append(performance_values)
        return performance_values_per_class

    def get_total_test_performance_values(self, with_majority_voting=False, window_length=4000):
        truth_values = np.asarray([])
        predicted_values = np.asarray([])
        for split in range(len(self.truth_predicted_values_tuples)):
            if with_majority_voting:
                split_truth_values, split_pred_values = convert_split_to_majority_voting(
                    self.truth_predicted_values_tuples[split][0], self.truth_predicted_values_tuples[split][1],
                    self.test_exercise_ids[split], window_length=window_length)

            else:
                split_truth_values = self.truth_predicted_values_tuples[split][0]
                split_pred_values = self.truth_predicted_values_tuples[split][1]
            truth_values = np.concatenate((truth_values, split_truth_values))
            predicted_values = np.concatenate((predicted_values, split_pred_values))
        return get_performance_values(truth_values, predicted_values)

    def get_confusion_matrix(self, with_majority_voting=True):
        truth_values = np.asarray([])
        predicted_valeus = np.asarray([])
        for split in range(len(self.truth_predicted_values_tuples)):
            if with_majority_voting:
                split_truth_values, split_pred_values = convert_split_to_majority_voting(
                    self.truth_predicted_values_tuples[split][0], self.truth_predicted_values_tuples[split][1],
                    self.test_exercise_ids[split])

            else:
                split_truth_values = self.truth_predicted_values_tuples[split][0]
                split_pred_values = self.truth_predicted_values_tuples[split][1]
            truth_values = np.concatenate((truth_values, split_truth_values))
            predicted_valeus = np.concatenate((predicted_valeus, split_pred_values))

        print(get_performance_values(truth_values, predicted_valeus))
        cm = np.zeros((10, 10))
        for i in range(0, len(truth_values)):
            cm[int(truth_values[i]), int(predicted_valeus[i])] += 1
        return cm.astype(np.int)



    def get_grid_search_parameter_test_accuracy(self, with_majority_voting=False, window_length = 4000):
        per_parameter_accuracy = {}
        per_parameter_truth_values = {}
        per_parameter_predicted_values = {}
        per_parameter_ids = {}

        for subject_id in range(len(self.truth_predicted_values_tuples)):
            subject = self.truth_predicted_values_tuples[subject_id]
            for param in subject[0].keys():
                if param not in per_parameter_predicted_values.keys():
                    per_parameter_truth_values[param] = np.asarray([])
                    per_parameter_predicted_values[param] = np.asarray([])
                    per_parameter_ids[param] = np.asarray([])
                per_parameter_truth_values[param] = np.concatenate(
                    (per_parameter_truth_values[param], subject[0][param]))
                per_parameter_predicted_values[param] = np.concatenate(
                    (per_parameter_predicted_values[param], subject[1][param] + 1))
                per_parameter_ids[param] = np.concatenate(
                    (per_parameter_ids[param], self.test_exercise_ids[subject_id][param]))
        for param in per_parameter_truth_values.keys():
            if with_majority_voting:
                print("getting majori voting perfo values for param {}".format(param))
                accuracy_ = get_majority_voting_performance_values(per_parameter_truth_values[param],
                                                                       per_parameter_predicted_values[param], per_parameter_ids[param],
                                                                       window_length=window_length, step_percentage=1 - param)[
                        "accuracy"]
                if param not in per_parameter_accuracy.keys():
                    per_parameter_accuracy[param] = []
                    per_parameter_accuracy[param].append(accuracy_)
            else:
                per_parameter_accuracy[param] = accuracy_score(per_parameter_truth_values[param],
                                                               per_parameter_predicted_values[param])
            per_parameter_accuracy[param] = np.mean(np.asarray(per_parameter_accuracy[param]))
            print("mean")
            print(per_parameter_accuracy[param])
        # import matplotlib.pyplot as plt
        # plt.plot(per_parameter_accuracy.keys(), per_parameter_accuracy.values())
        return per_parameter_accuracy

    def get_common_errors(self):
        truth_values = np.asarray([])
        predicted_valeus = np.asarray([])
        for subject in self.truth_predicted_values_tuples:
            truth_values = np.concatenate((truth_values, subject[0]))
            predicted_valeus = np.concatenate((predicted_valeus, subject[1]))
        preds_unique = np.unique(truth_values)
        for cl in preds_unique:
            print(cl)
            indeces = np.squeeze(np.argwhere(truth_values == cl))
            corresponding_predictions = predicted_valeus[indeces]
            errors = corresponding_predictions[corresponding_predictions != cl]
            counts = collections.Counter(errors)
            new_list = sorted(errors, key=counts.get, reverse=True)
            print(new_list)
