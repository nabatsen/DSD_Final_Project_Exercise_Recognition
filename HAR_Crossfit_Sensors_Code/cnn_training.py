# set the matplotlib backend so figures can be saved in the background
import argparse
import os

import psutil
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import argmax
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import rep_counting
from constants import WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z, ANKLE_ACCEL_X, ANKLE_ACCEL_Y, \
    ANKLE_ACCEL_Z, WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z, ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z, ANKLE_ROT_Z, \
    ANKLE_ROT_Y, ANKLE_ROT_X, WRIST_ROT_Z, WRIST_ROT_Y, WRIST_ROT_X, best_rep_counting_models_params_file_name, \
    numpy_exercises_data_path, ACCELEROMETER_CODE, GYROSCOPE_CODE, constrained_workout_rep_counting_loo_results
from majority_voting import get_majority_voting_performance_values
from results import CVResult
from simple_models import random_forest_param_selection, knn_param_selection, svc_param_selection
from utils import *


now = datetime.datetime.now()
start_time = now.strftime("%Y-%m-%d %H:%M")
nb_classes = 10
config = yaml_loader("./config_cnn.yaml")


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", help="Comma separated GPUS to run the algo on")
args = parser.parse_args()
gpus = config.get("cnn_params")['gpus']

if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import tensorflow as tf
import keras
from keras import Sequential, Input, Model
from keras.layers import Activation, Flatten, Dropout, Dense, Convolution2D, K, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model

from data_loading import get_grouped_windows_for_exerices, get_grouped_windows_for_rep_transistion_per_exercise
from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
test_accuracy_over_N = []
train_accuracy_over_N = []



class TrainingParameters:

    def __init__(self, window_step_slide, window_length):
        self.window_step_slide = window_step_slide
        self.window_length = window_length

    def __str__(self):
        return 'wl:{}, step:{}'.format(self.window_length, self.window_step_slide)


class TrainingRepCountingParameters(TrainingParameters):

    def __init__(self, exercise, window_step_slide, window_length, rep_start_portion=0.5, conv_layers=3, strides=(3, 1),
                 layers=5, activation_fun="relu", with_dropout=True, batch_normalization=False, normalize_input=False,
                 with_extra_layers=False,
                 sensors_as_channels_model=False):
        TrainingParameters.__init__(self, window_step_slide, window_length)
        self.conv_layers = conv_layers
        self.exercise = exercise
        self.strides = strides
        self.layers = layers
        self.with_extra_layers = with_extra_layers
        self.activation_fun = activation_fun
        self.normalize_input = normalize_input
        self.with_dropout = with_dropout
        self.batch_normalization = batch_normalization
        self.rep_start_portion = rep_start_portion
        self.sensors_as_channels = sensors_as_channels_model

    def __str__(self):
        return 'ex:{}, '.format(self.exercise) + TrainingParameters.__str__(self)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def get_statistics(a):
    max = np.max(a)
    min = np.min(a)
    std = np.std(a)
    avg = np.mean(a)
    return {"max": round(max, 2), "min": round(min, 2), "std": round(std, 2), "avg": round(avg, 2)}


class AccuracyHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.final_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

    def on_train_end(self, logs={}):
        # f = open("./reports/report_" + start_time + ".txt", "a+")
        # f.write("Val Accuracy:  %s\r\n" % self.val_acc[len(self.val_acc) - 1])
        # f.write("Accuracy:  %s\r\n" % self.acc[len(self.acc) - 1])
        # f.close()
        train_accuracy_over_N.append(self.acc[len(self.acc) - 1])
        test_accuracy_over_N.append(self.val_acc[len(self.val_acc) - 1])


history = AccuracyHistory()


def early_stopping(patience=15, monitor_value='val_acc'):
    return EarlyStopping(monitor=monitor_value,
                         min_delta=0.001,
                         patience=patience,
                         verbose=0,
                         mode='auto')


def get_model_checkpoint(name="weights.best.hdf5"):
    return ModelCheckpoint(name, save_best_only=True, monitor='val_acc', mode='max')


def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()


def rep_counting_model(input_shape,
                       strides,
                       layers=5,
                       filters=None,
                       with_dropout=True,
                       inner_dense_layer_neurons=250,
                       n_classes=2,
                       activation_fun="relu",
                       with_extra_layers=False,
                       batch_normalization=False):
    # sensor numpy_data_01
    K.clear_session()
    dropout = [0.5, 0.5, 0.5, 0.5, 0.5]
    if filters is None:
        filters = [100, 25, 75, 75, 25]
    conv_input = Input(shape=input_shape)
    input = conv_input
    for i in range(0, layers):
        # if i == 0:
        #     kernel_size = (12, 3)
        # else:
        kernel_size = (12, 18)

        conv_output = Convolution2D(filters=filters[i], kernel_size=kernel_size, strides=strides,
                                    input_shape=input_shape,
                                    border_mode='same',
                                    data_format="channels_last")(input)
        act = Activation(activation_fun)(conv_output)
        if batch_normalization:
            act = BatchNormalization(axis=2)(act)
        if with_dropout:
            after_dropout = Dropout(dropout[i])(act)
        else:
            after_dropout = act
        input = after_dropout

    flattened = Flatten()(after_dropout)

    # Merge and add dense layer
    pre_output = Dense(inner_dense_layer_neurons)(flattened)
    if (with_extra_layers):
        pre_output = Dense(int(inner_dense_layer_neurons / 2))(pre_output)
        pre_output = Dense(int(inner_dense_layer_neurons / 4))(pre_output)
        pre_output = Dense(int(inner_dense_layer_neurons / 8))(pre_output)
    output = (Dense(n_classes))(pre_output)
    output2 = Activation('softmax')(output)

    parameters = {'strides': strides, 'layers': layers, "filters": filters, 'kernel_size': kernel_size,
                  'with_dropout': with_dropout, 'inner_dense_layer_neurons': inner_dense_layer_neurons,
                  'activation_fun': activation_fun, 'batch_normalization': batch_normalization,
                  'with_extra_layers': with_extra_layers}

    # gr model with two inputs
    model = Model(inputs=[conv_input], outputs=[output2])
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return {"model": model, 'params': parameters}


def model_I(input_shape,
            conv_layer_1_filters=100, dropout_1=0.5,
            conv_layer_2_filters=25, dropout_2=0.5,
            conv_layer_3_filters=75, dropout_3=0.5,
            conv_layer_4_filters=75, dropout_4=0.5,
            conv_layer_5_filters=25, dropout_5=0.5,
            first_layer_kernel_size=(15, 3),
            first_layer_strides=(1, 3),
            inner_dense_layer_neurons=250,
            n_classes=nb_classes,
            activation_function="relu",
            with_dropout=True,
            with_input_normalization=False,
            with_batch_normalization=False
            ):
    print(n_classes)
    K.clear_session()
    model = Sequential()
    print("model shape is " + str(input_shape))

    if with_input_normalization:
        model.add(BatchNormalization(axis=2))
    model.add(
        Convolution2D(filters=conv_layer_1_filters, kernel_size=first_layer_kernel_size, strides=first_layer_strides,
                      input_shape=input_shape,
                      border_mode='same',
                      data_format="channels_last"))
    model.add(Activation(activation_function))
    if with_batch_normalization:
        model.add(BatchNormalization(axis=2))
    if with_dropout:
        model.add(Dropout(dropout_1))
    model.add(Convolution2D(filters=conv_layer_2_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    if with_batch_normalization:
        model.add(BatchNormalization(axis=2))
    if with_dropout:
        model.add(Dropout(dropout_2))
    model.add(Convolution2D(filters=conv_layer_3_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if with_batch_normalization:
        model.add(BatchNormalization(axis=2))
    if with_dropout:
        model.add(Dropout(dropout_3))
    model.add(Convolution2D(filters=conv_layer_4_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if with_batch_normalization:
        model.add(BatchNormalization(axis=2))
    if with_dropout:
        model.add(Dropout(dropout_4))
    model.add(Convolution2D(filters=conv_layer_5_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if with_batch_normalization:
        model.add(BatchNormalization(axis=2))
    if with_dropout:
        model.add(Dropout(dropout_5))
    model.add(Flatten())

    model.add(Dense(inner_dense_layer_neurons))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # plot_model(model, show_shapes=True, to_file='model.png')
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # mem = get_mem_usage()
    # print()
    # print()
    # print()
    # print('mem: {}'.format(mem))
    # print(model.summary())
    # print()
    # print()
    return model


def model_I_experiment(input_shape,
                       conv_layer_1_filters=25, dropout_1=0.5,
                       conv_layer_2_filters=25, dropout_2=0.5,
                       conv_layer_3_filters=75, dropout_3=0.5,
                       strides=(3, 1),
                       dropout=True,
                       with_extra_dense_layers=False,
                       batch_normalization=False,
                       inner_dense_layer_neurons=250,
                       n_classes=nb_classes,
                       activation_function="relu",
                       ):
    K.clear_session()
    model = Sequential()
    print("model shape is " + str(input_shape))

    kernel_size = (15, 3)
    model.add(
        Convolution2D(filters=conv_layer_1_filters, kernel_size=kernel_size, strides=strides,
                      input_shape=input_shape,
                      border_mode='same',
                      data_format="channels_last"))
    model.add(Activation(activation_function))
    if dropout:
        model.add(Dropout(dropout_1))
    model.add(
        Convolution2D(filters=25, kernel_size=kernel_size, strides=strides,
                      input_shape=input_shape,
                      border_mode='same',
                      data_format="channels_last"))
    model.add(Activation(activation_function))
    if batch_normalization:
        model.add(BatchNormalization(axis=2))
    if dropout:
        model.add(Dropout(dropout_2))

    model.add(Convolution2D(filters=75, kernel_size=kernel_size, strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if batch_normalization:
        model.add(BatchNormalization(axis=2))
    if dropout:
        model.add(Dropout(dropout_3))

    model.add(Convolution2D(filters=75, kernel_size=kernel_size, strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if batch_normalization:
        model.add(BatchNormalization(axis=2))
    if dropout:
        model.add(Dropout(dropout_3))

    model.add(Convolution2D(filters=25, kernel_size=kernel_size, strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation(activation_function))
    if batch_normalization:
        model.add(BatchNormalization(axis=2))
    if dropout:
        model.add(Dropout(dropout_3))
    model.add(Flatten())

    model.add(Dense(inner_dense_layer_neurons))
    if with_extra_dense_layers:
        model.add(Dense(int(inner_dense_layer_neurons / 2)))
        model.add(Dense(int(inner_dense_layer_neurons / 4)))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # plot_model(model, show_shapes=True, to_file='model.png')
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    mem = get_mem_usage()
    print()
    print()
    print()
    print('mem: {}'.format(mem))
    # print(model.summary())
    print()
    print()
    parameters = {'strides': strides, 'layers': 3, 'kernel_size': kernel_size,
                  'with_dropout': dropout, 'inner_dense_layer_neurons': inner_dense_layer_neurons,
                  'activation_fun': activation_function, 'batch_normalization': batch_normalization,
                  'with_extra_layers_dense_layers': with_extra_dense_layers}

    return {"model": model, "params": parameters}




def grid_search(model, X, Y, groups):
    # define params
    print("start Gridsearch")

    # params
    conv_layer_neurons = [50]
    # dropout_rate = [0.4, 0.6, 0.8]
    # inner_dense_layer_neurons = [100, 250, 500]

    conv_layer_1_filters = [50, 100]
    dropout_1 = [0.5]
    conv_layer_2_filters = [50, 100]
    dropout_2 = [0.5]
    conv_layer_3_filters = [50, 100]
    dropout_3 = [0.5]
    conv_layer_4_filters = [50, 100]
    dropout_4 = [0.5]
    conv_layer_5_filters = [50, 100]
    dropout_5 = [0.5]
    input_shape = [(X.shape[1], X.shape[2], 1)]
    n_conv_layer = [1, 2, 3, 5]
    param_grid = dict(input_shape=input_shape, conv_layer_1_filters=conv_layer_1_filters,
                      dropout_1=dropout_1,
                      conv_layer_2_filters=conv_layer_2_filters,
                      dropout_2=dropout_2,
                      conv_layer_3_filters=conv_layer_3_filters,
                      dropout_3=dropout_3,
                      conv_layer_4_filters=conv_layer_4_filters,
                      dropout_4=dropout_4,
                      conv_layer_5_filters=conv_layer_5_filters,
                      dropout_5=dropout_5,
                      epochs=[5],
                      batch_size=[10])

    model = KerasClassifier(build_fn=model, verbose=1)
    logo = GroupShuffleSplit(test_size=0.2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, verbose=10, n_jobs=1)
    print("Gridsearch fit")
    grid_result = grid.fit(X, Y, groups=groups)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    f = open("./reports/grid_search" + start_time + ".tx"
                                                    "t", "a+")
    for mean, stdev, param in zip(means, stds, params):
        f.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        print("%f (%f) with: %r" % (mean, stdev, param))
    f.close()
    return grid.best_estimator_


def grid_search_over_window_length():
    for wl in [ 1500, 2000, 3000, 4000, 5000, 7000, 10000]:
        result = CVResult("grid_search_over_" + str(wl) + "_window_length")
        X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False,
                                                                               config=config,
                                                                               window_length=wl,
                                                                               with_null_class=False)
        print(X.shape)
        gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
        for train_index, test_index in gss.split(X, Y, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(
                Y[test_index] - 1,
                10)

            model = model_I((X_train.shape[1], X_train.shape[2], 1),
                            first_layer_strides=(1, 3),
                            n_classes=10)

            # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
            if len(gpus) <= 1:
                print("[INFO] training with 1 GPU...")
                # otherwise, we are compiling using multiple GPUs
            else:
                print("[INFO] training with {} GPUs...".format(gpus))

                model = multi_gpu_model(model, gpus=len(gpus))

            print_asterisks_lines(3)
            test_people = persons[test_index]
            print("Testing on ")
            print(np.unique(test_people))
            print_asterisks_lines(3)
            history = model.fit(X_train, y_train,
                                epochs=config.get("cnn_params")['epochs'],
                                batch_size=config.get("cnn_params")['batch_size'],
                                validation_data=(X_test, y_test),
                                verbose=1,
                                callbacks=[early_stopping(patience=5), get_model_checkpoint("weights2.best.hdf5")])
            # load weights
            model.load_weights("weights2.best.hdf5")
            # Compile model (required to make predictions)
            sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            predicted_values = model.predict_classes(X_test)
            truth_values = argmax(y_test, axis=1)
            test_exercises_ids = exercise_ids[test_index]
            # print(np.unique(predicted_values))
            # print(np.unique(truth_values))
            result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                               test_exercise_ids=test_exercises_ids)


def split_train_test(X, y, groups, n_classes=nb_classes, test_size=0.1):
    gss = GroupShuffleSplit(test_size=test_size, random_state=RANDOMNESS_SEED)
    train_indexes, test_indexes = gss.split(X, y, groups=groups).next()
    X_train = X[train_indexes]
    y_train = np_utils.to_categorical(y[train_indexes] - 1, n_classes)
    groups_train = groups[train_indexes]
    X_test = X[test_indexes]
    y_test = np_utils.to_categorical(y[test_indexes] - 1, n_classes)
    groups_test = groups[test_indexes]
    return (X_train, y_train, groups_train), (X_test, y_test, groups_test)


def grid_search_single_rep_counting_model(model, X, Y, groups):
    print("start Gridsearch")

    # params
    conv_layer_neurons = [50]
    # dropout_rate = [0.4, 0.6, 0.8]
    # inner_dense_layer_neurons = [100, 250, 500]

    conv_layer_1_filters = [100]
    dropout_1 = [0.5]
    conv_layer_2_filters = [25]
    dropout_2 = [0.5]
    conv_layer_3_filters = [75]
    dropout_3 = [0.5]
    conv_layer_4_filters = [75]
    dropout_4 = [0.5]
    conv_layer_5_filters = [25]
    dropout_5 = [0.7]
    input_shape = [(X.shape[1], X.shape[2], 1)]
    param_grid = dict(input_shape=input_shape, conv_layer_1_filters=conv_layer_1_filters,
                      dropout_1=dropout_1,
                      conv_layer_2_filters=conv_layer_2_filters,
                      dropout_2=dropout_2,
                      conv_layer_3_filters=conv_layer_3_filters,
                      dropout_3=dropout_3,
                      conv_layer_4_filters=conv_layer_4_filters,
                      dropout_4=dropout_4,
                      conv_layer_5_filters=conv_layer_5_filters,
                      dropout_5=dropout_5,
                      epochs=[25],
                      batch_size=[30])

    model = KerasClassifier(build_fn=model, verbose=1)
    logo = GroupShuffleSplit(test_size=0.2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, verbose=10, n_jobs=1)
    print("Gridsearch fit")
    grid_result = grid.fit(X, Y, groups=groups)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    f = open("./reports/grid_search" + start_time + ".tx"
                                                    "t", "a+")
    for mean, stdev, param in zip(means, stds, params):
        f.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        print("%f (%f) with: %r" % (mean, stdev, param))
    f.close()
    return grid.best_estimator_


def update_best_model_paramters(ex, model_params):
    print(model_params)
    files = os.listdir("./")
    print(files)
    print("looking for {}".format(best_rep_counting_models_params_file_name))
    if best_rep_counting_models_params_file_name not in files:
        print("Best params file not found. Initialiazing it now".format(ex))
        best_counting_models_params = {}
    else:
        print("Best params file found".format(ex))
        best_counting_models_params = np.load(best_rep_counting_models_params_file_name).item()
    if ex not in best_counting_models_params.keys():
        print("best params for {} not found, adding it now for the first time ".format(ex))
        best_counting_models_params[ex] = model_params
    else:
        print(" best params for {} already exist".format(ex))
        previous_best = best_counting_models_params[ex]
        cv_scores_prev = previous_best['cv_scores']
        max_best = np.mean(cv_scores_prev)
        new_val_cv_scores = model_params['cv_scores']
        new_max = np.mean(new_val_cv_scores)
        print("old mean is {}".format(max_best))
        print("new mean is {}".format(new_max))
        if new_max > max_best:
            print("new best params for {} found ".format(ex))
            best_counting_models_params[ex] = model_params
        else:
            print("old best params for {} are still better".format(ex))

    np.save(best_rep_counting_models_params_file_name, best_counting_models_params)


def rep_counting_cv(training_parameters,
                    activation_fun="relu",
                    normalize_input=False,
                    with_dropout=True,
                    batch_normalization=False,
                    with_extra_layers=False,
                    experimental=False):
    data_per_exercise = get_grouped_windows_for_rep_transistion_per_exercise(training_params=training_parameters,
                                                                             config=config,
                                                                             use_exercise_code_as_group=True)
    for ex in data_per_exercise.keys():
        print(training_parameters[ex])
        X, classes, Y, groups = data_per_exercise[ex]
        if normalize_input:
            X, _ = standard_scale_data(X, None)
        if experimental:
            X = X[:, :, [WRIST_ACCEL_X, WRIST_GYRO_X, WRIST_ROT_X, ANKLE_ACCEL_X, ANKLE_GYRO_X, ANKLE_ROT_X,
                         WRIST_ACCEL_Y, WRIST_GYRO_Y, WRIST_ROT_Y, ANKLE_ACCEL_Y, ANKLE_GYRO_Y, ANKLE_ROT_Z,
                         WRIST_ACCEL_Z, WRIST_GYRO_Z, WRIST_ROT_Z, ANKLE_ACCEL_Z, ANKLE_GYRO_Z, ANKLE_ROT_Z], :]
            X = np.reshape(X, (X.shape[0], X.shape[1], 3, 6))
        test_size = 0.2
        gss = GroupShuffleSplit(test_size=test_size, n_splits=5, random_state=RANDOMNESS_SEED)
        cv_val_acc = []
        # cv

        for train_index, test_index in gss.split(X, Y, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 2), np_utils.to_categorical(
                Y[test_index] - 1,
                2)
            np.set_printoptions(linewidth=np.inf)
            if not experimental:
                model_and_params = rep_counting_model((X.shape[1], X.shape[2], 1),
                                                      strides=training_parameters[ex].strides,
                                                      batch_normalization=batch_normalization,
                                                      with_extra_layers=with_extra_layers,
                                                      activation_fun=activation_fun,
                                                      with_dropout=with_dropout)
            else:
                model_and_params = model_I_experiment((X.shape[1], X.shape[2], X.shape[3]), n_classes=2,
                                                      dropout=with_dropout,
                                                      batch_normalization=batch_normalization,
                                                      with_extra_dense_layers=with_extra_layers)
            if len(gpus) <= 1:
                print("[INFO] training with 1 GPU...")
                model = model_and_params["model"]
            # otherwise, we are compiling using multiple GPUs
            else:
                print("[INFO] training with {} GPUs...".format(gpus))

                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    # initialize the model
                    model = model_and_params["model"]
                # make the model parallel
                model = multi_gpu_model(model, gpus=len(gpus))

            history = model.fit(X_train, y_train, epochs=config.get("cnn_params")['epochs'],
                                validation_data=(X_test, y_test),
                                batch_size=config.get("cnn_params")['batch_size'],
                                callbacks=[early_stopping(5), get_model_checkpoint("rep_counting.best.hdf5")])
            # load weights
            model.load_weights("rep_counting.best.hdf5")
            preds = model.predict(X_test)
            preds = 1 - preds.argmax(axis=1)
            # Compile model (required to make predictions)
            sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            cv_val_acc.append(np.mean(accuracy_score(1 - y_test.argmax(axis=1), preds)))

        model_params = model_and_params["params"]
        if experimental:
            model_params["experimental"] = True
        model_params["normalize_input"] = normalize_input
        model_params["test_size"] = test_size
        model_params["cv_scores"] = np.mean(np.asarray(cv_val_acc))
        if batch_normalization:
            model_params["batch_normalization"] = True
        if with_dropout:
            model_params["with_dropout"] = True
        update_best_model_paramters(ex, model_params)


def train_and_save_repetition_counting_models(training_parameters):
    data_per_exercise = get_grouped_windows_for_rep_transistion_per_exercise(training_params=training_parameters,
                                                                             config=config,
                                                                             use_exercise_code_as_group=True)
    # for ex in data_per_exercise.keys():
    for ex in ["Pull ups"]:
        print(training_parameters[ex])
        X, classes, Y, groups = data_per_exercise[ex]
        if training_parameters[ex].normalize_input:
            X, _ = standard_scale_data(X, None)
        if training_parameters[ex].sensors_as_channels:
            X = X[:, :, [WRIST_ACCEL_X, WRIST_GYRO_X, WRIST_ROT_X, ANKLE_ACCEL_X, ANKLE_GYRO_X, ANKLE_ROT_X,
                         WRIST_ACCEL_Y, WRIST_GYRO_Y, WRIST_ROT_Y, ANKLE_ACCEL_Y, ANKLE_GYRO_Y, ANKLE_ROT_Z,
                         WRIST_ACCEL_Z, WRIST_GYRO_Z, WRIST_ROT_Z, ANKLE_ACCEL_Z, ANKLE_GYRO_Z, ANKLE_ROT_Z], :]
            X = np.reshape(X, (X.shape[0], X.shape[1], 3, 6))
        loo = LeaveOneGroupOut()
        ind = random.randint(0,40)
        i =-1
        for train_index, test_index in loo.split(X, Y, groups):
            i+=1
            if i!=ind:
                continue

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 2), np_utils.to_categorical(
                Y[test_index] - 1,
                2)
            np.set_printoptions(linewidth=np.inf)
            if not training_parameters[ex].sensors_as_channels:
                model_and_params = rep_counting_model((X.shape[1], X.shape[2], 1),
                                                      strides=training_parameters[ex].strides,
                                                      batch_normalization=training_parameters[ex].batch_normalization,
                                                      with_extra_layers=training_parameters[ex].with_extra_layers,
                                                      activation_fun=training_parameters[ex].activation_fun,
                                                      with_dropout=training_parameters[ex].with_dropout)
            else:
                model_and_params = model_I_experiment((X.shape[1], X.shape[2], X.shape[3]), n_classes=2,
                                                      dropout=training_parameters[ex].with_dropout,
                                                      batch_normalization=training_parameters[ex].batch_normalization,
                                                      with_extra_dense_layers=training_parameters[ex].with_extra_layers)
            if len(gpus) <= 1:
                print("[INFO] training with 1 GPU...")
                model = model_and_params["model"]
            # otherwise, we are compiling using multiple GPUs
            else:
                print("[INFO] training with {} GPUs...".format(gpus))

                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    # initialize the model
                    model = model_and_params["model"]
                # make the model parallel
                model = multi_gpu_model(model, gpus=len(gpus))

            model.fit(X_train, y_train, epochs=config.get("cnn_params")['epochs'],
                                validation_data=(X_test, y_test),
                                batch_size=config.get("cnn_params")['batch_size'],
                                callbacks=[early_stopping(40), get_model_checkpoint("best_rep_counting_params" + ex + ".best.hdf5") ])
            model.load_weights("best_rep_counting_params" + ex + ".best.hdf5")
            sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.save("./models/best_rep_counting_model_" + ex + ".h5")
            break




def generate_rep_sequences_with_LOO(training_parameters, exercises=None):
    data_per_exercise = get_grouped_windows_for_rep_transistion_per_exercise(training_params=training_parameters,
                                                                             config=config,
                                                                             use_exercise_code_as_group=True,
                                                                             exercises=exercises)

    ex_done = []
    for ex in data_per_exercise.keys():
        if ex not in ex_done:
            ex_done.append(ex)
        print(training_parameters[ex])
        X_sequences = []
        Y_rep_count_per_sequence = []
        exercises_ids = []
        labels = []
        X, classes, Y, groups = data_per_exercise[ex]
        logo = LeaveOneGroupOut()
        # cv
        count = 0
        for train_index, test_index in logo.split(X, Y, groups):
            labels.append(ex)
            exercises_ids.append(groups[test_index])
            print(ex)
            count += 1
            print_empty_lines(3)
            print("Exs done:")
            print(ex_done)
            print("Iteraton #{} for {}".format(str(count), ex))
            print_empty_lines(3)
            print(" ")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 2), np_utils.to_categorical(
                Y[test_index] - 1,
                2)
            if training_parameters[ex].normalize_input:
                X_train, X_test = standard_scale_data(X_train, X_test)
            np.set_printoptions(linewidth=np.inf)
            model_and_params = rep_counting_model((X_train.shape[1], X_train.shape[2], 1),
                                                  strides=training_parameters[ex].strides,
                                                  layers=training_parameters[ex].conv_layers,
                                                  batch_normalization=training_parameters[ex].batch_normalization,
                                                  activation_fun=training_parameters[ex].activation_fun,
                                                  with_dropout=training_parameters[ex].with_dropout)

            if len(gpus) <= 1:
                print("[INFO] training with 1 GPU...")
                model = model_and_params["model"]
            # otherwise, we are compiling using multiple GPUs
            else:
                print("[INFO] training with {} GPUs...".format(gpus))

                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    # initialize the model
                    model = model_and_params["model"]

                # make the model parallel
                model = multi_gpu_model(model, gpus=len(gpus))

            model.fit(X_train, y_train, epochs=config.get("cnn_params")['epochs'],
                      validation_data=(X_test, y_test),
                      batch_size=config.get("cnn_params")['batch_size'],
                      callbacks=[early_stopping(15), get_model_checkpoint("generate_sequences_loo" + ex + ".best.hdf5")])
            model.load_weights("generate_sequences_loo" + ex + ".best.hdf5")
            sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            preds = model.predict(X_test)
            preds = 1 - preds.argmax(axis=1)
            truth = (1 - y_test.argmax(axis=1))
            reps_truth \
                = rep_counting.count_real_reps(truth)
            X_sequences.append(preds)
            X_sequences.append(truth)
            Y_rep_count_per_sequence.append(reps_truth)
            Y_rep_count_per_sequence.append(reps_truth)
            print("truth")
            print(truth)
            print("reps truth")
            print(reps_truth)
            print("preds")
            print(preds)
            X_sequences_np = padd_sequences(X_sequences, padding=-1)
            print("Size of X Sequences is {}".format(str(X_sequences_np.shape)))
            np.save(constrained_workout_rep_counting_loo_results + "X_sequences_" + ex, X_sequences_np)
            np.save(constrained_workout_rep_counting_loo_results + "rep_count_per_sequence_" + ex, Y_rep_count_per_sequence)
            np.save(constrained_workout_rep_counting_loo_results + "exercises_ids_" + ex, exercises_ids)


def rearrange_sensor_order(X):
    permutation = [WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z, ANKLE_ACCEL_X, ANKLE_ACCEL_Y, ANKLE_ACCEL_Z,
                   WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z, ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z,
                   WRIST_ROT_X, WRIST_ROT_Y, WRIST_ROT_Z, ANKLE_ROT_X, ANKLE_ROT_Y, ANKLE_ROT_Z]
    i = np.argsort(permutation)
    return X[:, :, i, :]


def train_and_save_recognition_model():
    X, Y, _, _, _ = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                     with_null_class=False)
    Y = np_utils.to_categorical(Y - 1, 10)
    model = model_I((X.shape[1], X.shape[2], 1))

    if len(gpus) <= 1:
        print("[INFO] training with 1 GPU...")
    else:
        print("[INFO] training with {} GPUs...".format(gpus))
        model = multi_gpu_model(model, gpus=len(gpus))

    model.fit(X, Y,
              epochs=10,
              batch_size=config.get("cnn_params")['batch_size'],
              verbose=1)
    model.save("./models/recognition_model.h5")


def train_and_save_recognition_model_with_non_null_class():
    X, Y, _, _, _ = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                     with_null_class=True)
    Y = np_utils.to_categorical(Y - 1, 11)
    model = model_I((X.shape[1], X.shape[2], 1), n_classes=11)

    if len(gpus) <= 1:
        print("[INFO] training with 1 GPU...")
        # otherwise, we are compiling using multiple GPUs
    else:
        print("[INFO] training with {} GPUs...".format(gpus))

        model = multi_gpu_model(model, gpus=len(gpus))

    model.fit(X, Y,
              epochs=10,
              batch_size=config.get("cnn_params")['batch_size'],
              verbose=1)
    model.save("./models/recognition_model_with_null.h5")


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
    ex_to_rep_traning_model_params["Crunches"] = TrainingRepCountingParameters(exercise="Crunches", window_length=200,
                                                                               window_step_slide=0.05,
                                                                               with_dropout=True,
                                                                               sensors_as_channels_model=True)
    ex_to_rep_traning_model_params["Wall balls"] = TrainingRepCountingParameters(exercise="Wall balls",
                                                                                 window_length=200,
                                                                                 with_dropout=False,
                                                                                 window_step_slide=0.05,
                                                                                 activation_fun="elu")
    ex_to_rep_traning_model_params["KB Press"] = TrainingRepCountingParameters(exercise="Kb Press", window_length=150,
                                                                               window_step_slide=0.05,
                                                                               sensors_as_channels_model=True,
                                                                               batch_normalization=True)
    ex_to_rep_traning_model_params["KB Squat press"] = TrainingRepCountingParameters("Kb Squat press",
                                                                                     window_length=150,
                                                                                     with_dropout=False,
                                                                                     window_step_slide=0.05)

    return ex_to_rep_traning_model_params


def padd_sequences(X, padding=0):
    longest = -1
    for x in X:
        if len(x) > longest:
            longest = len(x)
    if padding == 0:
        padded_X = np.zeros((len(X), longest))
    else:
        padded_X = np.ones((len(X), longest)) * padding
    for i in range(0, len(X)):
        padded_X[i, 0:len(X[i])] = X[i]
    return padded_X


def exercise_vs_null_training():
    result = CVResult("exercise_vs_null_training")
    X, Y, groups, _, _ = get_grouped_windows_for_exerices(False, config, with_null_class=True)
    Y[Y != 11] = 1
    Y[Y == 11] = 0
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, test_size=0.1)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=2,
                    first_layer_kernel_size=(15, 18),
                    first_layer_strides=(1, 3)
                    )
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1]),
                        callbacks=[early_stopping()])
    result.set_result(history.history["val_acc"])


def all_sensors_training(normalize_input=False):
    result = CVResult("all_sensor_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)
    print(X.shape)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        if normalize_input:
            X_train, X_test = standard_scale_data(X_train, X_test)

        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def hand_training(normalize_input=False):
    result = CVResult("hand_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)
    X = X[:, :, WRIST_ACCEL_X:WRIST_ROT_Z + 1, :]
    print(X.shape)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        if normalize_input:
            X_train, X_test = standard_scale_data(X_train, X_test)

        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def foot_training(normalize_input=False):
    result = CVResult("foot_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)

    X = X[:, :, ANKLE_ACCEL_X:, :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        if normalize_input:
            X_train, X_test = standard_scale_data(X_train, X_test)

        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        model.load_weights("weights.best.hdf5")
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def acc_hand_training():
    result = CVResult("acc_hand_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)

    X = X[:, :, WRIST_ACCEL_X:WRIST_ACCEL_Z + 1, :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def gyro_hand_training():
    result = CVResult("gyro_hand_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)

    X = X[:, :, WRIST_GYRO_X:WRIST_GYRO_Z + 1, :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def acc_gyro_hand_training():
    result = CVResult("acc_gyro_hand_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)

    X = X[:, :, [WRIST_ACCEL_X, WRIST_GYRO_Z + 1], :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        model.fit(X_train, y_train,
                  epochs=config.get("cnn_params")['epochs'],
                  batch_size=config.get("cnn_params")['batch_size'],
                  validation_data=(X_test, y_test),
                  callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)
def acc_gyro_training():
    result = CVResult("acc_gyro_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)

    X = X[:, :, [WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z, WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z, ANKLE_ACCEL_X,
                 ANKLE_ACCEL_Y, ANKLE_ACCEL_Z, ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z], :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        model.fit(X_train, y_train,
                  epochs=config.get("cnn_params")['epochs'],
                  batch_size=config.get("cnn_params")['batch_size'],
                  validation_data=(X_test, y_test),
                  callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def orientation_hand_training():
    result = CVResult("rot_hand_training")
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           with_null_class=False)
    X = X[:, :, WRIST_ROT_X:WRIST_ROT_Z + 1, :]
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=15), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        test_exercises_ids = exercise_ids[test_index]
        # print(np.unique(predicted_values))
        # print(np.unique(truth_values))
        result.set_results(truth_values, predicted_values, accuracy_score(truth_values, predicted_values),
                           test_exercise_ids=test_exercises_ids)


def standard_scale_data(X_train, X_test=None):
    scalers = {}
    for i in range(X_train.shape[2]):
        scalers[i] = StandardScaler()
        X_train[:, :, i, 0] = scalers[i].fit_transform(X_train[:, :, i, 0])
    if X_test is not None:
        for i in range(X_test.shape[2]):
            X_test[:, :, i, 0] = scalers[i].transform(X_test[:, :, i, 0])
    return X_train, X_test


def best_overlap_grid_search(wl=None):
    overlaps = [0.99, 0.95, 0.90, 0.80, 0.50, 0.25, 0.10, 0]
    if wl is None:
        result = CVResult("over_lap_grid_search_w_4000")
        wl = 4000
    else:
        result = CVResult("over_lap_grid_search_w_" + str(wl))
    X, Y, groups, persons, exercise_ids = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                                           window_length=wl,
                                                                           with_null_class=False)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)
        model = model_I((X_train.shape[1], X_train.shape[2], 1),
                        first_layer_strides=(1, 3),
                        n_classes=10)

        # model = model_I_experiment(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))['model']
        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            model = multi_gpu_model(model, gpus=len(gpus))

        print_asterisks_lines(3)
        test_people = persons[test_index]
        print("Testing on ")
        print(np.unique(test_people))
        print_asterisks_lines(3)
        model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping(patience=5), get_model_checkpoint()])
        # load weights
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        test_exercises_ids = exercise_ids[test_index]

        predicted_for_overlap = {}
        truth_for_overlap = {}
        test_exercise_ids_for_overlap = {}
        for overlap in overlaps:
            X_2, Y_2, groups_2, persons_2, exercise_ids_2 = get_grouped_windows_for_exerices(
                with_feature_extraction=False,
                config=config,
                window_length=wl,
                window_step=(1.0 - overlap),
                ids=test_exercises_ids,
                with_null_class=False)
            predicted_values = model.predict_classes(X_2)
            print(predicted_values)
            print(Y_2)
            predicted_for_overlap[overlap] = predicted_values
            truth_for_overlap[overlap] = Y_2
            test_exercise_ids_for_overlap[overlap] = exercise_ids_2
        result.set_results(truth_for_overlap, predicted_for_overlap, 0,
                           test_exercise_ids=test_exercise_ids_for_overlap)




def baseline_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='elu'))
    model.add(Dense(int(input_dim / 2), input_dim=input_dim, kernel_initializer='normal', activation='elu'))
    model.add(Dense(int(input_dim / 4), input_dim=int(input_dim / 2), kernel_initializer='normal', activation='elu'))
    model.add(Dense(int(input_dim / 8), input_dim=int(input_dim / 4), kernel_initializer='normal', activation='elu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model


def binary_rep_sequences_training(X_train, y_train, X_test, y_test, GT_of_loo_test, sequence_labels):
    model = baseline_model((X_train[0].shape)[0])
    model.fit(X_train, y_train,
              epochs=config.get("cnn_params")['epochs'],
              batch_size=config.get("cnn_params")['batch_size'],
              validation_data=(X_test, y_test),
              verbose=0,
              callbacks=[early_stopping(monitor_value="val_mean_squared_error")])
    preds = np.squeeze(np.around(model.predict(X_test))).astype(np.int32)
    rep_counting.print_sequence_prediction_results(GT_of_loo_test, X_test, preds, sequence_labels, y_test)
    # print("Test accuracy $d".format(mse))


def print_rep_counting_best_params():
    best_rep_counting_params = np.load(best_rep_counting_models_params_file_name).item()
    for ex, params in best_rep_counting_params.items():
        print(ex)
        for par_key, par_val in params.items():
            if not (par_key == 'val_acc' or par_key == 'acc'):
                print("{} : {}".format(par_key, par_val))


def random_search_rep_counting():
    param_dist = {
        "normalize_input": [True, False],
        "experimental": [True, False],
        "with_extra_layers": [True, False],
        "batch_normalization": [True, False],
        "with_dropout": [True, False],
        "activation_fun": ["relu", "elu"]
    }
    iterations = 50
    for i in range(iterations):
        params = get_random_params(param_dist)
        print(params)
        rep_counting_cv(init_best_rep_counting_models_params(), normalize_input=params["normalize_input"],
                        with_extra_layers=params["with_extra_layers"], with_dropout=params["with_dropout"],
                        batch_normalization=params["batch_normalization"],
                        activation_fun=params["activation_fun"])


def simple_models_grid_search():
    _, X_features, y, groups = get_grouped_windows_for_exerices(with_feature_extraction=True, config=config)
    random_forest_param_selection(X_features, y, groups)
    knn_param_selection(X_features, y, groups)
    svc_param_selection(X_features, y, groups)




if __name__ == "__main__":  #
    # ### RECOGNTION ####
    all_sensors_training()
    # hand_training()
    # foot_training()
    # acc_hand_training()
    # acc_gyro_hand_training()
    # gyro_hand_training()
    # orientation_hand_training()
    # grid_search_over_window_length()
    # best_overlap_grid_search()
    #
    # #REPETITION COUTNING
    # generate_rep_sequences_with_LOO(init_best_rep_counting_models_params())

    # #REPETITION COUTNING RANDOM SEARCH
    # random_search_rep_counting()
    # print_rep_counting_best_params() ##print the best found parameters
    #
    # #BAELINE MODELS GRID SEARCH
    # simple_models_grid_search()
    #
    # #GENERATE MODELS TO USE ON UNCONSTRAINED WORKOUT DATA
    # train_and_save_recognition_model()
    # train_and_save_recognition_model_with_non_null_class()
    # train_and_save_repetition_counting_models()
