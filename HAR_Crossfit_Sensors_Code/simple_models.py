import datetime

import numpy as np
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

from data_loading import get_grouped_windows_for_exerices, RANDOMNESS_SEED
from utils import yaml_loader

now = datetime.datetime.now()
start_time = now.strftime("%Y-%m-%d %H:%M")


config = yaml_loader("./config_cnn.yaml")


def random_forest_randomized_search():
    X = np.load("./X_sequences.npy")
    X = np.pad(X, (4, 0), 'constant', constant_values=(1,))[4:, :]
    Y = np.load("./rep_count_per_sequence.npy")
    params = {'bootstrap': [True, False],
              'max_depth': [75, 100, 125],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 5, 10, 20, 50],
              'min_samples_split': [8, 10, 15],
              'n_estimators': [350, 500, 750]}
    n_iter_search = 100
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=n_iter_search, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X, Y)
    print(rf_random.best_params_)


def linear_regression_randomized_search():
    X = np.load("./X_sequences.npy")
    X = np.pad(X, (4, 0), 'constant', constant_values=(1,))[4:, :]
    Y = np.load("./rep_count_per_sequence.npy")
    params = {'kernel': ['rbf', 'sigmoid', 'linear'],
              'gamma': [0.0001, 0.01, 0.1, 10, 100, 1000],
              'C': [0.0001, 0.01, 0.1, 10, 100, 1000]}
    n_iter_search = 100
    svr = SVR()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=svr, param_distributions=params, n_iter=n_iter_search, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X, Y)
    print(rf_random.best_params_)



def svc_param_selection(X, y, groups):
    print("svc")
    gss = GroupShuffleSplit(test_size=0.2, n_splits=5, random_state=RANDOMNESS_SEED)
    Cs = [0.1, 1, 10]
    gammas = [0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=gss, verbose=10)
    grid_search.fit(X, y, groups=groups)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print grid_search.best_params_


def knn_param_selection(X, y, groups):
    print("knn")
    gss = GroupShuffleSplit(test_size=0.2, n_splits=5, random_state=RANDOMNESS_SEED)

    param_grid = [{'algorithm': ['auto']},
                  {'n_neighbors': [30, 50, 100, 150, 200]}]
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=gss)
    grid_search.fit(X, y, groups=groups)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print grid_search.best_params_


def random_forest_param_selection(X, y, groups):
    from sklearn.model_selection import RandomizedSearchCV
    print "RF"

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    gss = GroupShuffleSplit(test_size=0.2, n_splits=5, random_state=RANDOMNESS_SEED)

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv=gss, verbose=10,
                                   random_state=RANDOMNESS_SEED, n_jobs=-1, n_iter=40)

    rf_random.fit(X, y, groups=groups)
    means = rf_random.cv_results_['mean_test_score']
    stds = rf_random.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, rf_random.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print rf_random.best_params_


def random_forest_training():
    print "Random forset training all sensors"
    _, X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=True, config=config)

    rf = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=2000, max_features='sqrt',
                               min_samples_split=5, max_depth=None)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    scores = []
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        print score
        scores.append(score)
    print(np.mean(np.asarray(scores)))


def random_forest_wrist_training():
    print "Random forset training wirst"
    _, X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=True, config=config,
                                                       smart_watches=["wrist"])

    rf = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=2000, max_features='sqrt',
                               min_samples_split=5, max_depth=None)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    scores = []
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        print score
        scores.append(score)
    print(np.mean(np.asarray(scores)))


def random_forest_foot_training():
    print "Random forset training foot"
    _, X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=True, config=config,
                                                       smart_watches=["foot"])

    rf = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=2000, max_features='sqrt',
                               min_samples_split=5, max_depth=None)
    gss = GroupShuffleSplit(test_size=0.20, n_splits=5, random_state=RANDOMNESS_SEED)
    scores = []
    for train_index, test_index in gss.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        print score
        scores.append(score)
    print(np.mean(np.asarray(scores)))


if __name__ == "__main__":
    pass
    # random_forest_training()
    # random_forest_wrist_training()
    # random_forest_foot_training()
