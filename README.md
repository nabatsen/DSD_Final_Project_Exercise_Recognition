# DSD final project

Using deep learning to do exercise classification. The project is based on the article https://www.mdpi.com/1424-8220/19/3/714 and the code published by the authors. The biggest change is replacing the CNN model used in the article with a GRU-based RNN. To facilitate running the code in a modern environment, it was converted to Python 3 and TensorFlow 2.

Note: Evaluation with only hand sensors was targeted, running anything that is not mentioned in this document is not guaranteed to work.

## Getting Started

### Prerequisites
Conda: Version 4.11.0 was used, but any should work.

Run 
```
conda create --name <name_for_your_env> --file requirements.txt
```

This will create the environment and install all the dependencies needed. Do not forget to activate the environment with 

```
conda activate <name_for_your_env>
```

and `cd` into the source directory:

```
cd HAR_Crossfit_Sensors_Code/
```

## Training the model on constrained workout with 5-fold cross validation and saving the results
Note: the results of the training are already saved in the repository, you do not have to train the model and generate the results yourself. Training might take several hours even on a GPU

Despite there being many different training procedures in the original code, only `hand_training` was adapted to use the RNN model. To run it, make sure that only the `hand_training()` call is not commented in the end of the rnn_training.py and run

```
python rnn_training.py
```

The results of the experiments are saved in a pickled object in the folder "constrained_workout_results" which is then read in the `print_and_plot_final_results.py` script.

## Print the results and plot the confusion matrix for the 5-fold cross-validation on constrained workout
Make sure only the `plot_hand_confusion_matrix()` call is uncommented in the main section of `print_and_plot_final_results.py` and then run:

```
python print_and_plot_final_results.py
```
This file will print the resulting accuracy and other metrics for the hand training, plot the confusion matrix and save it into a file in the `plots` folder.

## Training the model to be used on the unconstrained workout.

Note: the trained model is already saved in the repository, you do not have to train it yourself. Training might take several hours even on a GPU

In rnn_training.py uncomment the line:
```
train_and_save_recognition_model_with_non_null_class()
```

Make sure all the other calls under `if __name__ == "__main__":` are commented out and run

```
python rnn_training.py
```

This will train a model using the whole constrained workout dataset as the training data and save it into `models/recognition_model_with_null.h5` . The saved model is required to perform the evaluation on unconstrained workout.

## Unconstrained workout results
Make sure only the `print_results_for_free_workout(FREE_WORKOUT_10_REPS)` call is uncommented in the main section of `unconstrained_workout_evaluation.py` and then run:

```
python unconstrained_workout_evaluation.py
```

It plots the unconstrained workout chronological progression for every participant and saves the plots in the `plots` folder.
