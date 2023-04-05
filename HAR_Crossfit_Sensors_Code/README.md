# SPORT ANALYSIS USING SMARTWATCHES

Using deep learning to do exercise classification and repetition counting. 

## Getting Started

### Prerequisites
Conda: Version 4.4.4
Run conda create --name <name_for_your_env> --file environment_reqs.txt

This will install all the dependencies needed


### Uploading data and scripts to GPU Cluster

- In order to upload the scripts and the data for training on the GPU Clusters open the two following scripts
-- upload_data.sh
-- upload_script.sh
- Replace "soroa@tik42x.ethz.ch:/usr/itetnas03/data-tik-01/soroa/" with your GPU Cluster address
- Run the scripts

## Preprocessing
Running preprocessing.py generates the numpy training data from the database


## Running the constrained workout experiments on the GPUS
Script: cnn_training.py (Run on GPUs)
All the experiments for constrained workout are performed here. Check out the main function at the bottom and uncomment specific experiments to be run

Before running the script check the gpu numbers that are reserved using the following command:
grep -h $(whoami) /tmp/lock-gpu*/info.txt

Also update the config_cnn.yaml file with the current GPUS numbers

e.g.
gpus:
  - 2,5

Then run the script setting the currect gpus as argument :
e.g. python cnn_training.py -g 2,5

The results of the experiments are saved in special result objects in the folder "constrained_workout_results" and
are then read in the print_and_plot_final_results.py script.

## Downloading the results objects

All recognition and repetition counting experiments for the constrained workout generate result objects that are saved in the
constrained_workout_results folder.

If you are working on a remote GPU cluster and want to download them to your local machine
- open download_results_folder.py
- Replace "soroa@tik42x.ethz.ch:/usr/itetnas03/data-tik-01/soroa/" with your GPU Cluster address
- run the script

## Print the results
### Script: print_and_plot_final_results.py
See main function at the bottom of the script
In this script are also all plotting functions


## Training the models to be used on the unconstrained workout.

In cnn_training.py run:

train_and_save_recognition_model()
train_and_save_recognition_model_with_non_null_class()
train_and_save_repetition_counting_models()

Then open the donload_models.sh script:
Replace soroa@tik42x.ethz.ch:/usr/itetnas03/data-tik-01/soroa/ with your gpu cluster path
run the script


## Unconstrained workout results

Run script: unconstrained_workout_evaluation.py

For each participants prints out recognized exercises and counted repetitions
In this function one can also print out the unconstrained workout chronological progression plot by uncommenting the
commented out line:

# plot_free_workout((preds_majority), window * step, with_null_class=with_null_class, participant_name=name)
