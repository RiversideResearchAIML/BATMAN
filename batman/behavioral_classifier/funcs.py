"""Functions to be called in the main controller
for the behavioral classifier"""
import abc
import optuna
import keras
import pandas as pd
import os
from datetime import datetime
import sys
import logging
from params import load_params
import pymongo
import numpy as np
import json


class Model(abc.ABC):
    """Abstract model class, to best work for the optimizers"""
    def __init__(self) -> None:
        ...
    
    def get_dataset(self) -> pd.DataFrame:
        """Loads, and returns the dataset"""
        ...

    def define_model(self, parms:dict) -> None:
        """Configures the model"""
        ...

    def model_compile(self, params:dict) -> None:
        """Compiles the model.
        
        Params shoud contain the optimizer, loss function, and metrics"""
        ...

    def fit(self, dataset:pd.DataFrame, params:dict, callbacks:list) -> None:
        """Fits the model to some dataset"""
        ...


def objective(trial, model:Model, params:dict) -> float:
    """Objective function for optuna study
    
    Parameters
    ----------
    trial: optuna trial, which is sent from the tune)_ function below
    model: a custom datatype defined above. Trying to generalize the models to work
           for TF or sci-kit
    params: a dictionary with all the define parameters for the model, and tuning session. Look in params.py

    Returns
    -------
    float, of some score. This is what optuna will tune against. The direction of the tune is defeined in the tune() function.
    """
    #pick hyperparameters

    #EXAMPLES from YOLO
    params["batch_sz"] = trial.suggest_int("batch_sz", 10, 32, step=2) #16
    params["learning_rate"] = trial.suggest_float("learning rate", 1e-6, 1e-2, log=True) #.005
    params["conf_thresh"] = trial.suggest_float("conf_thresh", 0.2, 0.6) #.03
    params["initial_epochs"] = trial.suggest_int("epochs", 2, 50) #2
    params["l_obj"] = trial.suggest_discrete_uniform("l obj", 1, 20, q=.1) #10
    params["l_wh"] = trial.suggest_discrete_uniform("l wh", 0.1, 8, 0.1) #1

    #load dataset
    dataset = model.get_dataset()

    ###############################################################################
    #load data, define model
    model.define_model()
    #model1.summary()

    #define callbacks. 
    #This is for TF models to check for val values to save the model at the end of every epoch

    model_check_callback = keras.callbacks.ModelCheckpoint(filepath=f"{params['temp_dir_name']}/best_model.hdf5",
            monitor="val_f1mean", save_weights_only=False, mode="max", save_best_only=True, save_freq="epoch",
            verbose=1)

    callbacks = [model_check_callback]
    
    #compile and run the model
    model.compile(params)

    #fit the model
    model.fit(dataset, params, callbacks)

    #loads the best saved model from the model check call back
    best_model = keras.models.load_model(f"{params['temp_dir_name']}/best_model.hdf5",
                                         custom_objects=params['custom_objects'])

    #EXAMPLE from YOLO
    #The metric for optuna is taken from the validation set
    val_metrics = best_model.evaluate(dataset["validation_set"],dataset["validation_set_targets"], 
            verbose=0, batch_size=params["batch_sz"])   #verbose=0 silences TF
    #displayes the metrics from the validation set
    print(f"Trial {trial.number} with best epoch metrics {val_metrics}")
    #in this example, the optuna objective metric if the validation f1mean 
    obj_metric = val_metrics[10]
    
    #---------------------------------------------------------------------
    """okay, bare with me here. I think this is neat. Hall of Fame implementation of trials. 
    IE. save the best models from the tuning session.
    Each saved model is the model from the model check callback.

    """
    #setting these bools to default false
    save = False
    sort = False

    #base option. not enough entries in the list, so always append.
    #this path is followed until enough trials have been run, to start pruning the worst performing models
    if len(params['best_model_scores']) < params["total_saved_models"]:
        #not enough models have been collected, so append to list
        params['best_model_scores'].append((trial.number, obj_metric))
        save = True
        sort = True
    
    elif obj_metric < params['best_model_scores'][0][1]:
        #Check if the current trial performed worse than the worst performing
        #in the current HOF. If so, just prune/remove it (by just passing)
        ...
    
    else:
        #result is somewhere in the top 5.
        remove_model = params['best_model_scores'].pop(0)
        os.remove(f"{params['temp_dir_name']}/trained_model__{remove_model[0]}.h5")
        #add the new file to the list
        params['best_model_scores'].append((trial.number, obj_metric))
        sort = True
        save = True
        
    if sort:
        #sorts the list by the second value of the tuple entry, which is the obj metric
        params['best_model_scores'].sort(key=lambda x : x[1])
    
    if save:
        #save the trained model and set of hyper parameters
        best_model.save(f"{params['temp_dir_name']}/trained_model__{trial.number}.h5")
    
    #return the obj for otpuna to tune over
    return obj_metric

def tune():
    """Performs the optuna study to look at hyperparameters"""
    #saves date and time, to make temp folder
    today = datetime.now().strftime("tuning_session__%Y_%m_%d__%H_%M_%S__utc")
    temp_dir_name = f"temp_models/{today}"
    
    model = Model()
    params = load_params()
    
    #check if the temp directory exists
    if not os.path.isdir(temp_dir_name):
        #create folder
        #IMPORTANT!!! FOLDER IS NOT TRACKED WITH GIT
        #MOVE ALL MODELS TO saved_models
        os.makedirs(temp_dir_name)
    
    #trick to pass parameters to optuna objective function
    obj = lambda trial: objective(trial, temp_dir_name)

    db_name = f"sqlite:///{temp_dir_name}/study.db"
    
    study_name = f"{today}"
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.CmaEsSampler(),
            study_name=study_name, storage=db_name)
    study.optimize(obj, n_trials=30)

    output_path = f"{temp_dir_name}/results.txt"
    load_study(study_name=study_name, study_path=db_name, output_file=output_path)
    

    dataset = model.get_dataset()

    #loads the best model
    best_model = keras.models.load_model(f"{temp_dir_name}/best_model.hdf5",
                                         custom_objects=params['custom_objects'])


    val_metrics = best_model.evaluate(dataset["validation_set"],dataset["validation_set_targets"], 
            verbose=0, batch_size=params["batch_sz"])

    print(f"Best val f1 mean is {val_metrics[10]}")
    #print out metrics of this model here with best_model

def load_study(study_name:str, study_path:str, output_file:str):
    """Loads an optuna study, and displays the relevant information"""
    study = optuna.load_study(study_name=study_name, storage=study_path)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    temp = f"Study statistics: \n"
    temp += f"  Number of finished trials: {len(study.trials)}\n"
    temp += f"  Number of pruned trials: {len(pruned_trials)}\n"
    temp += f"  Number of complete trials: {len(complete_trials)}\n"

    temp += f"\nBest trial: {trial.number=}\n"
    trial = study.best_trial

    temp += f"  Value: {trial.value}\n"

    temp += f"  Params: \n"
    for key, value in trial.params.items():
        temp += f"    {key}: {value}\n"

    temp += "\nFull Results\n"
    
    temp += str(study.trials_dataframe(attrs=("number", "value", "params", "state")))

    print(temp)

    with open(output_file, 'w') as f:
        f.write(temp)

# Function that connects to DocumentDB and returns data as a dataframe
def db_to_df(uri) -> pd.DataFrame:
    client = pymongo.MongoClient(uri)
    db = client.sample_database
    collection = db.collection_name
    data = pd.DataFrame(list(collection.find()))
    return data

# Function that imports dataframe to DocumentDB
def df_to_db(non_ais, data:pd.DataFrame):
    client = pymongo.MongoClient(non_ais)
    db = client.sample_database
    collection = db.collection_name
    data.reset_index(inplace=True)
    data_dict = data.to_dict("records")
    collection.insert_many(data_dict)

def json_flatten(nested_json, flattened_dict={}):
    """
    This function takes a hierarchical, nested JSON file and flattens it to an unpacked dictionary
   
    Author: Michael M. Jerge

    Code modified from flatten_json

    Parameters
    ----------
    x: JSON file
    default type_name: empty string
    Returns
    -------
    dictionary containing former json FIELDS
    """

    def flatten(x, type_name=""):
        if type(x) is dict:
            for key in x:
                flatten(x[key], type_name + key + "_")
        elif type(x) is list:
            for i in range(len(x)):
                flatten(x[i], type_name + str(i) + "_")
        else:
            flattened_dict[type_name] = x
        return flattened_dict

    return flatten(nested_json)


def main():
    
    URI = None
    NONAIS_URI = " "

    #db_to_df(URI)
    test_json_flatten()

if __name__ == "__main__":
    main()
