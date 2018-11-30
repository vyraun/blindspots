import os
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import torch
import numpy

def store_exp(model_definition, loss, filename, artifact=None):
    log_param("Model Definition", model_definition)    
    log_metric("Loss", loss)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt".format(), "w") as f:
        f.write("")
    log_artifacts("outputs")

def log_loss(loss):
    log_metric("Loss", loss)

def log_hyperparameter(name, value):
    log_param(name, value)

def log_model(model):
    s = ""
    for m in model.children():
        s += str(m)
    log_param("Model Definition", s)

def log_exp(value, rname):
    exp_id = value
    mlflow.start_run(experiment_id=exp_id, run_name=rname)

def log_new(name, rname):
    exp_id = mlflow.create_experiment(name)
    #exp_id = 2 #mlflow.get_experiment_by_name(name)
    # Start run as a child of that experiment
    mlflow.start_run(experiment_id=exp_id, run_name=rname)