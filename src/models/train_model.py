import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml
from dotenv import load_dotenv
from pathlib import Path

from src.utils.manage_mlflow import (
    create_mlflow_experiment,
    register_best_xgboost_experiment,
    register_best_random_forest_experiment,
    register_best_model
)

from src.utils.hpo import classification_optimization

def convert_label_to_binary(label_encoder, labels):
    normal_idx = np.where(label_encoder.classes_ == 'normal')[0][0]
    my_labels = labels.copy()
    my_labels[my_labels != normal_idx] = 1 
    my_labels[my_labels == normal_idx] = 0
    return my_labels

def read_transform():
    data_path = Path("data/processed/kddcup_cleaned.parquet")
    data = pd.read_parquet(data_path)
    
    # capture the categorical variables and one-hot encode them
    cat_vars = ['protocol_type', 'service', 'flag', 'land', 'logged_in','is_host_login', 'is_guest_login']

    # find unique labels for each category
    cat_data = pd.get_dummies(data[cat_vars])
    
    numeric_vars = list(set(data.columns.values.tolist()) - set(cat_vars))
    numeric_vars.remove('label')
    numeric_data = data[numeric_vars].copy()
    
    # concat numeric and the encoded categorical variables
    x_full = pd.concat([numeric_data, cat_data], axis=1)
    
    le = LabelEncoder()
    le.fit(data.label)
    
    # capture the labels
    labels = data['label'].copy()
    
    # convert labels to integers
    integer_labels = le.transform(labels)

    
    y_full = convert_label_to_binary(le, integer_labels)

    return x_full, y_full


def main():
    """Main function to run the optimization process. It loads the training
    dataset, optimizes the hyperparameters, registers the best experiment,
    and registers the best model."""
    load_dotenv()
    
    # Load params.yaml file
    params_file = Path("params.yaml")
    modeling_params = yaml.safe_load(
        open(params_file, encoding="utf-8"))["modeling"]
    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]
    
    create_mlflow_experiment(f"{selected_model_family}_experiment")

    # Load the appropriate training dataset based on the model family
    x_train, y_train = read_transform()
    
    if selected_model_family == "catboost":
        

        best_classification_params = classification_optimization(
            x_train=x_train, y_train=y_train,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials, diagnostic=True)
        
        register_best_catboost_experiment(
            x_train=x_train, y_train=y_train,
            loss_function=selected_loss_function,
            best_params=best_classification_params)
        
        register_best_model(
            model_family=selected_model_family,
            loss_function=selected_loss_function)

    elif selected_model_family == "xgboost":

        best_classification_params = classification_optimization(
            x_train=x_train, y_train=y_train,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials, diagnostic=True)

        register_best_xgboost_experiment(
            x_train=x_train, y_train=y_train,
            loss_function=selected_loss_function,
            best_params=best_classification_params,
            )
        
        register_best_model(
            model_family=selected_model_family,
            loss_function=selected_loss_function)

    elif selected_model_family == "random_forest":

        best_classification_params = classification_optimization(
            x_train=x_train, y_train=y_train,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials, diagnostic=True)

        register_best_random_forest_experiment(
            x_train=x_train, y_train=y_train,
            loss_function=selected_loss_function,
            best_params=best_classification_params)
        
        register_best_model(
            model_family=selected_model_family,
            loss_function=selected_loss_function)

    else:
        raise ValueError(f"Unsupported model_family '{selected_model_family}'. "
                         "Supported families are 'catboost', 'xgboost', and 'random_forest'.")

if __name__ == "__main__":
    main()