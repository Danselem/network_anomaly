"""Manage MLflow credentials and experiments"""
import os
import random
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.visualization.visualize import plot_confusion_matrix
from src import logger



# Configuration constants
DAGSHUB_REPO_OWNER = "Danselem"
DAGSHUB_REPO = "network_anomaly"
seed = 1024

def config_mlflow() -> None:
    """Configure MLflow to log metrics to the Dagshub repository"""
    load_dotenv()
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}.mlflow")

    mlflow.autolog()
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_mlflow_experiment(experiment_name: str) -> None:
    """Create or set an MLflow experiment"""
    config_mlflow()
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            mlflow.create_experiment(experiment_name)
            logger.info(f"MLflow experiment '{experiment_name}' created.")
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to create MLflow experiment '{experiment_name}': {e}")
            raise
    else:
        logger.info(f"MLflow experiment '{experiment_name}' already exists.")
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'.")


def register_best_model(model_family: str, loss_function: str) -> None:
    """Register the best model after the optimization process.

    Args:
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
    """
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{model_family}_experiment")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{loss_function} ASC"]  # Corrected to ASC for minimum loss
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/", f"{model_family}_best_model")


def register_best_xgboost_experiment(
    x_train: ArrayLike, y_train: ArrayLike,
    loss_function: str,
    best_params: dict) -> str:
    """Register the best experiment found by the optimization process.

    Args:
        params: Hyperparameter dictionary for the given model.

    Returns:
        run_id: The ID of the run in MLflow.
    """
    total_samples_size = x_train.shape[0]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)
    
    # Load constants/categorical_features from params.yaml
    # with open("params.yaml", encoding="utf-8") as file:
    #     dvc_params = yaml.safe_load(file)

    # best_params["cat_features"] = dvc_params["categorical_features"]

    model = XGBClassifier(**best_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], ) # early_stopping_rounds=10

    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)
    test_probs = model.predict_proba(x_val)[:, 1]
    train_probs = model.predict_proba(x_train)[:, 1]
    
    signature = infer_signature(x_val, test_hat)

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(best_params)
        mlflow.log_param("total_samples_size", x_train.shape[0])

        # Log metrics for validation set
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("precision", precision_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("recall", recall_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, test_probs))

        # Log metrics for training set
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_precision", precision_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_recall", recall_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, train_probs))
        mlflow.log_param("loss_function", loss_function)
        mlflow.xgboost.log_model(model, "model", signature=signature,)

        # Ensure file paths are correct
        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")

        # Clean up files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        # mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id

def register_best_random_forest_experiment(
        x_train: ArrayLike, y_train: ArrayLike,
        loss_function: str,
        best_params: dict) -> str:
    """Register the best Random Forest experiment found by the optimization process.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        best_params: Hyperparameter dictionary for Random Forest.

    Returns:
        run_id: The ID of the run in MLflow.
    """
    total_samples_size = x_train.shape[0]
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)

    # Initialize the Random Forest model with the best parameters
    model = RandomForestClassifier(**best_params, random_state=seed)
    model.fit(x_train, y_train)

    # Predict on validation and training sets
    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)
    test_probs = model.predict_proba(x_val)[:, 1]
    train_probs = model.predict_proba(x_train)[:, 1]
    
    signature = infer_signature(x_val, test_hat)

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(best_params)
        mlflow.log_param("total_samples_size", x_train.shape[0])

        # Log metrics for validation set
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("precision", precision_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("recall", recall_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, test_probs))

        # Log metrics for training set
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_precision", precision_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_recall", recall_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, train_probs))
        mlflow.log_param("loss_function", loss_function)
        mlflow.log_param("total_samples_size", total_samples_size)
       

        # Log the Random Forest model
        mlflow.sklearn.log_model(sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model")

        # Plot train confusion matrix
        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        # Plot test confusion matrix
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")
        # Delete the confusion matrix files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        # Log additional artifacts
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id

def load_model_by_name(model_name: str):
    """Load a pre-trained model from an MLflow server"""
    config_mlflow()
    client = mlflow.MlflowClient()
    registered_model = client.get_registered_model(model_name)
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model