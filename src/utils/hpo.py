"""Modeling Optimization with Hyperopt"""
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

seed = 1024


def classification_objective(
    x_train: ArrayLike,
    y_train: ArrayLike,
    model_family: str,
    loss_function: str,
    params: dict,
) -> dict:
    """Trainable function for classification models.

    Args:
        x_train (ArrayLike): Training features
        y_train (ArrayLike): Training target
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
        params: Hyperparameter dictionary for the given model.

    Returns:
        A dictionary containing the metrics from training.
    """

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )

    if model_family in ['xgboost', 'random_forest']:
        if model_family == 'xgboost':
            model = XGBClassifier(**params, enable_categorical=True)
        else:
            # Create Random Forest model
            model = RandomForestClassifier(
                **params,
            )

        model.fit(x_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(x_val)

    else:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. "
            "Supported families are 'xgboost', and 'random_forest'."
        )

    # Calculate the loss
    if loss_function == 'F1':
        loss = 1 - f1_score(y_val, y_pred, pos_label=1)
    elif loss_function == 'Accuracy':
        loss = 1 - accuracy_score(y_val, y_pred)
    elif loss_function == 'Precision':
        loss = 1 - precision_score(y_val, y_pred, pos_label=1)

    return {'loss': loss, 'status': STATUS_OK}


def classification_optimization(
    x_train: ArrayLike,
    y_train: ArrayLike,
    model_family: str,
    loss_function: str,
    objective_function: str,
    num_trials: int,
    diagnostic: bool = False,
) -> dict:
    """Optimize hyperparameters for a model using Hyperopt."""

    if model_family == 'random_forest':
        max_feature = ['sqrt', 'log2', None]
        bstrap = [True, False]
        criterion = ['gini', 'entropy', 'log_loss']
        # cweight = ['balanced', 'balanced_subsample']
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 100, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
            'max_features': hp.choice('max_features', max_feature),
            'bootstrap': hp.choice('bootstrap', bstrap),
            'criterion': hp.choice('criterion', criterion),
            # 'class_weight': hp.choice('class_weight', cweight)
        }

    elif model_family == 'xgboost':
        search_space = {
            'max_depth': hp.choice('max_depth', np.arange(1, 20, 1, dtype=int)),
            'eta': hp.uniform('eta', 0, 1),
            'gamma': hp.uniform('gamma', 0, 100),  # Adjusted upper bound for clarity
            'reg_alpha': hp.uniform('reg_alpha', 1e-7, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'n_estimators': hp.choice(
                'n_estimators', np.arange(10, 1000, 10, dtype=int)
            ),
            'learning_rate': hp.quniform(
                'learning_rate', 0.001, 0.3, 0.01
            ),  # Will need to convert to float
            'min_child_weight': hp.choice(
                'min_child_weight', np.arange(1, 10, 1, dtype=int)
            ),
            'max_delta_step': hp.choice(
                'max_delta_step', np.arange(0, 10, 1, dtype=int)
            ),  # Adjusted range
            'subsample': hp.uniform('subsample', 0.5, 1),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': seed,
        }

    else:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. Supported families are 'random_forest', 'catboost', and 'xgboost'."
        )

    rstate = np.random.default_rng(seed)
    trials = Trials()
    best_params = fmin(
        fn=lambda params: classification_objective(
            x_train, y_train, model_family, loss_function, params
        ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=trials,
        rstate=rstate,
    )

    if model_family == 'random_forest':
        best_params['bootstrap'] = bstrap[best_params['bootstrap']]['bootstrap']

        best_params['max_features'] = max_feature[best_params['max_features']][
            'max_features'
        ]

        best_params['criterion'] = criterion[best_params['criterion']]['criterion']

        # best_params['class_weight'] = cweight[
        #     best_params['class_weight']]['class_weight']

        # Convert specified parameters to integers if they are present
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")

    elif model_family == 'xgboost':
        # Convert and handle parameters for xgboost
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['eta'] = float(best_params['eta'])
        best_params['gamma'] = float(best_params['gamma'])
        best_params['reg_alpha'] = float(best_params['reg_alpha'])
        best_params['reg_lambda'] = float(best_params['reg_lambda'])
        best_params['colsample_bytree'] = float(best_params['colsample_bytree'])
        best_params['colsample_bynode'] = float(best_params['colsample_bynode'])
        best_params['colsample_bylevel'] = float(best_params['colsample_bylevel'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['learning_rate'] = float(best_params['learning_rate'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['max_delta_step'] = int(best_params['max_delta_step'])
        best_params['subsample'] = float(best_params['subsample'])
        # best_params['objective'] = best_params.get('objective', 'binary:logistic')
        best_params['eval_metric'] = best_params.get('eval_metric', 'aucpr')

        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")

    return best_params
