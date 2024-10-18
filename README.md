# Network Anomaly Detection
==============================

A network anomaly project implemented following the cookiecutter template employing MLOps best practices.

## Data Set


Instructions
------------
1. Clone the repo: `git clone https://github.com/Danselem/network_anomaly.git`.
2. Run `cd network_anomaly` to move into the directory.
3. *Optional:* Run `make virtualenv` to create a python virtual environment. Skip if using conda or some other env manager.
    1. Run `source .venv/bin/activate` to activate the virtual environment.
4. Run `make requirements` to install required python packages.
5. Put the raw data in `data/raw` with the command `python -m src.data.ingest_data`.
6. To save the raw data to the DVC cache, run `dvc add data/raw`
7. To preprocess the data, run `python -m src.data.clean_data`.
8. Process your data, train and evaluate your model using `dvc repro` or `make reproduce`
9. To run the pre-commit hooks, run `make pre-commit-install`
10. For setting up data validation tests, run `make setup-setup-data-validation`
11. For **running** the data validation tests, run `make run-data-validation`
12. When you're happy with the result, commit files (including .dvc files) to git.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── metrics.txt    <- Relevant metrics after evaluating the model.
    │   └── training_metrics.txt    <- Relevant metrics from training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── great_expectations  <- Folder containing data integrity check files
    │   │   ├── make_dataset.py
    │   │   └── data_validation.py  <- Script to run data integrity checks
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── .pre-commit-config.yaml  <- pre-commit hooks file with selected hooks for the projects.
    ├── dvc.lock           <- constructs the ML pipeline with defined stages.
    └── dvc.yaml           <- Traing a model on the processed data.


--------


Made with 🐶 by [DAGsHub](https://dagshub.com/).
