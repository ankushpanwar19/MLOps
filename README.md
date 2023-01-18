MLOps
==============================

To create MLOps pipeline for neural machine learning project using tool such as mlflow, dvc etc

Environment creation
------------
    conda create -n mlops python=3.8 
    conda activate mlops
    pip install -r requirements.txt

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

To create a similar project structure, use below commands:


    pip install cookiecutter
    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

## ML flow tracking:
The MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. MLflow Tracking lets you log and query experiments. MLflow tracking tutorial.

Example : You can check the MLflow tracking in src/train.py. or You can refer [MLflow tracking tutorial](https://link-url-here.org)

     with mlflow.start_run() as run:
        <Training code>
        
        # Track Parameters
        mlflow.log_param("learning_rate", args.lr)

        #  Track metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Track model
        mlflow.pytorch.log_model(model, "classifier")

        # Save model artifact
        model_save_dir = "models/mnist_cnn.pt"
        torch.save(model.state_dict(), model_save_dir)
        mlflow.log_artifact(model_save_dir)

Once you run the `python src/train.py` file, MLflow tracking will create mlruns folder. MLflow organises ML tracking into experiments and runs. An Experiment (default:0) is created to deal with a particular task and multiple runs (named:run_id) are performed under experiment.

mlruns folder structure:

    ├── <Experiment_id>
        ├── <run_id>
            ├── artifacts   <- Stores models,features, enivorment file etc
            ├── metrics     <- Stores metric like accuracy
            ├── params      <- Stores hyperparameters
            ├── tags        <- contains mlflow run info, github commit hash etc  
            ├── meta.yaml   <- run info

