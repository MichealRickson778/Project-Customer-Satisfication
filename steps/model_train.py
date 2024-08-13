import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                config: ModelNameConfig) -> RegressorMixin:
    try:
        model = None
        if config.model_name == "LinearRegressionModel":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(x_train, y_train)
            return train_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))

    except Exception as e:
        logging.error("Error in training model: {}".format(config.model_name))
        raise e




