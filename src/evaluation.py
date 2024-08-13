import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

import numpy as np


class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in Calcluating mse: {}".format(e))
            raise e

class R2_score(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Claculating R2_Score")
            r2score = r2_score(y_true, y_pred)
            logging.info("R2_Score: {}".format(r2score))
            return r2score
        except Exception as e:
            logging.error("Error in Calculating r2_score: {}".format(e))
            raise e

class RMSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmserror =root_mean_squared_error(y_true, y_pred)
            logging.info("RMSE: {}".format(rmserror))
            return rmserror
        except Exception as e:
            logging.error("Error in Calculating RMSE: {}".format(e))
            raise e



