import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class model(ABC):

    @abstractmethod
    def train(self, x_train, y_train):
        pass

class LinearRegressionModel(model):
    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("model training completed")
            return reg
        except Exception as e:
            logging.error("Error in Training Model")
            raise e

