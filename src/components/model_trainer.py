import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "poisson", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.3]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ## To get best model score from model_report dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from model_report dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            logging.info("Best model found for both training and testing datasets.")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score



        except Exception as e:
            raise CustomException(e,sys)