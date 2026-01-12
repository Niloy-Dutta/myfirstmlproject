import os
import sys

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
        def __init__(self): 
              self.model_trainer_config = ModelTrainerConfig()
        
        def initiate_model_trainer(self,train_array,test_array): 
              try: 
                    logging.info("Splitting Training and Test Data")
                    X_train,y_train,X_test,y_test = (
                          train_array[:,:-1],
                          train_array[:,-1],
                          test_array[:,:-1],
                          test_array[:,-1]
                          
                    )
                    models = {
                        "Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Linear Regression": LinearRegression(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor(),
            }
                    model_report:dict = evaluate_model(X_train= X_train,y_train= y_train,
                                                       X_test = X_test,y_test = y_test,
                                                       models = models)
                    
                    # To get best name of a model 
                    best_model_name = max(model_report,key = model_report.get)
                    #To get best model name from a dict
                    best_model_score = model_report[best_model_name]

                    best_model = models[best_model_name]
                    if best_model_score <.60:
                          logging.info("No best model found")
                    logging.info('Best found model on both training and testing dataset')
                    
                    save_object(
                          file_path=self.model_trainer_config.trained_model_file_path,
                          obj=best_model
                    )
                    logging.info("Saved the model trianer as pickle file")
                    predicted = best_model.predict(X_test)
                    r_square = r2_score(y_test,predicted)

                    return r_square

              
              except Exception as e: 
                    raise CustomException(e,sys)