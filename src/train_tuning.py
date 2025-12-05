# for saving models
import pickle
import json
import os
import joblib

# helpers for sklearn wrapper
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance

# models used
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# other helpers
import pandas as pd
import numpy as np
import seaborn as sb
sb.set()


SEED = 42


class ModelError(Exception):
    pass

class Model:
    def __init__(self, is_binary, code="N/A"):
        self.code = code
        self.is_binary = is_binary

    def _convert_model_name(self):
        if self.code == "logit":
            self.name = "Logistic Regression Model"
        elif self.code == "svm":
            self.name = "Support Vector Classification Model"
        elif self.code == "rf":
            self.name = "Random Forest Classifier Model (Bagging)"
        elif self.code == "xgb":
            self.name = "Extreme Gradient Boosting (XGBoost) Model"
        elif self.code == "lgb":
            self.name = "Light Gradient Boosting Machine (LightGBM) Model"
        elif self.code == "cat":
            self.name = "CatBoost Model"
    
    def create_model(self, model_name, **kwargs):
        if model_name == "logit":
            self.hparams = {
                "penalty": kwargs.get("penalty", "l2"),
                "C": kwargs.get("C", 1.0)
            }
            if self.hparams["penalty"] == "elasticnet":
                self.hparams["l1_ratio"] = kwargs.get("l1_ratio", 0.5)
            self._model = LogisticRegression(**self.hparams)
            
        elif model_name == "svm":
            self.hparams = {
                "C": kwargs.get("C", 1.0),
                "kernel": kwargs.get("kernel", "rbf"),
                "gamma": kwargs.get("gamma", "scale")
            }
            if self.hparams["kernel"] == "poly":
                self.hparams["degree"] = kwargs.get("degree", 3)
            self._model = SVC(**self.hparams, probability = True)

        elif model_name == "rf":
            self.hparams = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "min_samples_split": kwargs.get("min_samples_split", 2),
                "max_features": kwargs.get("max_features", "sqrt"),
                "max_samples": kwargs.get("subsample", 1)
            }
            self._model = RandomForestClassifier(**self.hparams)

        elif model_name == "xgb":
            self.hparams = {
                "num_parallel_tree": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.3),
                "subsample": kwargs.get("subsample", 1),
                "colsample_bytree": kwargs.get("colsample_bytree", 1)
            }
            if self.is_binary:
                self._model = XGBClassifier(tree_method = "hist", **self.hparams)
            else:
                self._model = XGBClassifier(objective = "multi:softprob", tree_method = "hist", **self.hparams)

        elif model_name == "lgb":
            self.hparams = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.3),
                "subsample": kwargs.get("subsample", 1),
                "colsample_bytree": kwargs.get("colsample_bytree", 1)
            }
            self._model = LGBMClassifier(**self.hparams)

        elif model_name == "cat":
            self.hparams = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.3),
                "subsample": kwargs.get("subsample", 1)
            }
            self._model = CatBoostClassifier(bootstrap_type = "Bernoulli", **self.hparams)

        self.code = model_name
        self._convert_model_name()

    def fit(self, X, y):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")
        self._model.fit(X, y)

    def predict(self, X):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")
        return self._model.predict(X)

    def predict_proba(self, X):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")
        return self._model.predict_proba(X)

    def score(self, X ,y):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")
        return self._model.score(X, y)

    def save_model(self, model=None, file_name="model"):
        if model is None:
            if not hasattr(self, "_model"):
                raise ModelError("Please create/load a model first!")
            model = self._model

        if file_name:
            file_name = "-"+file_name
        if self.is_binary:
            file_path = f"models/binary/{self.code}{file_name}.pkl"
        else:
            file_path = f"models/multi/{self.code}{file_name}.pkl"
        
        try:
            joblib.dump(model, file_path)
        except Exception as e:
            print(f"Error saving model with joblib: {e}")
            raise

    def load_model(self, file_name):
        model_path = file_name
        parts = file_name.split("-")
        self.code = parts[0]
        self._convert_model_name()

        try:
            self._model = joblib.load(model_path)
            return self._model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model using joblib: {e}")

    def k_fold_cv(self, X, y, k = 5):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")
        
        scores = cross_val_score(self._model, X, y, cv = k)
        print(f"{k}-fold Cross Validation Scores:\t{scores}")
        print(f"{k}-fold Cross Validation Average Score:\t{np.mean(scores)}")
        return scores

    def grid_search(self, X, y, param_grid, verbose = 3):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")

        gs = GridSearchCV(self._model, param_grid, refit = True, verbose = verbose)
        gs.fit(X, y)
        print(f"Model type:\t{self.name}")
        print(f"Best score:\t{gs.best_score_}")
        print(f"Best parameters:\t{gs.best_params_}")
        print(f"Best estimator:\t{gs.best_estimator_}")
        return gs

    def get_feature_importances(self, **kwargs):
        if not hasattr(self, "_model"):
            raise ModelError("Please create/load a model first!")

        if not hasattr(self._model, "feature_names_in_"):
            feature_names = [f"feature_{n+1}" for n in range(self._model.n_features_in_)]
        else:
            feature_names = self._model.feature_names_in_

        if any(self.code in code for code in ["logit", "svm"]):
            if self.code == "svm" and self.hparams["kernel"] != "linear":
                if not ("X_test" in kwargs and "y_test" in kwargs):
                    raise ModelError("For SVM that uses a non-linear kernel, please pass in X_test and y_test parameters for feature importance!")
                fi = dict(zip(feature_names, permutation_importance(self._model, kwargs.get("X_test"), kwargs.get("y_test"), n_repeats=10, random_state=42).importances_mean))
            else:
                fi = dict(zip(feature_names, self._model.coef_[0].tolist()))
        else:
            fi = dict(zip(feature_names, self._model.feature_importances_.tolist()))
        
        fi_sorted = {k: v for k, v in sorted(fi.items(), key=lambda item: abs(item[1]), reverse=True)}

        return fi_sorted