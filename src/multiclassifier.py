import copy

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
import os
import joblib
import numpy as np

class ModelError(Exception):
    pass

class OWMultiClassifier():

    def __init__(self, binary_classifier = None, multi_classifier = None):
        self._binary_classifier = binary_classifier
        self._multi_classifier = multi_classifier

    def set_binary_classifier(self, binary_classifier):
        self._binary_classifier = binary_classifier

    def set_multi_classifier(self, multi_classifier):
        self._multi_classifier = multi_classifier

    def fit(self, X, y):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        if not isinstance(y, pd.DataFrame) or (not all(col in y.columns for col in ["y_true", "y_binary"])):
            raise TypeError("y must be a pandas DataFrame with columns 'y_true' and 'y_binary'!")
        print("Fitting binary classifier...")
        self._binary_classifier.fit(X, y[["y_binary"]])
        print("Binary classifier trained!")
        print("Fitting multi classifier...")
        X_closed = X.loc[y["y_binary"] != -1]
        y_closed = y.loc[y["y_binary"] != -1]
        self._multi_classifier.fit(X_closed, y_closed[["y_true"]])
        print("Multi classifier trained!")

    def predict(self, X):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        open_pred = self._binary_classifier.predict(X)
        open_pred = pd.DataFrame({"sample": X.index.tolist(), "y_binary": open_pred.tolist()})

        X_mon = X.loc[open_pred["y_binary"] != -1]
        closed_pred = self._multi_classifier.predict(X_mon)
        closed_pred = pd.DataFrame({"sample": X_mon.index.tolist(), "y_true": closed_pred.tolist()})

        open_pred = open_pred[~open_pred["sample"].isin(closed_pred["sample"])]
        open_pred.set_index("sample", inplace=True)
        open_pred.columns = ["y_true"]
        closed_pred.set_index("sample", inplace=True)

        res = pd.concat([open_pred, closed_pred]).sort_index()["y_true"].values
        for i, v in enumerate(res):
            if type(v) is list:
                res[i] = v[0]

        return res.astype(int)

    def predict_proba(self, X):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        open_pred = self._binary_classifier.predict_proba(X)
        open_pred = pd.DataFrame({"sample": X.index.tolist(), "y_binary": open_pred.tolist()})

        X_mon = X.loc[open_pred["y_binary"] != -1]
        closed_pred = self._multi_classifier.predict_proba(X_mon)
        closed_pred = pd.DataFrame({"sample": X_mon.index.tolist(), "y_true": closed_pred.tolist()})

        open_pred = open_pred[~open_pred["sample"].isin(closed_pred["sample"])]
        open_pred.set_index("sample", inplace=True)
        open_pred.columns = ["y_true"]
        closed_pred.set_index("sample", inplace=True)

        res = pd.concat([open_pred, closed_pred]).sort_index()["y_true"].values
        for i, v in enumerate(res):
            if type(v) is list:
                res[i] = v[0]

        return res.astype(int)

    def score(self, X, y):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        y_true = y["y_true"].values.astype(int)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def k_fold_cv(self, X, y, k = 5):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        copy_binary_classifier = copy.deepcopy(self._binary_classifier)
        copy_multi_classifier = copy.deepcopy(self._multi_classifier)

        kf = KFold(n_splits=k, shuffle=True)
        print(f"Conducting K-Fold CV on {k} folds...")
        accuracies = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {i+1}...")
            print(test_index)
            self.fit(X.iloc[train_index].reset_index(), y.iloc[train_index].reset_index())
            score = self.score(X.iloc[test_index].reset_index(), y.iloc[test_index].reset_index())
            accuracies.append(score)
            print(f"Fold {i+1} accuracy: {score}")

        print()
        print(f"Mean accuracy on K-Fold CV on {k} folds: {np.mean(accuracies)}")
        self._binary_classifier = copy_binary_classifier
        self._multi_classifier = copy_multi_classifier
        return accuracies


    def save(self, file_name="ow_multiclassifier"):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        os.makedirs("models", exist_ok=True)
        folder_name = f"models/{file_name}"
        os.makedirs(folder_name, exist_ok=True)

        try:
            joblib.dump(self._binary_classifier, f"{folder_name}/binary.pkl")
            joblib.dump(self._multi_classifier, f"{folder_name}/multi.pkl")
            print("Classifier saved!")
        except Exception as e:
            print(f"Error saving model with joblib: {e}")
            raise

    def load(self, file_name):
        model_path = os.path.join("models", file_name)

        try:
            self._binary_classifier = joblib.load(os.path.join(model_path, "binary.pkl"))
            self._multi_classifier = joblib.load(os.path.join(model_path, "multi.pkl"))
            print("Classifier loaded!")
        except FileNotFoundError:
            raise FileNotFoundError(f"Classifier folder not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model using joblib: {e}")