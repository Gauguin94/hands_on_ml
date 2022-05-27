from datapr import*
from dataload import*
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

def display_scores(scores):
    print("score:", scores)
    print("mean:", scores.mean())
    print("standard deviation:", scores.std())


if __name__ == "__main__":
    dataloader = dataLoader()
    dataspliter = dataSplit()
    housing = dataloader.whichOS("isWindows")
    train_data, test_data = dataspliter.stratified_sampling(housing)

    label = train_data["median_house_value"].copy()
    train_data = train_data.drop("median_house_value", axis=1)
    datapreprocessor = dataPreprocessing(train_data)
    train_pipe = datapreprocessor.make_pipe()
    x = train_pipe.fit_transform(train_data)

    dtr = DecisionTreeRegressor()
    dtr.fit(x, label)

    scores = cross_val_score(dtr, x, label, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)
    joblib.dump(dtr, "decision_tree_model.pkl")