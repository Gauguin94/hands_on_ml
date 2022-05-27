from datapr import*
from dataload import*
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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

    rfr = RandomForestRegressor()
    rfr.fit(x, label)

    scores = cross_val_score(rfr, x, label, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    display_scores(forest_rmse_scores)

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)
    grid_search.fit(x, label)
    grid_search.best_params_    


    joblib.dump(rfr, "forest_model.pkl")