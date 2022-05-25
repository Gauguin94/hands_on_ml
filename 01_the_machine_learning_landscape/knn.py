import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors
from dataload import*
from datapr import*

if __name__ == "__main__":
    dataloader = dataLoader()
    data = dataloader.whichOS("isWindows") # work on Linux, u should type "isLinux"
    oecd_bli = data[0]
    gdp_per_capita = data[1]

    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]] # X_data( ex) input) => GDP per capita
    y = np.c_[country_stats["Life satisfaction"]] # y_data( ex) answer) => Life satisfaction

    country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
    plt.show()

    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3) # choose model, model's criterion based on the nearest 3 vector.

    model.fit(X, y) # train model

    X_new = [[22587]] # make new data for criticizing performance(inference). In this case, X_new is Republic of Cyprus's GDP.
    print('')
    print('Republic of Cyprus\'s Life satisfaction: {}'.format(model.predict(X_new)[0][0])) # model's predict. -> 5.96242338 (Life satisfaction)
    print('')
