# This file gonna combine two dataFrame on file, "oecd_bli_2015.csv" and "gdp_per_capita.csv".

import pandas as pd

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"] # extract data that labeled "TOT" on "Inquality". "oecd_bli_2015.csv"
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value") # abandon useless data on label and make new vector, row: country, column: indicator
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True) # change label name "2015" to "GDP per capita".
    gdp_per_capita.set_index("Country", inplace=True) # abandon useless data, like header in this file.
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, # combine two vector.
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True) # sort new vector by ascending.
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices)) # abandon useless data.
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
