import pandas as pd
import os

class dataLoader:
    def __init__(self):
        pass

    def whichOS(self, arg):
        self.case_name = str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()

    def isWindows(self):
        dirpath = os.path.dirname(__file__)+"\datasets\lifesat"
        oecd_bli = pd.read_csv("{}\oecd_bli_2015.csv".format(dirpath), thousands=',')
        gdp_per_capita = pd.read_csv("{}\gdp_per_capita.csv".format(dirpath), thousands=',',\
            delimiter='\t', encoding='latin1', na_values="n/a")
        return oecd_bli, gdp_per_capita

    def isLinux(self):
        dirpath = os.path.dirname(__file__)+"/datasets/lifesat"
        oecd_bli = pd.read_csv("{}/oecd_bli_2015.csv".format(dirpath), thousands=',')
        gdp_per_capita = pd.read_csv("{}/gdp_per_capita.csv".format(dirpath), thousands=',',\
            delimiter='\t', encoding='latin1', na_values="n/a")
        return oecd_bli, gdp_per_capita
