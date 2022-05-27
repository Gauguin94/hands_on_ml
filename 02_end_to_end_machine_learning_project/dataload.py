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
        dirpath = os.path.dirname(__file__)+"\datasets\housing"
        housing = pd.read_csv("{}\housing.csv".format(dirpath), thousands=',')
        return housing

    def isLinux(self):
        dirpath = os.path.dirname(__file__)+"/datasets/housing"
        housing = pd.read_csv("{}/housing.csv".format(dirpath), thousands=',')
        return housing