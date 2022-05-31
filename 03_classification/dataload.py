import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

class downLoader:
    def __init__(self):
        pass

    def download(self):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        print(mnist.keys())
        return mnist

class dataSaver:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def whichOS(self, arg):
        self.case_name = str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()

    def isWindows(self):
        dirpath = os.path.dirname(__file__)+r"\datasets"
        np.save(dirpath + r"\x", self.x)
        np.save(dirpath + r"\y", self.y)        
        print("save process finish! (OS: Windows)")
        return 0

    def isLinux(self):
        dirpath = os.path.dirname(__file__)+"/datasets"
        np.save(dirpath+"/x", self.x)
        np.save(dirpath+"/y", self.y)        
        print("save process finish! (OS: Linux)")
        return 0

class dataLoader:
    def __init__(self):
        pass

    def whichOS(self, arg):
        self.case_name = str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()

    def isWindows(self):
        dirpath = os.path.dirname(__file__)+r"\datasets"
        x = np.load(dirpath + r"\x.npy", allow_pickle=True)
        y = np.load(dirpath + r"\y.npy", allow_pickle=True)
        return x, y

    def isLinux(self):
        dirpath = os.path.dirname(__file__)+"/datasets"
        x = np.load(dirpath+"/x.npy", allow_pickle=True)
        y = np.load(dirpath+"/y.npy", allow_pickle=True)
        return x, y