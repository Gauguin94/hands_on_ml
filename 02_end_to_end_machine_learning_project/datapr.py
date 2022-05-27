from dataclasses import replace
import numpy as np
import pandas as pd
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class dataPreprocessing(CombinedAttributesAdder):
    def __init__(self, train_data):
        super().__init__()
        self.tr = train_data
        self.num_attribs = list(train_data.drop("ocean_proximity", axis=1))
        self.cat_attribs = ["ocean_proximity"]

    def consist_of_num(self):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
        return num_pipeline

    def num_combine_with_str(self, num_pipeline):
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, self.num_attribs),
            ("cat", OneHotEncoder(), self.cat_attribs)
        ])
        return full_pipeline

    def make_pipe(self):
        num_pipeline = self.consist_of_num()
        full_pipeline = self.num_combine_with_str(num_pipeline)
        #return full_pipeline.fit_transform(self.tr)
        return full_pipeline

class dataSplit():
    def __init__(self):
        pass

    def income_cat(self, data):
        data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1,2,3,4,5])
        return data

    def del_income_cat(self, tr_set, tt_set):
        for phase in ["train", "test"]:
            if phase == "train":
                tr_set.drop("income_cat", axis=1, inplace=True)
            else:
                tt_set.drop("income_cat", axis=1, inplace=True)
        return tr_set, tt_set

    def stratified_sampling(self, data):
        data = self.income_cat(data)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data, data["income_cat"]):
            strat_train_set = data.loc[train_index]
            strat_test_set = data.loc[test_index]
        return self.del_income_cat(strat_train_set, strat_test_set)


# class dataPreprocessing:
#     def __init__(self):
#         pass

#     def div_by_income_cat(self, data):
#         data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1,2,3,4,5])
#         return data

#     def del_income_cat(self, tr_set, tt_set):
#         for phase in ["train", "test"]:
#             if phase == "train":
#                 tr_set.drop("income_cat", axis=1, inplace=True)
#             else:
#                 tt_set.drop("income_cat", axis=1, inplace=True)
#         return tr_set, tt_set

#     def make_feature(self, tr, tt):
#         tr["rooms_per_household"] = tr["total_rooms"]/tr["households"]
#         tr["bedrooms_per_room"] = tr["total_bedrooms"]/tr["total_rooms"]
#         tr["population_per_household"] = tr["population"]/tr["households"]
#         tt["rooms_per_household"] = tt["total_rooms"]/tt["households"]
#         tt["bedrooms_per_room"] = tt["total_bedrooms"]/tt["total_rooms"]
#         tt["population_per_household"] = tt["population"]/tt["households"]
#         return tr, tt

#     def replace_null(self, tr, tt, opt): # check the "KNNImputer"
#         imputer = SimpleImputer(strategy=opt)
#         train_set = tr.drop("ocean_proximity", axis=1) # if opt is "median", imputer can't calculate median that made of string.
#         imputer.fit(train_set) # return value initialized on "imputer.statistics_". if you print "imputer.statistics_", you can show return value.
#         test_set = tt.drop("ocean_proximity", axis=1)
#         imputer.fit(test_set)
#         tr = imputer.transform(train_set) # replace "Null" to new value, but return type is "NumPy".        
#         tt = imputer.transform(test_set)
#         tr = pd.DataFrame(tr, columns=train_set.columns, index=train_set.index)
#         tt = pd.DataFrame(tt, columns=test_set.columns, index=test_set.index)
#         return tr, tt

#     def one_hot_encoder(self, tr, tt):
#         encoder = OneHotEncoder()
#         tr_str = tr[["ocean_proximity"]]
#         tt_str = tt[["ocean_proximity"]]
#         tr_to_1hot = encoder.fit_transform(tr_str)
#         tt_to_1hot = encoder.fit_transform(tt_str)
#         return tr_to_1hot, tt_to_1hot

    # def replace_null(self, tr, tt, opt):
    #     if opt == 0: # delete Null data
    #         tr = tr.dropna(subset=["total_bedrooms"])
    #         tt = tt.dropna(subset=["total_bedrooms"])
    #     elif opt == 1: # delete category that naming "total_bedrooms"
    #         tr = tr.drop("total_bedrooms", axis=1)
    #         tt = tt.drop("total_bedrooms", axis=1)
    #     else: # fullfill with median in specific category
    #         tr_median = tr["total_bedrooms"].median()
    #         tt_median = tt["total_bedrooms"].median()
    #         tr = tr["total_bedrooms"].fillna(tr_median, inplace=True)
    #         tt = tr["total_bedrooms"].fillna(tt_median, inplace=True)
    #     return tr, tt

    # def test_set_check(self, identifier, test_ratio): # return bool by checking crc32
    #     return crc32(np.int64(identifier)) & 0xfffffffff < test_ratio*(2**32)

    # def split_train_test(self, data, test_ratio): # using none skill
    #     shuffled_indices = np.random.permutation(len(data))
    #     test_set_size = int(len(data)*test_ratio)
    #     test_indices = shuffled_indices[:test_set_size]
    #     train_indices = shuffled_indices[test_set_size:]
    #     return data.iloc[train_indices], data[test_indices]

    # def split_train_test_by_id(self, data, test_ratio, id_column):
    #     ids = data[id_column]
    #     in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio))
    #     return data.loc[~in_test_set], data.loc[in_test_set]

    # def split_by_random_state(self, data): # "random_state = 42" => Deep thought's number in "The Hitchhiker's Guide to the Galaxy"
    #     train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    #     return train_set, test_set