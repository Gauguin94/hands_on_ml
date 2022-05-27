from dataload import*
from datapr import*
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def showMe_income(data):
    plt.hist(data)
    plt.show()

def showMe_polar(tr, tt):
    plt.scatter(tr["longitude"], tr["latitude"], alpha=0.4, s=tr["population"]/100, label="population", c=tr["median_house_value"], cmap=plt.get_cmap("jet"))
    plt.colorbar()
    plt.legend()
    #plt.scatter(tt["longitude"], tt["latitude"], alpha=0.4, s=tt["population"], c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    plt.show()

def pearson_corr(tr, tt):
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    corr_train = tr.corr()
    corr_test = tt.corr()
    tr_about_median_house_value = corr_train["median_house_value"].sort_values(ascending=False)
    tt_about_median_house_value = corr_test["median_house_value"].sort_values(ascending=False)
    print(tr_about_median_house_value)
    print(tt_about_median_house_value)
    # scatter_matrix(housing[attributes],figsize=(12, 8))
    plt.scatter(tr["median_income"], tr["median_house_value"], alpha=0.1)
    plt.show()

def pearson_corr_edited(tr, tt):
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    corr_train = tr.corr()
    corr_test = tt.corr()
    tr_about_median_house_value = corr_train["median_house_value"].sort_values(ascending=False)
    tt_about_median_house_value = corr_test["median_house_value"].sort_values(ascending=False)
    print(tr_about_median_house_value)
    print(tt_about_median_house_value)
    # scatter_matrix(housing[attributes],figsize=(12, 8))
    plt.scatter(tr["median_income"], tr["median_house_value"], alpha=0.1)
    plt.show()

if __name__ == "__main__":
    dataloader = dataLoader()
    dataspliter = dataSplit()
    datapreprocessor = dataPreprocessing()
    housing = dataloader.whichOS("isWindows")
    income_data = datapreprocessor.div_by_income_cat(housing)
    #showMe_income(income_data["income_cat"])
    train_set, test_set = dataspliter.stratified_sampling(income_data)
    train_set, test_set = datapreprocessor.del_income_cat(train_set, test_set)
    #showMe_polar(train_set, test_set)
    #pearson_corr(train_set, test_set)
    train_set_edit, test_set_edit = datapreprocessor.make_feature(train_set, test_set)
    #pearson_corr_edited(train_set, test_set)
    train_set, test_set = datapreprocessor.replace_null(train_set, test_set, opt="median")
    train_set_str, test_set_str = datapreprocessor.one_hot_encoder(train_set_edit, test_set_edit)
    print(train_set_str.toarray())
    print(test_set_str.toarray())