from dataload import*
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.preprocessing import StandardScaler as stdscaler, scale
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.base import clone
import matplotlib.pyplot as plt

def save_process():
    downloader = downLoader()
    mnist = downloader.download()
    x, y = mnist["data"], mnist["target"]
    datasaver = dataSaver(x, y)
    datasaver.whichOS("isWindows")

def div_data(x, y):
    y = y.astype(np.uint8)
    X_train, X_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    return X_train, X_test, y_train, y_test, y_train_5, y_test_5

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g--", label="재현율")
    print(thresholds)
    plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

if __name__ == "__main__":
    # save_process() # execute this function first time. This function download mnist file on your space. Then you should comment out this line.
    dataloader = dataLoader()
    data = dataloader.whichOS("isWindows")
    X_train, X_test, y_train, y_test, y_train_5, y_test_5 = div_data(data[0], data[1])
    scaler = stdscaler()
    some_digit = X_train[0] 

    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    sgd_clf = sgd(random_state=42)
    clone_sgd = clone(sgd_clf)

    sgd_clf.fit(X_train_scaled, y_train)
    print(sgd_clf.predict([some_digit]))

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    print("precision score: {}, recall score: {}, f1 score: {}".format(
        precision_score(y_train, y_train_pred, average='weighted'), recall_score(y_train, y_train_pred, average='weighted'), f1_score(y_train, y_train_pred, average='weighted')))
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)

    sgd_clf.fit(X_train_scaled, y_train_5)
    y_scores = cross_val_predict(sgd_clf, X_train_scaled, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    plot_roc_curve(fpr, tpr)
    print("AUC: {}".format(roc_auc_score(y_train_5, y_scores)))