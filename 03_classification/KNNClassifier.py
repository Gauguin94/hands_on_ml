from dataload import*
from plot_digits_ import*
from sklearn.neighbors import KNeighborsClassifier as knn
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

def make_noise(x_train, x_test):
    noise = np.random.randint(0, 100, (len(x_train), 784))
    X_train_mod = x_train + noise
    noise = np.random.randint(0, 100, (len(x_test), 784))
    X_test_mod = x_test + noise
    y_train_mod = x_train
    y_test_mod = x_test

    some_digit_noise = X_test_mod[0]
    some_digit_noise_image = some_digit_noise.reshape(28, 28)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1).set_title('Plus Noise')
    plt.imshow(some_digit_noise_image, cmap="binary")
    plt.axis("off")

    some_digit_noise = y_test_mod[0]
    some_digit_noise_image = some_digit_noise.reshape(28, 28)
    fig.add_subplot(1, 2, 2).set_title('Origin')
    plt.imshow(some_digit_noise_image, cmap="binary")
    plt.axis("off")
    plt.show()

    return X_train_mod, y_train_mod, X_test_mod, y_test_mod

if __name__ == "__main__":
    # save_process() # execute this function first time. This function download mnist file on your space. Then you should comment out this line.
    dataloader = dataLoader()
    data = dataloader.whichOS("isWindows")
    X_train, X_test, y_train, y_test, y_train_5, y_test_5 = div_data(data[0], data[1])
    X_train_mod, y_train_mod, X_test_mod, y_test_mod = make_noise(X_train, X_test)
    
    knn_clf = knn()
    some_index = 0
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digits(clean_digit)