import numpy as np
from sklearn.model_selection import train_test_split

def load_data():

    # Sample dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    return X, y


def preprocess_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test