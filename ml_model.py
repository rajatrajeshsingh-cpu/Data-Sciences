from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def make_prediction(model, new_input):

    prediction = model.predict(new_input)

    return prediction