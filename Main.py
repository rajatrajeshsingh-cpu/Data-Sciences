# Deloitte AI & Machine Learning Assessment
# Candidate: Rajat Rajesh Singh

from data_processing import load_data, preprocess_data
from ml_model import train_model, make_prediction

def main():

    # Step 1: Load dataset
    X, y = load_data()

    # Step 2: Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train ML model
    model = train_model(X_train, y_train)

    # Step 4: Make prediction
    prediction = make_prediction(model, [[6]])

    print("Prediction for input value 6:", prediction)

if __name__ == "__main__":
    main()