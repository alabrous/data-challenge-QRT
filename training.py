from preprocess import preprocess_main
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

def train_model(X_train, y_train):
    """
    Train a Random Survival Forest model on the training data.
    """
    model = RandomSurvivalForest(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using concordance index.
    """
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_YEARS'], model.predict(X_test))
    return c_index[0]

def main():
    X_train, X_test, y_train, y_test = preprocess_main('X_train/clinical_train.csv','target_train.csv')
    model = train_model(X_train, y_train)
    c_index = evaluate_model(model, X_test, y_test)
    print(f"Concordance Index: {c_index:.2f}")


if __name__ == "__main__":
    main()