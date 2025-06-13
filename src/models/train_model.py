from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, path):
    joblib.dump(model, path)
