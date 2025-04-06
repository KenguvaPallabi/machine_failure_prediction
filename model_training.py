import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Save model
    joblib.dump(model, "models/model.pkl")

    # Save evaluation
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    with open("results/evaluation_report.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))
