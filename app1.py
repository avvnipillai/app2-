from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

@app.route('/')
def home():
    return "Naive Bayes API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Expect CSV file upload
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    # Example: last column as target
    target_col = df.columns[-1]
    X = df[df.columns[:-1]]
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train).tolist()
    cm_test = confusion_matrix(y_test, y_pred_test).tolist()

    return jsonify({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_confusion_matrix": cm_train,
        "test_confusion_matrix": cm_test
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)