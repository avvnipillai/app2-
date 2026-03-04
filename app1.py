from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
 
app = Flask(__name__)
 
# HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Naive Bayes Predictor</title>
</head>
<body>
    <h2>Naive Bayes CSV Upload</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit">Upload & Predict</button>
    </form>
 
    {% if result %}
        <h3>Results</h3>
        <p><b>Train Accuracy:</b> {{ result.train_accuracy }}</p>
        <p><b>Test Accuracy:</b> {{ result.test_accuracy }}</p>
 
        <h4>Train Confusion Matrix</h4>
        <pre>{{ result.train_confusion_matrix }}</pre>
 
        <h4>Test Confusion Matrix</h4>
        <pre>{{ result.test_confusion_matrix }}</pre>
    {% endif %}
</body>
</html>
"""
 
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)
 
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
 
    file = request.files['file']
    df = pd.read_csv(file)
 
    # Last column as target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    model = GaussianNB()
    model.fit(X_train, y_train)
 
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
 
    result = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_confusion_matrix": confusion_matrix(y_train, model.predict(X_train)).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, model.predict(X_test)).tolist()
    }
 
    return render_template_string(HTML_PAGE, result=result)
 
if __name__ == "__main__":
    app.run(debug=True)
 