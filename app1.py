from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Naive Bayes Classifier</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0a0a0f;
      --surface: #13131a;
      --card: #1a1a24;
      --border: #2a2a3a;
      --accent: #7c6aff;
      --accent2: #ff6a9b;
      --text: #e8e8f0;
      --muted: #6b6b85;
      --success: #4ade80;
      --mono: 'Space Mono', monospace;
      --sans: 'DM Sans', sans-serif;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      padding: 40px 20px;
    }

    /* animated background grid */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(124,106,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,106,255,0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    .container {
      max-width: 760px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }

    header {
      margin-bottom: 48px;
      animation: fadeDown 0.6s ease both;
    }

    .tag {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--accent);
      background: rgba(124,106,255,0.1);
      border: 1px solid rgba(124,106,255,0.25);
      padding: 4px 12px;
      border-radius: 20px;
      display: inline-block;
      margin-bottom: 16px;
    }

    h1 {
      font-family: var(--mono);
      font-size: clamp(28px, 5vw, 42px);
      font-weight: 700;
      line-height: 1.15;
      background: linear-gradient(135deg, #e8e8f0 0%, var(--accent) 60%, var(--accent2) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .subtitle {
      color: var(--muted);
      font-size: 15px;
      margin-top: 10px;
      font-weight: 300;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 24px;
      animation: fadeUp 0.5s ease both;
    }

    .card-title {
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 20px;
    }

    .drop-zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 48px 24px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(124,106,255,0.02);
      position: relative;
    }

    .drop-zone:hover, .drop-zone.drag-over {
      border-color: var(--accent);
      background: rgba(124,106,255,0.06);
    }

    .drop-zone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }

    .drop-icon {
      font-size: 36px;
      margin-bottom: 12px;
      display: block;
    }

    .drop-text {
      font-size: 15px;
      color: var(--muted);
    }

    .drop-text strong {
      color: var(--accent);
    }

    .file-name {
      margin-top: 14px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--success);
      display: none;
    }

    .btn {
      display: block;
      width: 100%;
      margin-top: 20px;
      padding: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #fff;
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.15s;
    }

    .btn:hover { opacity: 0.88; transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    /* Loader */
    .loader {
      display: none;
      text-align: center;
      padding: 32px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 13px;
    }

    .spinner {
      width: 32px; height: 32px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
      margin: 0 auto 16px;
    }

    /* Results */
    #results { display: none; animation: fadeUp 0.5s ease both; }

    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }

    .metric-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      text-align: center;
    }

    .metric-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .metric-value {
      font-family: var(--mono);
      font-size: 32px;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .matrix-section { margin-top: 8px; }

    .matrix-title {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
      margin-top: 20px;
    }

    .matrix-table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono);
      font-size: 14px;
    }

    .matrix-table td {
      padding: 10px 14px;
      text-align: center;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
    }

    .matrix-table td.highlight {
      background: rgba(124,106,255,0.12);
      color: var(--accent);
      font-weight: 700;
    }

    .error-box {
      background: rgba(255,80,80,0.08);
      border: 1px solid rgba(255,80,80,0.3);
      border-radius: 10px;
      padding: 16px;
      color: #ff8080;
      font-family: var(--mono);
      font-size: 13px;
      display: none;
      margin-top: 16px;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeDown {
      from { opacity: 0; transform: translateY(-12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
<div class="container">

  <header>
    <div class="tag">ML · Classification</div>
    <h1>Naive Bayes<br>Classifier</h1>
    <p class="subtitle">Upload a CSV file — last column is used as the target label.</p>
  </header>

  <div class="card" style="animation-delay:0.1s">
    <div class="card-title">01 — Upload Dataset</div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="csvFile" accept=".csv" />
      <span class="drop-icon">📂</span>
      <div class="drop-text">Drop your <strong>CSV file</strong> here or click to browse</div>
      <div class="file-name" id="fileName"></div>
    </div>
    <button class="btn" id="runBtn" disabled onclick="runModel()">Run Classifier</button>
    <div class="error-box" id="errorBox"></div>
  </div>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    Training model…
  </div>

  <div id="results">
    <div class="card" style="animation-delay:0.05s">
      <div class="card-title">02 — Accuracy</div>
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">Training Accuracy</div>
          <div class="metric-value" id="trainAcc">—</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Testing Accuracy</div>
          <div class="metric-value" id="testAcc">—</div>
        </div>
      </div>
    </div>

    <div class="card" style="animation-delay:0.1s">
      <div class="card-title">03 — Confusion Matrices</div>
      <div class="matrix-section">
        <div class="matrix-title">Training Set</div>
        <div id="trainMatrix"></div>
        <div class="matrix-title">Testing Set</div>
        <div id="testMatrix"></div>
      </div>
    </div>
  </div>

</div>

<script>
  const fileInput = document.getElementById('csvFile');
  const dropZone  = document.getElementById('dropZone');
  const fileName  = document.getElementById('fileName');
  const runBtn    = document.getElementById('runBtn');

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      fileName.textContent = '✓ ' + fileInput.files[0].name;
      fileName.style.display = 'block';
      runBtn.disabled = false;
    }
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      fileName.textContent = '✓ ' + e.dataTransfer.files[0].name;
      fileName.style.display = 'block';
      runBtn.disabled = false;
    }
  });

  function buildMatrix(data) {
    let html = '<table class="matrix-table">';
    data.forEach((row, i) => {
      html += '<tr>';
      row.forEach((val, j) => {
        html += `<td class="${i === j ? 'highlight' : ''}">${val}</td>`;
      });
      html += '</tr>';
    });
    return html + '</table>';
  }

  async function runModel() {
    const file = fileInput.files[0];
    if (!file) return;

    document.getElementById('results').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    runBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();

      document.getElementById('loader').style.display = 'none';

      if (data.error) {
        const eb = document.getElementById('errorBox');
        eb.textContent = '⚠ ' + data.error;
        eb.style.display = 'block';
      } else {
        document.getElementById('trainAcc').textContent = (data.train_accuracy * 100).toFixed(1) + '%';
        document.getElementById('testAcc').textContent  = (data.test_accuracy  * 100).toFixed(1) + '%';
        document.getElementById('trainMatrix').innerHTML = buildMatrix(data.train_confusion_matrix);
        document.getElementById('testMatrix').innerHTML  = buildMatrix(data.test_confusion_matrix);
        document.getElementById('results').style.display = 'block';
      }
    } catch (err) {
      document.getElementById('loader').style.display = 'none';
      const eb = document.getElementById('errorBox');
      eb.textContent = '⚠ Could not connect to server.';
      eb.style.display = 'block';
    }

    runBtn.disabled = false;
  }
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    target_col = df.columns[-1]
    X = df[df.columns[:-1]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    cm_train  = confusion_matrix(y_train, y_pred_train).tolist()
    cm_test   = confusion_matrix(y_test,  y_pred_test).tolist()

    return jsonify({
        "train_accuracy":        train_acc,
        "test_accuracy":         test_acc,
        "train_confusion_matrix": cm_train,
        "test_confusion_matrix":  cm_test
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)