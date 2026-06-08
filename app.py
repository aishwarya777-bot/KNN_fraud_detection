import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Matches your exact pickle file name to prevent errors
MODEL_PATH = "Knn_Model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find the model file: '{MODEL_PATH}' in the root directory.")
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Single-file HTML/CSS Layout Served Digitally via String
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .custom-input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #f8fafc;
            transition: all 0.2s ease-in-out;
        }
        .custom-input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
            outline: none;
        }
    </style>
</head>
<body class="text-slate-200 py-10 px-4 sm:px-6 lg:px-8">
    <div class="max-w-5xl mx-auto">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 sm:text-5xl">
                KNN Model Prediction Portal
            </h1>
            <p class="mt-3 max-w-2xl mx-auto text-sm text-slate-400">
                Provide real-time telemetry details below to query the analytics engine.
            </p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2">
                <form id="predictionForm" class="glass-card rounded-2xl p-6 sm:p-8 space-y-6">
                    <h2 class="text-lg font-semibold text-indigo-300 border-b border-slate-700/50 pb-3">Feature Parameters</h2>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Transaction Amount ($)</label>
                            <input type="number" step="0.01" name="transaction_amount" value="150.00" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Hour of Day (0-23)</label>
                            <input type="number" min="0" max="23" name="hour_of_day" value="14" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Is Weekend?</label>
                            <select name="is_weekend" class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                                <option value="0">No (Weekday)</option>
                                <option value="1">Yes (Weekend)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Number of Items</label>
                            <input type="number" min="1" name="num_items" value="2" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Customer Age</label>
                            <input type="number" min="0" name="customer_age" value="34" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Previous Transactions</label>
                            <input type="number" min="0" name="prev_transactions" value="5" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Distance From Home (km)</label>
                            <input type="number" step="0.01" name="distance_from_home" value="4.2" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Device Type (ID)</label>
                            <input type="number" name="device_type" value="1" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Network Quality (0-3)</label>
                            <input type="number" min="0" max="3" name="network_quality" value="3" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Is First Transaction?</label>
                            <select name="is_first_transaction" class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Store Type (ID)</label>
                            <input type="number" name="store_type" value="2" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Velocity Score</label>
                            <input type="number" step="0.001" name="velocity_score" value="0.015" required class="w-full rounded-lg px-4 py-2.5 custom-input text-sm">
                        </div>
                    </div>
                    <div class="pt-4">
                        <button type="submit" class="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-3 px-4 rounded-xl shadow-lg transform active:scale-[0.99] transition-all">
                            Execute Framework Prediction
                        </button>
                    </div>
                </form>
            </div>

            <div class="lg:col-span-1">
                <div class="glass-card rounded-2xl p-6 sticky top-6 text-center flex flex-col justify-between h-[450px]">
                    <div>
                        <h2 class="text-lg font-semibold text-purple-300 border-b border-slate-700/50 pb-3 text-left">Output Diagnostics</h2>
                        <p class="text-xs text-slate-400 text-left mt-2">Evaluation results will compile automatically upon submitting the left frame matrix.</p>
                    </div>
                    <div id="resultContainer" class="my-auto py-6">
                        <div id="placeholderState" class="text-slate-500 space-y-2">
                            <svg class="w-16 h-16 mx-auto stroke-current opacity-40" fill="none" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
                            </svg>
                            <span class="text-sm font-medium block">Awaiting Entry Metrics</span>
                        </div>
                        <div id="successState" class="hidden space-y-4">
                            <span id="badgeOutput" class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold tracking-wide"></span>
                            <div class="text-5xl font-extrabold text-white tracking-tight" id="predictionText"></div>
                            <p class="text-sm text-slate-300 px-4" id="summaryText"></p>
                            <div class="border-t border-slate-700/50 mt-4 pt-4">
                                <span class="text-xs text-slate-400 block">Model Confidence Scale</span>
                                <span class="text-lg font-bold text-slate-200" id="confidenceText"></span>
                            </div>
                        </div>
                    </div>
                    <div class="text-[10px] text-slate-500 tracking-wider">POWERED BY KNN_MODEL.PKL</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const placeholder = document.getElementById('placeholderState');
            const successState = document.getElementById('successState');
            const badgeOutput = document.getElementById('badgeOutput');
            const predictionText = document.getElementById('predictionText');
            const summaryText = document.getElementById('summaryText');
            const confidenceText = document.getElementById('confidenceText');

            placeholder.classList.remove('hidden');
            placeholder.innerHTML = `<span class="text-sm animate-pulse text-indigo-400">Processing vector calculations...</span>`;
            successState.classList.add('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: new FormData(this)
                });
                const data = await response.json();
                placeholder.classList.add('hidden');

                if (data.success) {
                    successState.classList.remove('hidden');
                    predictionText.innerText = data.result_text;
                    summaryText.innerText = data.summary;
                    confidenceText.innerText = data.confidence;
                    badgeOutput.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold tracking-wide";
                    
                    if (data.badge_color === 'danger') {
                        badgeOutput.classList.add('bg-red-500/10', 'text-red-400', 'border', 'border-red-500/20');
                        badgeOutput.innerText = "CRITICAL METRIC ALERT";
                    } else {
                        badgeOutput.classList.add('bg-emerald-500/10', 'text-emerald-400', 'border', 'border-emerald-500/20');
                        badgeOutput.innerText = "STABLE CLASSIFICATION";
                    }
                } else {
                    placeholder.classList.remove('hidden');
                    placeholder.innerHTML = `<span class="text-sm text-red-400">Application Error:<br>${data.error}</span>`;
                }
            } catch (err) {
                placeholder.classList.remove('hidden');
                placeholder.innerHTML = `<span class="text-sm text-red-400">Failed to connect to backend pipeline.</span>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded on the server.'}), 500
    
    try:
        # Array matches the feature order your Knn_Model structure relies on
        features = [
            float(request.form.get('transaction_amount', 0)),
            int(request.form.get('hour_of_day', 0)),
            int(request.form.get('is_weekend', 0)),
            int(request.form.get('num_items', 1)),
            int(request.form.get('customer_age', 30)),
            int(request.form.get('prev_transactions', 0)),
            float(request.form.get('distance_from_home', 0.0)),
            int(request.form.get('device_type', 0)),
            int(request.form.get('network_quality', 0)),
            int(request.form.get('is_first_transaction', 0)),
            int(request.form.get('store_type', 0)),
            float(request.form.get('velocity_score', 0.0))
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = float(np.max(probabilities) * 100)
        except AttributeError:
            confidence = None

        result_text = f"Class {prediction}"
        badge_color = "danger" if prediction == 1 else "success"
        summary = "High Alert / High Risk Detected" if prediction == 1 else "Normal / Safe Status Verified"

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'result_text': result_text,
            'summary': summary,
            'badge_color': badge_color,
            'confidence': f"{confidence:.2f}%" if confidence else "N/A"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
