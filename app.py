from flask import Flask, render_template, request, jsonify
import os, pickle, json, numpy as np, subprocess
from cbir_system import CBIRSystem
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Prices
def load_prices():
    defaults = {"Apple": 120.0, "Banana": 40.0, "Mango": 150.0, "Orange": 60.0, "Tomato": 35.0, "Beetroot": 50.0}
    if os.path.exists('prices.json'):
        with open('prices.json', 'r') as f: return json.load(f)
    return defaults

cbir = CBIRSystem()
with open("dataset_features.pkl", 'rb') as f:
    data = pickle.load(f)

def run_ai(img_path):
    feat = cbir.extract_features(img_path).reshape(1, -1)
    sims = cosine_similarity(feat, data['features'])[0]
    idx = np.argmax(sims)
    label = data['labels'][idx].capitalize()
    return label, round(float(sims[idx]) * 100, 1)

@app.route('/')
def index():
    return render_template('index.html', prices=load_prices())

@app.route('/scan_phone', methods=['POST'])
def scan_phone():
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    label, conf = run_ai(path)
    # FIX: Ensure 'img' key is sent correctly
    return jsonify({'label': label, 'unit_price': load_prices().get(label, 0.0), 'conf': conf, 'img': '/static/uploads/' + filename})

@app.route('/scan_pi', methods=['POST'])
def scan_pi():
    filename = "pi_cap.jpg"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        subprocess.run(["rpicam-still", "-o", path, "-t", "1", "--immediate", "--nopreview"], check=True)
        label, conf = run_ai(path)
        return jsonify({'label': label, 'unit_price': load_prices().get(label, 0.0), 'conf': conf, 'img': '/static/uploads/' + filename})
    except:
        return jsonify({'error': 'Camera Busy'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
