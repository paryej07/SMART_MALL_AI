from cbir_system import CBIRSystem
import os, pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
if not os.path.exists("dataset_features.pkl"):
    print("Error: Run build_dataset.py first!")
else:
    cbir = CBIRSystem()
    with open("dataset_features.pkl", 'rb') as f:
        data = pickle.load(f)
    query_path = input("Enter path to test image: ").strip().replace("'", "").replace('"', "")
    if os.path.exists(query_path):
        query_feat = cbir.extract_features(query_path).reshape(1, -1)
        sims = cosine_similarity(query_feat, data['features'])[0]
        best_idx = np.argmax(sims)
        print(f"? Result: {data['labels'][best_idx].upper()} (Score: {sims[best_idx]:.4f})")
    else:
        print("Image not found.")
