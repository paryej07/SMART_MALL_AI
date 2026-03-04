from cbir_system import CBIRSystem
import os, pickle
DATASET_ROOT = os.path.join(os.getcwd(), "dataset")
print(f"Looking for images in: {DATASET_ROOT}")
if not os.path.exists(DATASET_ROOT):
    print("Error: 'dataset' folder not found!")
else:
    cbir = CBIRSystem()
    f, l, p = cbir.build_dataset(DATASET_ROOT)
    with open("dataset_features.pkl", 'wb') as file:
        pickle.dump({'features': f, 'labels': l, 'paths': p}, file)
    print(f"? Success! Processed {len(l)} images.")
