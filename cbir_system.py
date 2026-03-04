import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
class CBIRSystem:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array, verbose=0)
        return features.flatten() / (np.linalg.norm(features) + 1e-6)
    def build_dataset(self, dataset_root):
        features_list, labels_list, paths_list = [], [], []
        for subset in ['train', 'test']:
            subset_path = os.path.join(dataset_root, subset)
            if not os.path.exists(subset_path): continue
            for class_name in os.listdir(subset_path):
                class_path = os.path.join(subset_path, class_name)
                if not os.path.isdir(class_path): continue
                print(f"  Processing Class: {class_name}")
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            features_list.append(self.extract_features(img_path))
                            labels_list.append(class_name)
                            paths_list.append(img_path)
                        except:
                            continue
        return np.array(features_list), np.array(labels_list), np.array(paths_list)
