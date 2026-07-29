"""
Exporta SVM sklearn a formato FileStorage de OpenCV (XML)
sin usar cv2.ml - compatible con OpenCV 5.x
"""
import pickle
import numpy as np
from pathlib import Path
from skimage.feature import hog
import cv2
import random

MODELS_DIR  = "models"
DATASET_DIR = "dataset_final"
IMG_SIZE    = (128, 128)
HOG_PARAMS  = {
    "orientations"    : 9,
    "pixels_per_cell" : (8, 8),
    "cells_per_block" : (2, 2),
    "block_norm"      : "L2-Hys",
    "visualize"       : False,
    "transform_sqrt"  : True,
    "feature_vector"  : True,
    "channel_axis"    : None,
}

def extract_hog(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return hog(gray, **HOG_PARAMS)

def main():
    print("Exportando modelo para C++...")

    # Cargar modelo sklearn
    with open(f"{MODELS_DIR}/hog_svm_giron.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"  ✓ SVM cargado | kernel: {svm.kernel}")

    # Guardar support vectors y parámetros del SVM
    sv  = svm.support_vectors_.astype(np.float32)
    dv  = svm.dual_coef_.astype(np.float32)
    b   = float(svm.intercept_[0])
    gamma = float(svm._gamma)  # gamma calculado

    # Guardar scaler
    mean  = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    np.save(f"{MODELS_DIR}/sv.npy",    sv)
    np.save(f"{MODELS_DIR}/dv.npy",    dv)
    np.save(f"{MODELS_DIR}/mean.npy",  mean)
    np.save(f"{MODELS_DIR}/scale.npy", scale)

    # Guardar parámetros en archivo de texto
    with open(f"{MODELS_DIR}/svm_params.txt", "w") as f:
        f.write(f"intercept={b}\n")
        f.write(f"gamma={gamma}\n")
        f.write(f"n_support={sv.shape[0]}\n")
        f.write(f"n_features={sv.shape[1]}\n")

    print(f"  Support vectors : {sv.shape}")
    print(f"  Intercept (b)   : {b:.6f}")
    print(f"  Gamma           : {gamma:.8f}")
    print(f"  Features        : {sv.shape[1]}")

    # Verificar con muestra pequeña
    print("\n  Verificando con 20 imágenes...")
    random.seed(42)
    pos = random.sample(
        list(Path(f"{DATASET_DIR}/positives").glob("*.jpg")), 10)
    neg = random.sample(
        list(Path(f"{DATASET_DIR}/negatives").glob("*.jpg")), 10)

    correct = 0
    for p, label in [(pos, 1), (neg, 0)]:
        for img_path in p:
            feat = extract_hog(img_path)
            if feat is None:
                continue
            feat_scaled = (feat - mean) / scale
            pred = svm.predict([feat_scaled])[0]
            if pred == label:
                correct += 1

    print(f"  Accuracy muestra: {correct}/20")
    print(f"\n  ✅ Archivos exportados en models/:")
    print(f"     sv.npy, dv.npy, mean.npy, scale.npy")
    print(f"     svm_params.txt")

if __name__ == "__main__":
    main()
