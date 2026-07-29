"""
=====================================================
ENTRENAMIENTO HOG + SVM - Bus Cooperativa Girón
Proyecto Integrador Parte II - Visión Artificial UPS
=====================================================
Versión compatible con OpenCV 5.x
Usa skimage.feature.hog para extracción de descriptores

Uso:
  python3 train_hog_svm.py

Salida:
  models/hog_svm_giron.pkl   ← modelo SVM entrenado
  models/scaler.pkl          ← normalizador
  models/hog_params.pkl      ← parámetros HOG
  models/metrics.txt         ← métricas de evaluación
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import StandardScaler

# =============================================================
# CONFIGURACIÓN
# =============================================================
DATASET_DIR = "dataset_final"
MODELS_DIR  = "models"
IMG_SIZE    = (128, 128)
RANDOM_SEED = 42

HOG_PARAMS = {
    "orientations"    : 9,
    "pixels_per_cell" : (8, 8),
    "cells_per_block" : (2, 2),
    "block_norm"      : "L2-Hys",
    "visualize"       : False,
    "transform_sqrt"  : True,
    "feature_vector"  : True,
    "channel_axis"    : None,
}

SVM_PARAMS = {
    "C"           : 1.0,
    "kernel"      : "rbf",
    "gamma"       : "scale",
    "probability" : True,
    "random_state": RANDOM_SEED,
}


def extract_hog_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return hog(gray, **HOG_PARAMS)


def load_dataset(dataset_dir):
    pos_dir = Path(dataset_dir) / "positives"
    neg_dir = Path(dataset_dir) / "negatives"
    X, y = [], []

    print("\n  Cargando positivas...")
    pos_images = sorted(list(pos_dir.glob("*.jpg")) +
                        list(pos_dir.glob("*.jpeg")) +
                        list(pos_dir.glob("*.png")))
    errors = 0
    for i, p in enumerate(pos_images):
        feat = extract_hog_features(p)
        if feat is None:
            errors += 1
            continue
        X.append(feat)
        y.append(1)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(pos_images)} positivas...")
    print(f"  ✓ {sum(1 for l in y if l==1)} positivas ({errors} errores)")

    print("\n  Cargando negativas...")
    neg_images = sorted(list(neg_dir.glob("*.jpg")) +
                        list(neg_dir.glob("*.jpeg")) +
                        list(neg_dir.glob("*.png")))
    errors = 0
    for i, p in enumerate(neg_images):
        feat = extract_hog_features(p)
        if feat is None:
            errors += 1
            continue
        X.append(feat)
        y.append(0)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(neg_images)} negativas...")
    print(f"  ✓ {sum(1 for l in y if l==0)} negativas ({errors} errores)")

    return np.array(X), np.array(y)


def evaluate_model(svm, scaler, X_test, y_test, models_dir):
    X_scaled = scaler.transform(X_test)
    y_pred   = svm.predict(X_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=["No Bus Girón", "Bus Girón"])

    tn, fp, fn, tp = cm.ravel()
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1          = (2 * precision * recall /
                   (precision + recall)
                   if (precision + recall) > 0 else 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  MÉTRICAS DE EVALUACIÓN")
    print(f"  {'='*50}")
    print(f"  Accuracy     : {accuracy*100:.2f}%")
    print(f"  Precisión    : {precision*100:.2f}%")
    print(f"  Sensibilidad : {recall*100:.2f}%")
    print(f"  Especificidad: {specificity*100:.2f}%")
    print(f"  F1-Score     : {f1*100:.2f}%")
    print(f"\n  Matriz de Confusión:")
    print(f"                  Pred No Bus  Pred Bus")
    print(f"  Real No Bus  :  {tn:10}  {fp:8}")
    print(f"  Real Bus     :  {fn:10}  {tp:8}")
    print(f"\n{report}")

    metrics_path = Path(models_dir) / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("MÉTRICAS HOG+SVM — Bus Cooperativa Girón\n")
        f.write("Proyecto Integrador Parte II — UPS 2026\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy     : {accuracy*100:.2f}%\n")
        f.write(f"Precisión    : {precision*100:.2f}%\n")
        f.write(f"Sensibilidad : {recall*100:.2f}%\n")
        f.write(f"Especificidad: {specificity*100:.2f}%\n")
        f.write(f"F1-Score     : {f1*100:.2f}%\n\n")
        f.write(f"Matriz de Confusión:\n")
        f.write(f"  TN={tn}  FP={fp}\n")
        f.write(f"  FN={fn}  TP={tp}\n\n")
        f.write("Reporte completo:\n")
        f.write(report)

    print(f"  ✓ Métricas guardadas: {metrics_path}")
    return accuracy, precision, recall, f1


def main():
    print("="*60)
    print("  ENTRENAMIENTO HOG+SVM — Bus Cooperativa Girón")
    print("  Visión Artificial UPS — Parte II 2026")
    print("="*60)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(DATASET_DIR).exists():
        print(f"\n❌ No encontrado: {DATASET_DIR}/")
        print(f"   Ejecuta primero: python3 dataset_preparation.py")
        return

    print("\n─── PASO 1: Parámetros HOG ───")
    cx, cy = HOG_PARAMS["pixels_per_cell"]
    bx, by = HOG_PARAMS["cells_per_block"]
    ori    = HOG_PARAMS["orientations"]
    n_cx   = IMG_SIZE[0] // cx
    n_cy   = IMG_SIZE[1] // cy
    desc   = (n_cx - bx + 1) * (n_cy - by + 1) * bx * by * ori
    print(f"  Orientaciones     : {ori}")
    print(f"  Pixels por celda  : {HOG_PARAMS['pixels_per_cell']}")
    print(f"  Celdas por bloque : {HOG_PARAMS['cells_per_block']}")
    print(f"  Features/imagen   : {desc}")
    print(f"  Tamaño imagen     : {IMG_SIZE}")

    print("\n─── PASO 2: Cargando y extrayendo HOG ───")
    print("  (Puede tomar 3-8 minutos...)")
    t0 = time.time()
    X, y = load_dataset(DATASET_DIR)
    print(f"\n  Cargado en {(time.time()-t0)/60:.1f} min")
    print(f"  Total     : {len(X)} muestras")
    print(f"  Positivas : {np.sum(y==1)}")
    print(f"  Negativas : {np.sum(y==0)}")
    print(f"  Features  : {X.shape[1]}")

    print("\n─── PASO 3: Split train/test (80/20) ───")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y)
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    print("\n─── PASO 4: Normalizando features ───")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print(f"  ✓ StandardScaler aplicado")

    print("\n─── PASO 5: Entrenando SVM ───")
    print(f"  Kernel: {SVM_PARAMS['kernel']} | C: {SVM_PARAMS['C']}")
    print(f"  (Puede tomar 5-20 minutos en CPU...)")
    t0  = time.time()
    svm = SVC(**SVM_PARAMS)
    svm.fit(X_train, y_train)
    print(f"  ✓ Entrenado en {(time.time()-t0)/60:.1f} min")

    print("\n─── PASO 6: Evaluando modelo ───")
    accuracy, precision, recall, f1 = evaluate_model(
        svm, scaler, X_test, y_test, MODELS_DIR)

    print("\n─── PASO 7: Guardando modelo ───")
    with open(f"{MODELS_DIR}/hog_svm_giron.pkl", "wb") as f:
        pickle.dump(svm, f)
    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODELS_DIR}/hog_params.pkl", "wb") as f:
        pickle.dump(HOG_PARAMS, f)
    print(f"  ✓ models/hog_svm_giron.pkl")
    print(f"  ✓ models/scaler.pkl")
    print(f"  ✓ models/hog_params.pkl")

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"  Accuracy : {accuracy*100:.2f}%")
    print(f"  F1-Score : {f1*100:.2f}%")
    if accuracy >= 0.90:
        print(f"  ✅ Modelo listo para detección en tiempo real")
        print(f"  👉 Siguiente: telegram_bot.py + giron_detector.cpp")
    elif accuracy >= 0.80:
        print(f"  ⚠  Accuracy aceptable — considera más imágenes")
    else:
        print(f"  ⚠  Accuracy baja — revisa calidad del dataset")
    print("="*60)


if __name__ == "__main__":
    main()
