# 🚌 Sistema Inteligente de Detección del Bus Cooperativa Girón

**Proyecto Integrador Final — Parte II**  
**Asignatura:** Visión Artificial  
**Universidad Politécnica Salesiana — Carrera de Computación**  
**Docente:** Ing. Vladimir Robles Bykbaev  
**Período Lectivo:** Abril – Agosto 2026

---

## 📋 Descripción

Sistema de monitoreo de tráfico inteligente que detecta el **Bus Cooperativa Girón** (ruta Cuenca–Girón) en tiempo real usando técnicas clásicas de visión por computador y deep learning. El sistema integra dos componentes:

- **App de escritorio C++** — detecta el bus frame a frame usando HOG + SVM
- **Bot de Telegram Python** — recibe la alerta y segmenta la escena con YOLOv8-seg

Cuando la app C++ confirma la presencia del Bus Girón, envía automáticamente al Bot de Telegram tres entregables: imagen original con alerta, imagen segmentada y video corto.

---

## 🏗️ Arquitectura del Sistema

```
Webcam / Video
      ↓
App C++ (HOG+SVM)
  → Detección frame a frame
  → Confidence > 0.3 + 5 frames consecutivos
  → HTTP POST → localhost:5000
                    ↓
           Bot Python (Flask)
             → YOLOv8n-seg.pt
             → Segmentación de instancias
                    ↓
            Telegram API
              1. Imagen original + alerta
              2. Imagen segmentada (máscaras YOLO)
              3. Video MP4 segmentado (5s)
```

---

## 📊 Resultados del Modelo HOG+SVM

| Métrica | Valor |
|---|---|
| Accuracy | **97.70%** |
| Precisión | 98.06% |
| Sensibilidad | 96.89% |
| Especificidad | 98.39% |
| F1-Score | 97.47% |
| Support Vectors | 4.284 |
| Features HOG | 8.100 por imagen |

**Matriz de Confusión (test set — 1.827 muestras):**

|  | Pred No Bus | Pred Bus Girón |
|---|---|---|
| Real No Bus | 975 | 16 |
| Real Bus Girón | 26 | 810 |

---

## 📁 Estructura del Proyecto

```
proyecto_final/
├── dataset_preparation.py      # Extracción frames + augmentation
├── download_negatives.py       # Descarga negativas de COCO 2017
├── train_hog_svm.py            # Entrenamiento HOG+SVM
├── export_svm_opencv.py        # Exporta modelo a formato C++
├── giron_detector.cpp          # App de escritorio C++
├── telegram_bot.py             # Bot de Telegram + YOLOv8-seg
├── models/
│   ├── hog_svm_giron.pkl       # Modelo SVM (sklearn)
│   ├── scaler.pkl              # Normalizador
│   ├── sv.npy                  # Support vectors
│   ├── dv.npy                  # Dual coefficients
│   ├── mean.npy                # Scaler mean
│   ├── scale.npy               # Scaler std
│   ├── svm_params.txt          # Parámetros SVM (gamma, intercept)
│   └── metrics.txt             # Métricas de evaluación
├── dataset_giron/
│   ├── positives/
│   │   ├── fotos/              # 31 fotos originales del Bus Girón
│   │   └── videos/             # 6 videos del Bus Girón
│   └── negatives/              # Imágenes COCO + otros buses
└── dataset_final/
    ├── positives/              # 4.177 imágenes augmentadas
    └── negatives/              # 4.954 imágenes negativas
```

---

## ⚙️ Requisitos

### Python (Bot + Entrenamiento)
```
Python 3.13.7
opencv-python >= 5.0.0
scikit-learn >= 1.9.0
scikit-image
albumentations >= 2.0.8
ultralytics >= 8.4.0
flask
psutil
requests
python-telegram-bot
numpy
```

### C++ (Detector)
```
g++ >= 15.2.0
OpenCV 4.10.0 (libopencv-dev)
libcurl4-openssl-dev
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-giron.git
cd proyecto-giron
```

### 2. Crear entorno virtual Python
```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install opencv-python numpy scikit-learn \
    scikit-image albumentations ultralytics \
    flask psutil requests python-telegram-bot
```

### 3. Preparar el dataset
```bash
python3 dataset_preparation.py
```

### 4. Entrenar el modelo
```bash
python3 train_hog_svm.py
```

### 5. Exportar modelo a C++
```bash
python3 export_svm_opencv.py
```

### 6. Compilar la app C++
```bash
g++ giron_detector.cpp -o giron_detector \
    $(pkg-config --cflags --libs opencv4) \
    -lcurl -std=c++17 -O2
```

### 7. Ejecutar el sistema completo

**Terminal 1 — Bot de Telegram:**
```bash
source env/bin/activate
python3 telegram_bot.py
```

**Terminal 2 — Detector (webcam):**
```bash
./giron_detector
```

**Terminal 2 — Detector (archivo de video):**
```bash
./giron_detector video_giron.mp4
```

---

## 🔧 Configuración

### Bot de Telegram
Edita en `telegram_bot.py`:
```python
BOT_TOKEN = "tu_token_aqui"   # Obtenido de @BotFather
CHAT_ID   = "tu_chat_id"      # Obtenido de @userinfobot
```

### Parámetros del detector C++
Edita en `giron_detector.cpp`:
```cpp
const float SVM_THRESHOLD   = 0.3f;  // Umbral de decisión
const int   CONSEC_REQUIRED = 5;     // Frames consecutivos requeridos
const int   VOTE_WINDOW     = 10;    // Ventana de votación
const float VOTE_RATIO      = 0.6f;  // Ratio mínimo positivos
const int   CLIP_FRAMES     = 150;   // Duración del clip (5s a 30fps)
const int   COOLDOWN_F      = 200;   // Frames entre alertas
```

---

## 📈 Métricas de Rendimiento

| Componente | FPS | RAM |
|---|---|---|
| App C++ (detector) | ~25 FPS | ~340 MB |
| Bot Python (YOLOv8) | ~4-5 FPS | ~860 MB |
| Latencia Telegram | ~3-5 segundos | — |

---

## 🗂️ Dataset

- **Positivas:** 4.177 imágenes del Bus Cooperativa Girón
  - Fuente: fotos y videos propios (terminal Cuenca, ruta Cuenca-Girón)
  - Augmentation: flip, brillo, rotación, ruido, blur, perspectiva
- **Negativas:** 4.954 imágenes
  - COCO 2017 validation set (calles, personas, vehículos)
  - Otros buses ecuatorianos (Santa Isabel, Santiago de Gualaceo, etc.)
- **Herramienta:** Albumentations 2.0

---

## 📚 Referencias Bibliográficas

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR 2005. https://doi.org/10.1109/CVPR.2005.177

- Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297. https://doi.org/10.1007/BF00994018

- Jocher, G., et al. (2023). *Ultralytics YOLOv8*. Ultralytics. https://github.com/ultralytics/ultralytics

- Buslaev, A., et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information, 11(2), 125. https://doi.org/10.3390/info11020125

- Lin, T. Y., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014. https://cocodataset.org

- Rithika Chowta (2023). *Object Detection using LBP Cascade Classifier*. Medium. https://rithikachowta.medium.com/object-detection-lbp-cascade-classifier-generation-a1d1a1c2d0b

- Ejemplo-11 Object Detection MobileNetV3 OpenCV — Ing. Vladimir Robles Bykbaev (UPS, 2026). Código base empleado como referencia estructural para la arquitectura de la app de escritorio C++.

---

## 🔐 Notas de Seguridad

- El token del Bot de Telegram **NO** debe subirse al repositorio. Usa variables de entorno o un archivo `.env` ignorado por `.gitignore`.
- Agrega `models/*.pkl` y `dataset_final/` a `.gitignore` por el tamaño de los archivos.

---
