"""
=====================================================
BOT DE TELEGRAM - Bus Cooperativa Girón
Proyecto Integrador Parte II - Visión Artificial UPS
=====================================================
Recibe imagen + video desde la app C++ (o para pruebas
desde línea de comandos), procesa con YOLOv8-seg y
responde automáticamente con:
  1. Imagen original + mensaje de alerta
  2. Imagen segmentada con máscaras YOLO
  3. Video MP4 con segmentación frame a frame

Uso:
  python3 telegram_bot.py

El bot queda escuchando en http://localhost:5000
La app C++ envía POST a esa URL con la imagen y video.
"""

import os
import io
import time
import psutil
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from telegram import Bot
from telegram.error import TelegramError

# =============================================================
# CONFIGURACIÓN
# =============================================================
BOT_TOKEN   = "8651783456:AAH-wKZbSrh8M6aTsgBXrFKLHMJFBaVoIms"
CHAT_ID     = "8646740311"
YOLO_MODEL  = "yolov8n-seg.pt"   
LISTEN_PORT = 5000
VEHICLE_NAME = "Bus Cooperativa Girón"

# Colores para máscaras de segmentación (BGR)
MASK_COLORS = [
    (255, 0,   0),    # Azul
    (0,   255, 0),    # Verde
    (0,   0,   255),  # Rojo
    (255, 255, 0),    # Cyan
    (255, 0,   255),  # Magenta
    (0,   255, 255),  # Amarillo
    (128, 0,   255),  # Violeta
    (255, 128, 0),    # Naranja
]

# =============================================================
# LOGGING
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================
# FLASK APP (recibe peticiones de C++)
# =============================================================
app = Flask(__name__)

# =============================================================
# CARGAR MODELO YOLO
# =============================================================
log.info("Cargando modelo YOLOv8-seg...")
# ====model = YOLO(YOLO_MODEL)
model = YOLO(YOLO_MODEL)
model.to('cpu')

log.info(f"✓ Modelo cargado: {YOLO_MODEL}")

# =============================================================
# MÉTRICAS DE RENDIMIENTO
# =============================================================
def get_ram_usage():
    """Retorna uso de RAM en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_metrics(label, fps=None, confidence=None):
    ram = get_ram_usage()
    msg = f"[MÉTRICAS] {label} | RAM: {ram:.1f}MB"
    if fps is not None:
        msg += f" | FPS: {fps:.1f}"
    if confidence is not None:
        msg += f" | Confidence: {confidence:.2f}"
    print(msg)
    log.info(msg)


# =============================================================
# SEGMENTACIÓN CON YOLO
# =============================================================
def segment_image(img_bgr):
    """
    Aplica YOLOv8-seg a una imagen y dibuja las máscaras.
    Retorna (imagen_segmentada, confidence_max, fps)
    """
    t0 = time.time()

    # Inferencia YOLO
    results = model(img_bgr, verbose=False)

    fps = 1.0 / (time.time() - t0)

    # Copiar imagen para dibujar
    output = img_bgr.copy()
    max_conf = 0.0

    result = results[0]

    # Dibujar máscaras de segmentación
    if result.masks is not None:
        masks  = result.masks.data.cpu().numpy()
        boxes  = result.boxes
        h, w   = img_bgr.shape[:2]

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            max_conf = max(max_conf, conf)

            # Redimensionar máscara al tamaño de la imagen
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST)

            # Color de la máscara
            color = MASK_COLORS[i % len(MASK_COLORS)]

            # Aplicar máscara translúcida
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[mask_resized == 1] = color
            output = cv2.addWeighted(output, 1.0,
                                     colored_mask, 0.4, 0)

            # Dibujar contorno de la máscara
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, 2)

            # Dibujar bounding box y etiqueta
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output,
                          (x1, y1 - th - 8),
                          (x1 + tw, y1), color, -1)
            cv2.putText(output, text,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
    else:
        # Sin detecciones — escribir texto informativo
        cv2.putText(output, "No objects detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    return output, max_conf, fps


def segment_video(video_path, output_path):
    """
    Aplica YOLOv8-seg frame a frame a un video.
    Guarda el video segmentado en output_path.
    Retorna (fps_promedio, confidence_promedio)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"No se pudo abrir: {video_path}")
        return 0, 0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(
        str(output_path), fourcc,
        orig_fps, (width, height))

    fps_list  = []
    conf_list = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        segmented, conf, fps = segment_image(frame)
        out.write(segmented)
        fps_list.append(fps)
        if conf > 0:
            conf_list.append(conf)
        frame_num += 1

        if frame_num % 30 == 0:
            avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
            log.info(f"  Frame {frame_num} | "
                     f"FPS: {avg_fps:.1f} | "
                     f"RAM: {get_ram_usage():.0f}MB")

    cap.release()
    out.release()

    avg_fps  = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0
    return avg_fps, avg_conf


# =============================================================
# ENVIAR RESPUESTAS A TELEGRAM
# =============================================================
async def send_telegram_responses(image_path, video_path):
    """
    Envía los 3 entregables requeridos al chat de Telegram:
    1. Imagen original + mensaje de alerta
    2. Imagen segmentada con máscaras YOLO
    3. Video MP4 segmentado
    """
    bot = Bot(token=BOT_TOKEN)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # ── 1. Imagen original + alerta ────────────────────
        log.info("Enviando imagen original...")
        alert_msg = (
            f"🚌 *Vehículo objetivo {VEHICLE_NAME} "
            f"detectado en la escena*\n\n"
            f"📅 Fecha/hora: {timestamp}\n"
            f"🤖 Sistema: HOG+SVM + YOLOv8-seg\n"
            f"📍 Ruta: Cuenca–Girón"
        )
        with open(image_path, 'rb') as f:
            await bot.send_photo(
                chat_id=CHAT_ID,
                photo=f,
                caption=alert_msg,
                parse_mode='Markdown')
        log.info("✓ Imagen original enviada")

        # ── 2. Imagen segmentada ───────────────────────────
        log.info("Generando imagen segmentada...")
        t0 = time.time()
        img_bgr = cv2.imread(str(image_path))
        segmented_img, conf, fps = segment_image(img_bgr)
        t_seg = time.time() - t0

        print_metrics("Segmentación imagen",
                      fps=fps, confidence=conf)

        # Guardar imagen segmentada temporal
        with tempfile.NamedTemporaryFile(
                suffix='.jpg', delete=False) as tmp:
            seg_img_path = tmp.name

        cv2.imwrite(seg_img_path, segmented_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        seg_caption = (
            f"🔍 *Imagen segmentada por YOLOv8-seg*\n\n"
            f"⏱ Tiempo inferencia: {t_seg*1000:.0f}ms\n"
            f"🎯 Confidence máx: {conf:.2f}\n"
            f"💾 RAM usada: {get_ram_usage():.0f}MB\n"
            f"📐 Modelo: yolov8n-seg.pt (preentrenado)"
        )
        with open(seg_img_path, 'rb') as f:
            await bot.send_photo(
                chat_id=CHAT_ID,
                photo=f,
                caption=seg_caption,
                parse_mode='Markdown')
        os.unlink(seg_img_path)
        log.info("✓ Imagen segmentada enviada")

        # ── 3. Video segmentado ────────────────────────────
        log.info("Generando video segmentado...")
        with tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False) as tmp:
            seg_video_path = tmp.name

        t0 = time.time()
        avg_fps, avg_conf = segment_video(
            video_path, seg_video_path)
        t_vid = time.time() - t0

        print_metrics("Segmentación video",
                      fps=avg_fps, confidence=avg_conf)

        video_caption = (
            f"🎬 *Video segmentado (5s) — YOLOv8-seg*\n\n"
            f"📊 FPS promedio: {avg_fps:.1f}\n"
            f"🎯 Confidence prom: {avg_conf:.2f}\n"
            f"⏱ Tiempo proceso: {t_vid:.1f}s\n"
            f"💾 RAM usada: {get_ram_usage():.0f}MB"
        )
        with open(seg_video_path, 'rb') as f:
            await bot.send_video(
                chat_id=CHAT_ID,
                video=f,
                caption=video_caption,
                parse_mode='Markdown',
                supports_streaming=True)
        os.unlink(seg_video_path)
        log.info("✓ Video segmentado enviado")

        log.info("✅ Los 3 entregables enviados a Telegram")

    except TelegramError as e:
        log.error(f"Error de Telegram: {e}")
    except Exception as e:
        log.error(f"Error general: {e}")
        import traceback
        traceback.print_exc()


def run_async_send(image_path, video_path):
    """Ejecuta el envío asíncrono en un hilo separado."""
    asyncio.run(send_telegram_responses(image_path, video_path))


# =============================================================
# ENDPOINT FLASK - recibe peticiones de C++
# =============================================================
@app.route('/detect', methods=['POST'])
def receive_detection():
    """
    Endpoint que recibe imagen + video desde la app C++.
    Formato: multipart/form-data con campos:
      - image: archivo de imagen (JPEG)
      - video: archivo de video (MP4)
    """
    log.info("=== Detección recibida desde C++ ===")
    print_metrics("Recepción de datos")

    if 'image' not in request.files:
        return jsonify({"error": "No image received"}), 400

    # Guardar imagen recibida
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(
            suffix='.jpg', delete=False) as tmp:
        image_path = tmp.name
    image_file.save(image_path)
    log.info(f"  Imagen guardada: {image_path}")

    # Guardar video recibido
    video_path = None
    if 'video' in request.files:
        video_file = request.files['video']
        with tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        video_file.save(video_path)
        log.info(f"  Video guardado: {video_path}")
    else:
        log.warning("  No se recibió video")

    # Procesar en hilo separado para no bloquear C++
    t = threading.Thread(
        target=run_async_send,
        args=(image_path, video_path),
        daemon=True)
    t.start()

    return jsonify({
        "status"  : "ok",
        "message" : "Procesando y enviando a Telegram...",
        "ram_mb"  : round(get_ram_usage(), 1)
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Verificar que el bot está activo."""
    return jsonify({
        "status"  : "running",
        "model"   : YOLO_MODEL,
        "ram_mb"  : round(get_ram_usage(), 1),
        "chat_id" : CHAT_ID
    }), 200


# =============================================================
# PRUEBA MANUAL - para testear sin la app C++
# =============================================================
async def test_bot(image_path, video_path):
    """Prueba el bot con archivos locales."""
    print("\n" + "="*50)
    print("  PRUEBA DEL BOT DE TELEGRAM")
    print("="*50)
    print(f"  Imagen : {image_path}")
    print(f"  Video  : {video_path}")
    print(f"  CHAT_ID: {CHAT_ID}")
    print("="*50 + "\n")
    await send_telegram_responses(image_path, video_path)


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    import sys

    print("="*60)
    print("  BOT TELEGRAM — Bus Cooperativa Girón")
    print("  Visión Artificial UPS — Parte II 2026")
    print("="*60)
    print(f"  Modelo YOLO : {YOLO_MODEL}")
    print(f"  Chat ID     : {CHAT_ID}")
    print(f"  Puerto      : {LISTEN_PORT}")
    print(f"  RAM inicial : {get_ram_usage():.0f}MB")
    print("="*60)

    # Modo prueba: python3 telegram_bot.py test imagen.jpg video.mp4
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        img_path = sys.argv[2]
        vid_path = sys.argv[3] if len(sys.argv) >= 4 else None

        if not Path(img_path).exists():
            print(f"❌ No encontrado: {img_path}")
            sys.exit(1)

        if vid_path and not Path(vid_path).exists():
            print(f"❌ No encontrado: {vid_path}")
            sys.exit(1)

        asyncio.run(test_bot(img_path, vid_path))
    else:
        # Modo servidor: escucha peticiones de C++
        print(f"\n  Iniciando servidor en puerto {LISTEN_PORT}...")
        print(f"  Endpoint: POST http://localhost:{LISTEN_PORT}/detect")
        print(f"  Health  : GET  http://localhost:{LISTEN_PORT}/health")
        print(f"\n  Esperando detecciones del Bus Girón...\n")
        app.run(
            host='0.0.0.0',
            port=LISTEN_PORT,
            debug=False,
            threaded=True)
