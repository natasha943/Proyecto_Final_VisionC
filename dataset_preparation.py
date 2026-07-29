"""
=============================================================
DATASET PREPARATION - Bus Cooperativa Girón (Cuenca-Girón)
Proyecto Integrador Parte II - Visión Por Computador UPS 2026
=============================================================
Compatible con Albumentations >= 2.0

ACTUALIZACIÓN: Ahora extrae frames de videos tanto en
positives/videos/ como en negatives/videos/ automáticamente.

Estructura esperada de entrada:
  dataset_giron/
  ├── positives/
  │   ├── fotos/     ← imágenes del Bus Girón
  │   └── videos/    ← videos del Bus Girón
  └── negatives/
      ├── fotos/     ← imágenes COCO + fotos otros buses
      │   (o directamente en negatives/ sin subcarpeta)
      └── videos/    ← videos de otros buses (Santa Isabel, etc.)

Estructura de salida:
  dataset_final/
  ├── positives/     ← 4000+ imágenes augmentadas 128x128
  ├── negatives/     ← 4000+ imágenes negativas 128x128
  ├── raw_crops/     ← frames extraídos + fotos originales
  ├── positive.txt   ← lista para opencv_traincascade
  └── negative.txt   ← lista para opencv_traincascade
"""

import cv2
import os
import numpy as np
import albumentations as A
from pathlib import Path
import random
import shutil

# =============================================================
# CONFIGURACIÓN
# =============================================================
INPUT_DIR   = "dataset_giron"
OUTPUT_DIR  = "dataset_final"
TARGET_SIZE = (128, 128)
TARGET_POS  = 4000
TARGET_NEG  = 4000
FRAME_SKIP  = 3   # positivas: 1 de cada 3 frames
NEG_FRAME_SKIP = 5  # negativas: 1 de cada 5 frames (menos densas)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================
# PIPELINE DE AUGMENTATION POSITIVAS
# =============================================================
augment_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.4),
            contrast_limit=(-0.3, 0.3), p=1.0),
        A.RandomGamma(gamma_limit=(60, 140), p=1.0),
        A.CLAHE(clip_limit=4.0, p=1.0),
    ], p=0.8),
    A.OneOf([
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20, p=1.0),
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20, p=1.0),
        A.ToGray(p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.ISONoise(p=1.0),
    ], p=0.5),
    A.OneOf([
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.Perspective(scale=(0.03, 0.08), p=1.0),
    ], p=0.6),
    A.RandomResizedCrop(
        size=TARGET_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.8, 1.2), p=0.4),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(8, 24),
        hole_width_range=(8, 24),
        fill=0, p=0.3),
    A.OneOf([
        A.ImageCompression(quality_range=(40, 80), p=1.0),
        A.Downscale(scale_range=(0.5, 0.8), p=1.0),
    ], p=0.3),
])

# Pipeline más simple para negativas
neg_augment_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.OneOf([
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT, p=1.0),
    ], p=0.4),
])


def setup_output_dirs():
    for d in [f"{OUTPUT_DIR}/positives",
              f"{OUTPUT_DIR}/negatives",
              f"{OUTPUT_DIR}/raw_crops",
              f"{OUTPUT_DIR}/raw_neg_frames"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Carpetas creadas en: {OUTPUT_DIR}/")


def extract_frames_from_videos(video_dir, output_dir,
                                frame_skip=FRAME_SKIP,
                                label=""):
    """Extrae frames de todos los videos en video_dir."""
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        print(f"  ⚠ No existe: {video_dir}")
        return 0

    video_files = (list(video_dir.glob("*.mp4")) +
                   list(video_dir.glob("*.MP4")) +
                   list(video_dir.glob("*.avi")) +
                   list(video_dir.glob("*.MOV")) +
                   list(video_dir.glob("*.mov")))

    # También buscar videos directo en el directorio padre
    # (por si pusieron los videos directamente en negatives/)
    parent = video_dir.parent
    if parent != video_dir:
        video_files += (list(parent.glob("*.mp4")) +
                        list(parent.glob("*.MP4")) +
                        list(parent.glob("*.avi")) +
                        list(parent.glob("*.MOV")) +
                        list(parent.glob("*.mov")))

    # Eliminar duplicados
    video_files = list(set(video_files))

    if not video_files:
        print(f"  ⚠ No se encontraron videos en: {video_dir}")
        return 0

    total = 0
    for vp in sorted(video_files):
        cap = cv2.VideoCapture(str(vp))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        dur      = n_frames / fps if fps > 0 else 0
        print(f"  📹 {vp.name}: {n_frames} frames, "
              f"{fps:.0f}fps, {dur:.1f}s")

        fc = saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fc % frame_skip == 0:
                h, w = frame.shape[:2]
                if w > 1280:
                    s = 1280 / w
                    frame = cv2.resize(frame,
                                       (int(w*s), int(h*s)))
                prefix = label + "_" if label else ""
                name = f"{prefix}{vp.stem}_f{fc:05d}.jpg"
                cv2.imwrite(str(output_dir / name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            fc += 1
        cap.release()
        print(f"     → {saved} frames extraídos")
        total += saved

    return total


def copy_images(source_dir, dest_dir, prefix=""):
    """Copia imágenes de source_dir a dest_dir."""
    source_dir = Path(source_dir)
    dest_dir   = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"  ⚠ No existe: {source_dir}")
        return 0

    images = (list(source_dir.glob("*.jpg"))  +
              list(source_dir.glob("*.jpeg")) +
              list(source_dir.glob("*.png"))  +
              list(source_dir.glob("*.JPG"))  +
              list(source_dir.glob("*.JPEG")))

    count = 0
    for i, p in enumerate(sorted(images)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        name = f"{prefix}{i:04d}_{p.stem}.jpg"
        cv2.imwrite(str(dest_dir / name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    print(f"  ✓ {count} imágenes copiadas de {source_dir.name}")
    return count


def augment_images(source_dir, output_dir,
                   target_count=TARGET_POS,
                   target_size=TARGET_SIZE,
                   prefix="pos",
                   pipeline=None):
    """Augmenta imágenes hasta llegar a target_count."""
    if pipeline is None:
        pipeline = augment_pipeline

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = (list(source_dir.glob("*.jpg"))  +
             list(source_dir.glob("*.jpeg")) +
             list(source_dir.glob("*.png")))

    if not paths:
        print(f"  ⚠ No hay imágenes en: {source_dir}")
        return 0

    # Cargar todas las imágenes base
    base = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            base.append(img)

    print(f"  📦 {len(base)} imágenes base | "
          f"objetivo: {target_count}")

    saved = 0

    # Guardar originales redimensionadas primero
    for i, img in enumerate(base):
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(output_dir / f"{prefix}_orig_{i:04d}.jpg"),
            out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    # Generar augmentaciones hasta el objetivo
    aug_idx = 0
    while saved < target_count:
        img = random.choice(base)
        try:
            aug = pipeline(image=img)["image"]
            aug = cv2.resize(aug, target_size)
            out = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(output_dir /
                    f"{prefix}_aug_{aug_idx:06d}.jpg"),
                out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved   += 1
            aug_idx += 1
            if saved % 500 == 0:
                print(f"  ... {saved}/{target_count}")
        except Exception as e:
            print(f"  ⚠ Error aug: {e}")

    print(f"  ✅ {saved} imágenes en: {output_dir}")
    return saved


def generate_info_files(pos_dir, neg_dir, out_dir):
    pos_dir = Path(pos_dir)
    neg_dir = Path(neg_dir)
    out_dir = Path(out_dir)

    pos_imgs = list(pos_dir.glob("*.jpg"))
    neg_imgs = list(neg_dir.glob("*.jpg"))

    w, h = TARGET_SIZE

    with open(out_dir / "positive.txt", "w") as f:
        for p in pos_imgs:
            f.write(f"{p.resolve()} 1 0 0 {w} {h}\n")

    with open(out_dir / "negative.txt", "w") as f:
        for p in neg_imgs:
            f.write(f"{p.resolve()}\n")

    print(f"\n  ✓ positive.txt → {len(pos_imgs)} entradas")
    print(f"  ✓ negative.txt → {len(neg_imgs)} entradas")


def main():
    print("=" * 60)
    print("  DATASET PREPARATION — Bus Cooperativa Girón")
    print("  Visión Artificial UPS — Parte II 2026")
    print("=" * 60)

    if not Path(INPUT_DIR).exists():
        print(f"\n❌ No encontrado: {INPUT_DIR}/")
        return

    setup_output_dirs()

    # ── POSITIVAS ──────────────────────────────────────────
    print("\n" + "─"*50)
    print("PASO 1: Extracción de frames positivos (Bus Girón)")
    print("─"*50)
    pos_frames_dir = f"{OUTPUT_DIR}/raw_frames_pos"
    Path(pos_frames_dir).mkdir(parents=True, exist_ok=True)

    n_frames_pos = extract_frames_from_videos(
        f"{INPUT_DIR}/positives/videos",
        pos_frames_dir,
        frame_skip=FRAME_SKIP,
        label="pos")
    print(f"\n  Frames positivos extraídos: {n_frames_pos}")

    print("\n" + "─"*50)
    print("PASO 2: Copiar fotos positivas")
    print("─"*50)
    n_fotos_pos = copy_images(
        f"{INPUT_DIR}/positives/fotos",
        f"{OUTPUT_DIR}/raw_crops",
        prefix="pos_")

    # Copiar frames positivos a raw_crops
    for f in Path(pos_frames_dir).glob("*.jpg"):
        shutil.copy(str(f),
                    f"{OUTPUT_DIR}/raw_crops/{f.name}")

    total_pos_raw = len(list(
        Path(f"{OUTPUT_DIR}/raw_crops").glob("*.jpg")))
    print(f"\n  Material positivo crudo: {total_pos_raw} imágenes")

    # ── NEGATIVAS ──────────────────────────────────────────
    print("\n" + "─"*50)
    print("PASO 3: Extracción de frames negativos (otros buses)")
    print("─"*50)
    neg_frames_dir = f"{OUTPUT_DIR}/raw_neg_frames"

    # Buscar videos en negatives/videos/ y negatives/ directo
    n_frames_neg = extract_frames_from_videos(
        f"{INPUT_DIR}/negatives/videos",
        neg_frames_dir,
        frame_skip=NEG_FRAME_SKIP,
        label="neg")

    # También buscar videos directamente en negatives/
    neg_root = Path(f"{INPUT_DIR}/negatives")
    extra_videos = (list(neg_root.glob("*.mp4")) +
                    list(neg_root.glob("*.MP4")) +
                    list(neg_root.glob("*.avi")) +
                    list(neg_root.glob("*.MOV")))
    if extra_videos:
        print(f"\n  Videos adicionales en negatives/:")
        for vp in extra_videos:
            cap = cv2.VideoCapture(str(vp))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  📹 {vp.name}: {n} frames, {fps:.0f}fps")
            fc = saved = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if fc % NEG_FRAME_SKIP == 0:
                    name = f"neg_{vp.stem}_f{fc:05d}.jpg"
                    cv2.imwrite(
                        str(Path(neg_frames_dir) / name),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                fc += 1
            cap.release()
            n_frames_neg += saved
            print(f"     → {saved} frames extraídos")

    print(f"\n  Frames negativos extraídos de videos: {n_frames_neg}")

    print("\n" + "─"*50)
    print("PASO 4: Copiar fotos negativas (COCO + otros buses)")
    print("─"*50)

    # Crear carpeta temporal para todas las negativas crudas
    neg_raw_all = Path(f"{OUTPUT_DIR}/raw_neg_all")
    neg_raw_all.mkdir(parents=True, exist_ok=True)

    # Copiar fotos negativas existentes (COCO, etc.)
    n_neg_fotos = copy_images(
        f"{INPUT_DIR}/negatives",
        str(neg_raw_all),
        prefix="neg_coco_")

    # Copiar frames de videos negativos
    for f in Path(neg_frames_dir).glob("*.jpg"):
        shutil.copy(str(f), str(neg_raw_all / f.name))

    # Copiar fotos de subcarpeta negatives/fotos/ si existe
    if Path(f"{INPUT_DIR}/negatives/fotos").exists():
        n_neg_fotos2 = copy_images(
            f"{INPUT_DIR}/negatives/fotos",
            str(neg_raw_all),
            prefix="neg_bus_")
        print(f"  Fotos de otros buses: {n_neg_fotos2}")

    total_neg_raw = len(list(neg_raw_all.glob("*.jpg")))
    print(f"\n  Material negativo crudo total: {total_neg_raw}")

    # ── AUGMENTATION ───────────────────────────────────────
    print("\n" + "─"*50)
    print("PASO 5: Augmentation de positivas")
    print("─"*50)
    n_pos = augment_images(
        source_dir=f"{OUTPUT_DIR}/raw_crops",
        output_dir=f"{OUTPUT_DIR}/positives",
        target_count=TARGET_POS,
        prefix="giron",
        pipeline=augment_pipeline)

    print("\n" + "─"*50)
    print("PASO 6: Augmentation de negativas")
    print("─"*50)
    n_neg = augment_images(
        source_dir=str(neg_raw_all),
        output_dir=f"{OUTPUT_DIR}/negatives",
        target_count=TARGET_NEG,
        prefix="neg",
        pipeline=neg_augment_pipeline)

    # ── ARCHIVOS DE ENTRENAMIENTO ───────────────────────────
    print("\n" + "─"*50)
    print("PASO 7: Generando archivos de entrenamiento")
    print("─"*50)
    generate_info_files(
        pos_dir=f"{OUTPUT_DIR}/positives",
        neg_dir=f"{OUTPUT_DIR}/negatives",
        out_dir=OUTPUT_DIR)

    # ── RESUMEN ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESUMEN FINAL")
    print("="*60)
    print(f"  ✅ Positivas generadas : {n_pos}")
    ok = "✅" if n_neg >= TARGET_NEG else "⚠ "
    print(f"  {ok} Negativas generadas : {n_neg}")
    print(f"  📁 Dataset listo en   : {OUTPUT_DIR}/")
    print(f"\n  🎯 Listo para reentrenar")
    print(f"  👉 Siguiente: python3 train_hog_svm.py")
    print("="*60)


if __name__ == "__main__":
    main()