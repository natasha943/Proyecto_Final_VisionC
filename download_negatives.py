"""
=====================================================
DESCARGA DE IMÁGENES NEGATIVAS - COCO 2017
Proyecto Integrador Parte II - Visión Artificial UPS
=====================================================
Descarga SOLO las imágenes necesarias de COCO sin
bajar el dataset completo (~6GB).

Descarga selectiva:
  - Anotaciones JSON (~25MB)
  - ~4500 imágenes de calles/tráfico sin buses Girón

Uso:
  python3 download_negatives.py
"""

import requests
import json
import os
import zipfile
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =============================================================
# CONFIGURACIÓN
# =============================================================
OUTPUT_DIR    = "dataset_giron/negatives"
ANNOTATIONS_DIR = "coco_annotations"
TARGET_COUNT  = 4500       # Descargamos un poco más por si acaso
MAX_WORKERS   = 8          # Descargas paralelas
IMG_SIZE      = (128, 128) # Tamaño final

# Categorías de COCO que sirven como negativas
# (escenas urbanas SIN buses interprovincialesecuatorianos)
VALID_CATEGORIES = [
    'car', 'person', 'traffic light', 'stop sign',
    'bicycle', 'motorcycle', 'truck', 'bench',
    'backpack', 'handbag', 'umbrella', 'dog', 'cat',
    'chair', 'dining table', 'laptop', 'cell phone'
]

# Categorías a EXCLUIR (podrían confundirse con el bus Girón)
EXCLUDE_CATEGORIES = ['bus']


def download_file(url, dest_path, desc=""):
    """Descarga un archivo con barra de progreso simple."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  ✓ Ya existe: {dest_path.name}")
        return True

    print(f"  ⬇ Descargando {desc}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  ⬇ {desc}: {pct:.1f}% "
                          f"({downloaded//1024//1024}MB/"
                          f"{total//1024//1024}MB)",
                          end='', flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ✗ Error descargando {desc}: {e}")
        return False


def download_single_image(args):
    """Descarga una imagen individual de COCO."""
    img_info, output_dir = args
    url      = img_info['coco_url']
    filename = f"neg_{img_info['id']:012d}.jpg"
    dest     = Path(output_dir) / filename

    if dest.exists():
        return filename, True

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(dest, 'wb') as f:
                f.write(r.content)
            return filename, True
        return filename, False
    except:
        return filename, False


def main():
    print("=" * 60)
    print("  DESCARGA DE NEGATIVAS - COCO 2017")
    print("  Bus Girón — Visión Artificial UPS 2026")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Paso 1: Descargar anotaciones JSON (~25MB) ──────────
    print("\n─── PASO 1: Descargando anotaciones COCO (~25MB) ───")
    ann_zip  = f"{ANNOTATIONS_DIR}/annotations.zip"
    ann_file = f"{ANNOTATIONS_DIR}/instances_val2017.json"

    if not Path(ann_file).exists():
        ok = download_file(
            url="http://images.cocodataset.org/annotations/"
                "annotations_trainval2017.zip",
            dest_path=ann_zip,
            desc="anotaciones COCO 2017"
        )
        if ok:
            print("  📦 Extrayendo anotaciones...")
            with zipfile.ZipFile(ann_zip, 'r') as z:
                # Solo extraer el archivo de validación
                for name in z.namelist():
                    if 'instances_val2017' in name:
                        z.extract(name, ANNOTATIONS_DIR)
                        # Mover al directorio correcto
                        extracted = Path(ANNOTATIONS_DIR) / name
                        target    = Path(ann_file)
                        if extracted != target:
                            extracted.rename(target)
                        break
            print("  ✓ Anotaciones extraídas")
    else:
        print(f"  ✓ Anotaciones ya descargadas")

    # ── Paso 2: Filtrar imágenes válidas ────────────────────
    print("\n─── PASO 2: Filtrando imágenes válidas ───")
    print("  Cargando JSON de anotaciones...")

    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # Mapear categorías
    cat_name_to_id = {c['name']: c['id']
                      for c in coco['categories']}
    cat_id_to_name = {c['id']: c['name']
                      for c in coco['categories']}

    valid_ids   = {cat_name_to_id[n]
                   for n in VALID_CATEGORIES
                   if n in cat_name_to_id}
    exclude_ids = {cat_name_to_id[n]
                   for n in EXCLUDE_CATEGORIES
                   if n in cat_name_to_id}

    print(f"  Categorías válidas   : {len(valid_ids)}")
    print(f"  Categorías excluidas : {len(exclude_ids)} (buses)")

    # Agrupar anotaciones por imagen
    img_to_cats = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid not in img_to_cats:
            img_to_cats[iid] = set()
        img_to_cats[iid].add(ann['category_id'])

    # Filtrar: debe tener categorías válidas y NO tener buses
    img_id_to_info = {img['id']: img for img in coco['images']}

    valid_images = []
    for img_id, cats in img_to_cats.items():
        has_valid   = bool(cats & valid_ids)
        has_exclude = bool(cats & exclude_ids)
        if has_valid and not has_exclude:
            if img_id in img_id_to_info:
                valid_images.append(img_id_to_info[img_id])

    print(f"  Imágenes válidas encontradas: {len(valid_images)}")

    # Seleccionar aleatoriamente las que necesitamos
    random.seed(42)
    if len(valid_images) > TARGET_COUNT:
        selected = random.sample(valid_images, TARGET_COUNT)
    else:
        selected = valid_images

    print(f"  Imágenes seleccionadas: {len(selected)}")

    # ── Paso 3: Descargar imágenes en paralelo ───────────────
    print(f"\n─── PASO 3: Descargando {len(selected)} imágenes ───")
    print(f"  Destino: {OUTPUT_DIR}/")
    print(f"  Workers paralelos: {MAX_WORKERS}")
    print(f"  (Esto puede tomar 10-20 minutos según tu internet)\n")

    # Verificar cuántas ya están descargadas
    existing = len(list(Path(OUTPUT_DIR).glob("*.jpg")))
    if existing > 0:
        print(f"  ✓ Ya descargadas previamente: {existing}")

    args_list = [(img, OUTPUT_DIR) for img in selected]

    success = 0
    failed  = 0
    start   = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_image, a): a
                   for a in args_list}

        for i, future in enumerate(as_completed(futures)):
            filename, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1

            # Progreso cada 100 imágenes
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate    = success / elapsed if elapsed > 0 else 0
                eta     = (len(selected) - i) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(selected)}] "
                      f"✓{success} ✗{failed} | "
                      f"{rate:.1f} img/s | "
                      f"ETA: {eta/60:.1f}min")

    # ── Resumen ──────────────────────────────────────────────
    elapsed = time.time() - start
    total_in_dir = len(list(Path(OUTPUT_DIR).glob("*.jpg")))

    print(f"\n{'='*60}")
    print(f"  DESCARGA COMPLETADA")
    print(f"{'='*60}")
    print(f"  ✅ Descargadas exitosamente : {success}")
    print(f"  ✗  Fallidas                : {failed}")
    print(f"  📁 Total en carpeta        : {total_in_dir}")
    print(f"  ⏱  Tiempo total            : {elapsed/60:.1f} minutos")
    print(f"  📂 Ubicación               : {OUTPUT_DIR}/")

    if total_in_dir >= 4000:
        print(f"\n  ✅ Suficientes negativas para entrenar")
        print(f"  👉 Siguiente paso: ejecutar dataset_preparation.py")
    else:
        needed = 4000 - total_in_dir
        print(f"\n  ⚠  Faltan {needed} imágenes más")
        print(f"     Vuelve a ejecutar el script para reintentar")

    print("="*60)


if __name__ == "__main__":
    main()
