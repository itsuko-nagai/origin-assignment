import json, os, random, cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# ── paths ──────────────────────────────────────────────────────────────────
TAPING_DIR  = r"C:\Users\anayd\PycharmProjects\wall_cracks\Drywall-Join-Detect-1"
CRACKS_DIR  = r"C:\Users\anayd\PycharmProjects\wall_cracks\crack-1"
OUT_DIR     = r"C:\Users\anayd\PycharmProjects\wall_cracks\dataset"
#SAM_CKPT    = r"C:\Users\anayd\PycharmProjects\wall_cracks\sam_vit_b_01ec64.pth"
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────
def save_mask(mask_arr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray((mask_arr * 255).astype(np.uint8)).save(out_path)

def prompt_filename(image_filename, prompt):
    stem = os.path.splitext(image_filename)[0]  # "10.rf.f0b182..." # example
    return f"{stem}__{prompt.replace(' ', '_')}.png"

# ── process taping area ────────────────────────────────
def process_taping(split_name, ann_path, img_dir, out_split):
    print(f"\nProcessing taping area - {split_name}...")
    with open(ann_path) as f:
        data = json.load(f)

    id2file = {img['id']: img['file_name'] for img in data['images']}
    id2anns = {}
    for ann in data['annotations']:
        id2anns.setdefault(ann['image_id'], []).append(ann)

    prompt = "segment taping area"
    count = 0
    for img_info in data['images']:
        img_id   = img_info['id']
        img_path = os.path.join(img_dir, id2file[img_id])
        if not os.path.exists(img_path):
            continue

        h, w = img_info['height'], img_info['width']
        combined = np.zeros((h, w), dtype=np.uint8)

        for ann in id2anns.get(img_id, []):
            x, y, bw, bh = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            combined[y1:y2, x1:x2] = 1

        out_path = os.path.join(OUT_DIR, out_split, "masks",
                                prompt_filename(id2file[img_id], prompt))
        save_mask(combined, out_path)

        img_out = os.path.join(OUT_DIR, out_split, "images", id2file[img_id])
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        Image.open(img_path).save(img_out)

        count += 1
        if count % 50 == 0:
            print(f"  {count}/{len(data['images'])} done")

    print(f"  Finished {count} images.")

# ── process cracks ───────────────────────────────
def process_cracks(split_name, ann_path, img_dir, out_split):
    print(f"\nProcessing cracks - {split_name}...")
    with open(ann_path) as f:
        data = json.load(f)

    id2file = {img['id']: img['file_name'] for img in data['images']}
    id2anns = {}
    for ann in data['annotations']:
        id2anns.setdefault(ann['image_id'], []).append(ann)

    prompt = "segment crack"
    count  = 0
    for img_info in data['images']:
        img_id   = img_info['id']
        img_path = os.path.join(img_dir, id2file[img_id])
        if not os.path.exists(img_path):
            continue

        h, w = img_info['height'], img_info['width']
        combined = np.zeros((h, w), dtype=np.uint8)

        for ann in id2anns.get(img_id, []):
            for seg in ann.get('segmentation', []):
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(combined, [pts], 1)

        out_path = os.path.join(OUT_DIR, out_split, "masks",
                                prompt_filename(id2file[img_id], prompt))
        save_mask(combined, out_path)

        img_out = os.path.join(OUT_DIR, out_split, "images", id2file[img_id])
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        Image.open(img_path).save(img_out)

        count += 1
        if count % 50 == 0:
            print(f"  {count}/{len(data['images'])} done")

    print(f"  Finished {count} images.")

def split_cracks_train(ann_path):
    with open(ann_path) as f:
        data = json.load(f)
    imgs = data['images'][:]
    random.shuffle(imgs)
    cut  = int(len(imgs) * 0.85)
    return (
        {**data, 'images': imgs[:cut],
         'annotations': [a for a in data['annotations']
                         if a['image_id'] in {i['id'] for i in imgs[:cut]}]},
        {**data, 'images': imgs[cut:],
         'annotations': [a for a in data['annotations']
                         if a['image_id'] in {i['id'] for i in imgs[cut:]}]},
    )

# ── run everything ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # taping area
    process_taping("train", os.path.join(TAPING_DIR, "train", "_annotations.coco.json"),
                   os.path.join(TAPING_DIR, "train"), "train")
    process_taping("val",   os.path.join(TAPING_DIR, "valid", "_annotations.coco.json"),
                   os.path.join(TAPING_DIR, "valid"), "val")

    train_data, val_data = split_cracks_train(
        os.path.join(CRACKS_DIR, "train", "_annotations.coco.json"))

    import tempfile, json as _json
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        _json.dump(train_data, f); train_tmp = f.name
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        _json.dump(val_data, f);   val_tmp   = f.name

    process_cracks("train", train_tmp, os.path.join(CRACKS_DIR, "train"), "train")
    process_cracks("val",   val_tmp,   os.path.join(CRACKS_DIR, "train"), "val")
    process_cracks("test",  os.path.join(CRACKS_DIR, "test", "_annotations.coco.json"),
                   os.path.join(CRACKS_DIR, "test"), "test")

    os.unlink(train_tmp)
    os.unlink(val_tmp)

    print("\n Dataset saved to:", OUT_DIR)
    for split in ["train", "val", "test"]:
        masks = os.path.join(OUT_DIR, split, "masks")
        if os.path.exists(masks):
            print(f"  {split}: {len(os.listdir(masks))} masks")
