import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DATASET_DIR = r"C:\Users\anayd\PycharmProjects\wall_cracks\dataset"
CKPT        = r"C:\Users\anayd\PycharmProjects\wall_cracks\checkpoints_clipseg\best"
PRED_DIR    = r"C:\Users\anayd\PycharmProjects\wall_cracks\predictions_clipseg"
DEVICE      = "cuda"

os.makedirs(PRED_DIR, exist_ok=True)

print("Loading CLIPSeg...")
processor = CLIPSegProcessor.from_pretrained(CKPT)
model     = CLIPSegForImageSegmentation.from_pretrained(CKPT)
model.to(DEVICE).eval()
print("Loaded.")

def get_prompt(mask_filename):
    return mask_filename.split("__")[1].replace(".png","").replace("_"," ")

class TestDataset(Dataset):
    def __init__(self, split):
        self.img_dir  = os.path.join(DATASET_DIR, split, "images")
        self.mask_dir = os.path.join(DATASET_DIR, split, "masks")
        self.pairs = []
        for mf in os.listdir(self.mask_dir):
            stem = mf.split("__")[0]
            if os.path.exists(os.path.join(self.img_dir, stem + ".jpg")):
                self.pairs.append((stem + ".jpg", mf))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]
        prompt = get_prompt(mask_file)
        img    = Image.open(os.path.join(self.img_dir, img_file)).convert("RGB")
        mask   = Image.open(os.path.join(self.mask_dir, mask_file)).convert("L")
        mask   = mask.resize((352, 352), Image.NEAREST)
        mask_arr = (np.array(mask) > 0).astype(np.float32)
        enc    = processor(text=[prompt], images=[img],
                           padding=True, return_tensors="pt")
        return {
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mask":           torch.tensor(mask_arr).unsqueeze(0),
            "mask_file":      mask_file,
            "img_file":       img_file,
        }

def run_eval(split):
    ds = TestDataset(split)
    results = {"crack": {"iou":[],"dice":[]}, "taping":{"iou":[],"dice":[]}}
    t0 = time.time()

    with torch.no_grad():
        for item in ds:
            pv  = item["pixel_values"].unsqueeze(0).to(DEVICE)
            ids = item["input_ids"].unsqueeze(0).to(DEVICE)
            am  = item["attention_mask"].unsqueeze(0).to(DEVICE)
            gt  = item["mask"].unsqueeze(0).to(DEVICE)
            mask_file = item["mask_file"]
            img_file  = item["img_file"]

            out  = model(pixel_values=pv, input_ids=ids, attention_mask=am)
            pred = out.logits.unsqueeze(1)  # (1,1,H,W)

            # metrics at model output size
            pred_up  = nn.functional.interpolate(pred, size=gt.shape[-2:],
                                                  mode="bilinear", align_corners=False)
            pred_bin = (torch.sigmoid(pred_up) > 0.5).float()
            pred_f   = pred_bin.flatten(1); gt_f = gt.flatten(1)
            inter    = (pred_f * gt_f).sum(1)
            union    = pred_f.sum(1) + gt_f.sum(1) - inter
            iou      = ((inter+1e-6)/(union+1e-6)).item()
            dice     = ((2*inter+1e-6)/(pred_f.sum(1)+gt_f.sum(1)+1e-6)).item()

            ptype = "crack" if "crack" in mask_file else "taping"
            results[ptype]["iou"].append(iou)
            results[ptype]["dice"].append(dice)

            # save mask at original image size
            orig_img = Image.open(os.path.join(DATASET_DIR, split, "images", img_file))
            w, h     = orig_img.size
            pred_full = nn.functional.interpolate(pred, size=(h, w),
                                                   mode="bilinear", align_corners=False)
            pred_np   = (torch.sigmoid(pred_full)[0,0].cpu().numpy() > 0.5)
            Image.fromarray((pred_np*255).astype(np.uint8)).save(
                os.path.join(PRED_DIR, mask_file))

    elapsed = time.time() - t0
    print(f"\n=== {split.upper()} RESULTS ===")
    for ptype, vals in results.items():
        if vals["iou"]:
            print(f"  {ptype:10s} | n={len(vals['iou']):4d} | "
                  f"mIoU={np.mean(vals['iou']):.4f} | "
                  f"Dice={np.mean(vals['dice']):.4f}")
    all_iou  = results["crack"]["iou"]  + results["taping"]["iou"]
    all_dice = results["crack"]["dice"] + results["taping"]["dice"]
    print(f"  {'OVERALL':10s} | n={len(all_iou):4d} | "
          f"mIoU={np.mean(all_iou):.4f} | "
          f"Dice={np.mean(all_dice):.4f}")
    print(f"  Avg inference: {elapsed/len(ds)*1000:.1f} ms/image")

run_eval("val")
run_eval("test")
print(f"\nPredictions saved to: {PRED_DIR}")