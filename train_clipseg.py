import os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ── config ─────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\anayd\PycharmProjects\wall_cracks\dataset"
OUT_DIR     = r"C:\Users\anayd\PycharmProjects\wall_cracks\checkpoints_clipseg"
SEED        = 42
BATCH_SIZE  = 8
NUM_EPOCHS  = 30
LR          = 1e-4
DEVICE      = "cuda" # my RTX 4060 Laptop
IMG_SIZE    = 352

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── prompt map ─────────────────────────────────────────────────────────────
def get_prompt(mask_filename):
    raw = mask_filename.split("__")[1].replace(".png", "").replace("_", " ")
    return raw

# ── dataset ────────────────────────────────────────────────────────────────
class DrywallDataset(Dataset):
    def __init__(self, split, processor):
        self.img_dir  = os.path.join(DATASET_DIR, split, "images")
        self.mask_dir = os.path.join(DATASET_DIR, split, "masks")
        self.processor = processor

        self.pairs = []
        for mf in os.listdir(self.mask_dir):
            stem     = mf.split("__")[0]
            img_file = stem + ".jpg"
            if os.path.exists(os.path.join(self.img_dir, img_file)):
                self.pairs.append((img_file, mf))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]
        prompt = get_prompt(mask_file)

        img  = Image.open(os.path.join(self.img_dir,  img_file)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_file)).convert("L")

        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_arr = (np.array(mask) > 0).astype(np.float32)
        encoding = self.processor(
            text=[prompt],
            images=[img],
            padding=True,
            return_tensors="pt"
        )

        return {
            "pixel_values":  encoding["pixel_values"].squeeze(0),
            "input_ids":     encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "mask":          torch.tensor(mask_arr).unsqueeze(0),
            "prompt":        prompt,
        }

def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        b["input_ids"]      = torch.nn.functional.pad(b["input_ids"],      (0, pad))
        b["attention_mask"] = torch.nn.functional.pad(b["attention_mask"], (0, pad))
    return {
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "mask":           torch.stack([b["mask"]           for b in batch]),
    }

# ── loss & metrics ─────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred).flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    return 1 - (2*inter + smooth) / (pred.sum(1) + target.sum(1) + smooth)

def total_loss(pred, target):
    pred_up = nn.functional.interpolate(pred, size=target.shape[-2:],
                                         mode="bilinear", align_corners=False)
    return (nn.functional.binary_cross_entropy_with_logits(pred_up, target) +
            dice_loss(pred_up, target).mean())

def compute_metrics(pred, target):
    pred_up  = nn.functional.interpolate(pred, size=target.shape[-2:],
                                          mode="bilinear", align_corners=False)
    pred_bin = (torch.sigmoid(pred_up) > 0.5).float()
    pred_f   = pred_bin.flatten(1); target_f = target.flatten(1)
    inter    = (pred_f * target_f).sum(1)
    union    = pred_f.sum(1) + target_f.sum(1) - inter
    iou      = ((inter+1e-6)/(union+1e-6)).mean().item()
    dice     = ((2*inter+1e-6)/(pred_f.sum(1)+target_f.sum(1)+1e-6)).mean().item()
    return iou, dice

# ── load model ─────────────────────────────────────────────────────────────
print("Loading CLIPSeg...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(DEVICE)
print("CLIPSeg loaded.")

# ── data loaders ───────────────────────────────────────────────────────────
train_ds = DrywallDataset("train", processor)
val_ds   = DrywallDataset("val",   processor)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, collate_fn=collate_fn)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

# ── optimizer ──────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
best_iou  = 0.0
history   = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

# ── training loop ──────────────────────────────────────────────────────────
print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    t0 = time.time()
    train_losses = []

    for b_idx, batch in enumerate(train_dl):
        pixel_values  = batch["pixel_values"].to(DEVICE)
        input_ids     = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        masks_gt      = batch["mask"].to(DEVICE)

        outputs = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask)

        # outputs.logits shape: (B, H, W)
        pred = outputs.logits.unsqueeze(1)
        loss = total_loss(pred, masks_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (b_idx+1) % 50 == 0:
            print(f"  epoch {epoch} | batch {b_idx+1}/{len(train_dl)} "
                  f"| loss={loss.item():.4f}")

    # ── val ──────────────────────────────────────────────────────────────────
    model.eval()
    val_losses, ious, dices = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            pixel_values   = batch["pixel_values"].to(DEVICE)
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            masks_gt       = batch["mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask)
            pred = outputs.logits.unsqueeze(1)
            val_losses.append(total_loss(pred, masks_gt).item())
            iou, dice = compute_metrics(pred, masks_gt)
            ious.append(iou); dices.append(dice)

    scheduler.step()
    t_loss = np.mean(train_losses)
    v_loss = np.mean(val_losses)
    v_iou  = np.mean(ious)
    v_dice = np.mean(dices)

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["val_iou"].append(v_iou)
    history["val_dice"].append(v_dice)

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | train={t_loss:.4f} | "
          f"val={v_loss:.4f} | iou={v_iou:.4f} | dice={v_dice:.4f} | "
          f"time={time.time()-t0:.0f}s")

    if v_iou > best_iou:
        best_iou = v_iou
        model.save_pretrained(os.path.join(OUT_DIR, "best"))
        processor.save_pretrained(os.path.join(OUT_DIR, "best"))
        print(f"  → best checkpoint saved (iou={best_iou:.4f})")

model.save_pretrained(os.path.join(OUT_DIR, "final"))
processor.save_pretrained(os.path.join(OUT_DIR, "final"))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train loss")
axes[0].plot(history["val_loss"],   label="val loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
axes[1].plot(history["val_iou"],  label="val mIoU")
axes[1].plot(history["val_dice"], label="val Dice")
axes[1].set_title("Metrics"); axes[1].legend(); axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("training_curves_clipseg.png", dpi=150)
plt.show()
print(f"\nDone! Best val mIoU: {best_iou:.4f}")
