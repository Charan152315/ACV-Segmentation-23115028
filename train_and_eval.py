import os
import csv
import ssl
import time
import json
import random
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF


SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")


def download_kvasir_seg(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "Kvasir-SEG.zip")
    extract_path = os.path.join(data_dir, "Kvasir-SEG")
    
    if not os.path.exists(extract_path):
        print("Downloading Kvasir-SEG dataset (~46MB)...")
        url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
        
        # Bypass expired SSL certificate on Simula server
        ssl_context = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download and extraction complete!")
    return extract_path

class SegmentationDataContract(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=256, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = TF.resize(img, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)

        img_tensor = TF.to_tensor(img)
        img_norm = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask, dtype=np.uint8)
        mask_bin = (mask_np >= 128).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_bin)

        return img_norm, mask_tensor, img_tensor, os.path.basename(self.img_paths[idx])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ControlledUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, mode='baseline'):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        if mode == 'baseline':
            self.bottleneck = ConvBlock(64, 128, dilation=1)
        elif mode == 'dilated_bottleneck':
            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

def compute_confusion_matrix(preds, targets, num_classes=2, ignore_index=255):
    mask = (targets != ignore_index)
    preds, targets = preds[mask], targets[mask]
    return torch.bincount(num_classes * targets + preds, minlength=num_classes**2).reshape(num_classes, num_classes)

@torch.no_grad()
def evaluate_model(model, loader, criterion, num_classes=2):
    model.eval()
    total_loss = 0.0
    total_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=DEVICE)
    for imgs, masks, _, _ in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total_cm += compute_confusion_matrix(preds, masks, num_classes)

    intersection = torch.diag(total_cm).float()
    union = (total_cm.sum(dim=1) + total_cm.sum(dim=0) - torch.diag(total_cm)).float()
    iou_per_class = (intersection / (union + 1e-7)).cpu().numpy()
    return total_loss / len(loader.dataset), iou_per_class, float(np.mean(iou_per_class))

@torch.no_grad()
def measure_latency(model, input_shape=(1, 3, 256, 256), warmup=5, runs=30):
    model.eval()
    dummy = torch.randn(*input_shape, device=DEVICE)
    for _ in range(warmup):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ((time.perf_counter() - start) / runs) * 1000.0

def train_experiment(mode, train_loader, val_loader, test_loader, epochs=15, lr=1e-3):
    seed_everything(SEED)
    model = ControlledUNet(in_channels=3, num_classes=2, mode=mode).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_miou = 0.0
    history = []
    print(f"\n--- Training {mode.upper()} (Trainable Params: {params:,}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _, _ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss, val_ious, val_miou = evaluate_model(model, val_loader, criterion)
        
        row = {
            "mode": mode, "epoch": epoch,
            "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "val_iou_bg": round(float(val_ious[0]), 4),
            "val_iou_polyp": round(float(val_ious[1]), 4),
            "val_miou": round(val_miou, 4)
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f} | Polyp IoU: {val_ious[1]:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), f"best_{mode}.pth")

    model.load_state_dict(torch.load(f"best_{mode}.pth"))
    test_loss, test_ious, test_miou = evaluate_model(model, test_loader, criterion)
    latency_ms = measure_latency(model)

    summary = {
        "mode": mode, "parameters": params,
        "test_iou_bg": round(float(test_ious[0]), 4),
        "test_iou_polyp": round(float(test_ious[1]), 4),
        "test_miou": round(test_miou, 4),
        "latency_ms_per_img": round(latency_ms, 2),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    return model, history, summary

@torch.no_grad()
def save_visual_comparison(base_model, ctrl_model, test_ds, save_path="outputs/qualitative_results.png"):
    base_model.eval()
    ctrl_model.eval()
    indices = [0, 5, 10, 15]
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)))
    
    for row_idx, ds_idx in enumerate(indices):
        img_norm, mask_gt, img_raw, fname = test_ds[ds_idx]
        inp = img_norm.unsqueeze(0).to(DEVICE)
        pred_base = torch.argmax(base_model(inp), dim=1).squeeze(0).cpu().numpy()
        pred_ctrl = torch.argmax(ctrl_model(inp), dim=1).squeeze(0).cpu().numpy()
        
        axes[row_idx, 0].imshow(img_raw.permute(1, 2, 0).numpy())
        axes[row_idx, 0].set_title(f"Source: {fname[:10]}...")
        axes[row_idx, 1].imshow(mask_gt.numpy(), cmap="gray")
        axes[row_idx, 1].set_title("Ground Truth (Human)")
        axes[row_idx, 2].imshow(pred_base, cmap="gray")
        axes[row_idx, 2].set_title("Baseline (r=1,1)")
        axes[row_idx, 3].imshow(pred_ctrl, cmap="gray")
        axes[row_idx, 3].set_title("Controlled (r=2,4)")
        for col in range(4):
            axes[row_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved qualitative comparison grid to {save_path}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    root = download_kvasir_seg()
    img_dir, mask_dir = os.path.join(root, "images"), os.path.join(root, "masks")
    
    all_files = sorted(os.listdir(img_dir))
    random.Random(SEED).shuffle(all_files)
    
    # Automatically adjusts dataset size & epochs if running on CPU vs GPU
    num_samples = 400 if torch.cuda.is_available() else 160
    num_epochs = 20 if torch.cuda.is_available() else 12
    all_files = all_files[:num_samples]
    
    img_paths = [os.path.join(img_dir, f) for f in all_files]
    mask_paths = [os.path.join(mask_dir, f) for f in all_files]

    n_train, n_val = int(0.70 * len(all_files)), int(0.15 * len(all_files))
    train_ds = SegmentationDataContract(img_paths[:n_train], mask_paths[:n_train], augment=True)
    val_ds = SegmentationDataContract(img_paths[n_train:n_train+n_val], mask_paths[n_train:n_train+n_val], augment=False)
    test_ds = SegmentationDataContract(img_paths[n_train+n_val:], mask_paths[n_train+n_val:], augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    base_model, base_hist, base_sum = train_experiment("baseline", train_loader, val_loader, test_loader, epochs=num_epochs)
    ctrl_model, ctrl_hist, ctrl_sum = train_experiment("dilated_bottleneck", train_loader, val_loader, test_loader, epochs=num_epochs)

    with open("outputs/training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_hist[0].keys())
        writer.writeheader()
        writer.writerows(base_hist + ctrl_hist)

    results = {
        "baseline": base_sum,
        "controlled_change": ctrl_sum,
        "environment_versions": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "seed": SEED
        }
    }
    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    save_visual_comparison(base_model, ctrl_model, test_ds, save_path="outputs/qualitative_results.png")
    print("\nFINAL METRICS SUMMARY:\n", json.dumps(results, indent=2))