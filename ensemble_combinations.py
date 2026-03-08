# ensemble_combinations.py
# Comprehensive Ensemble Generation & Comparison (Scale x4)
# Generates: Individual, Pairwise, Global, Patch-wise (Variance), Frequency Fusion

import os
import csv
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from math import ceil
from models.edsr import edsr_x
from models.rrdb import RRDBNet
from models.srcnn import MedicalSRCNN_Plus as SRCNN

# ------------------ CONFIG ---------------------
HR_ROOT_FOLDER = "10_IMAGES"  # Or "brain_mri_scan_images"
OUT_ROOT = "results_combinations"
SCALE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# Ensure output directories exist
DIRS = [
    "edsr", "esrgan", "srcnn",
    "ens_es", "ens_eg", "ens_sg",  # Pairwise
    "ens_global",                  # E+S+G
    "ens_patch",                   # Variance-based
    "ens_freq",                    # FFT
    "comparisons"
]
for d in DIRS:
    os.makedirs(os.path.join(OUT_ROOT, d), exist_ok=True)

print(f"DEBUG: Output directory is {os.path.abspath(OUT_ROOT)}")

# ------------------ LOAD MODELS -----------------
def load_edsr(pth="weights/edsr.pth"):
    model = edsr_x(scale=SCALE, n_resblocks=16, n_feats=64, n_colors=1).to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"✔ EDSR loaded (Scale {SCALE})")
    return model

def load_esrgan(pth="weights/esrgan.pth"):
    # Note: gc=8 matches the checkpoint
    model = RRDBNet(in_nc=1, out_nc=1, nf=16, nb=1, gc=8, scale=SCALE).to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"✔ ESRGAN loaded (Scale {SCALE})")
    return model

def load_srcnn(pth="weights/srcnn.pth"):
    model = SRCNN().to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("✔ SRCNN loaded")
    return model

# ------------------ HELPERS ---------------------
def pil_to_np(img):
    return np.array(img).astype(np.float32) / 255.0

def np_to_pil(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))

def tensor_from_np(x):
    # Add batch and channel dims: (H, W) -> (1, 1, H, W)
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def tensor_to_np(t):
    return t.squeeze().detach().cpu().numpy().astype(np.float32)

def create_lr_bicubic(hr_np):
    h, w = hr_np.shape
    lr = cv2.resize(hr_np, (w//SCALE, h//SCALE), interpolation=cv2.INTER_CUBIC)
    return lr

def compute_metrics(hr, pred):
    hr = np.clip(hr, 0, 1)
    pred = np.clip(pred, 0, 1)
    p = sk_psnr(hr, pred, data_range=1.0)
    s = sk_ssim(hr, pred, data_range=1.0)
    return float(p), float(s)

# ------------------ PROCESSING ------------------
def create_a4_grid(grid_imgs, w, h, fname):

    # ---------------------------
    # A4 size in pixels at 300 DPI
    # ---------------------------
    A4_WIDTH = 2480      # px
    A4_HEIGHT = 3508     # px
    MARGIN = 60
    TITLE_SPACE = 150
    TEXT_HEIGHT = 150    # Space for text below image
    GAP = 40
    ROW_GAP = 100

    # Layout: 2 Rows
    # Row 1: 6 images
    # Row 2: 5 images
    COLS = 6
    
    # Calculate thumbnail width to fit 6 columns
    # Available width = A4_WIDTH - 2*MARGIN - (COLS-1)*GAP
    available_width = A4_WIDTH - 2 * MARGIN - (COLS - 1) * GAP
    scaled_w = int(available_width / COLS)
    
    # Maintain aspect ratio
    scale = scaled_w / w
    scaled_h = int(h * scale)

    # Final canvas
    canvas = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title_text = "Comparison of Super-Resolution Outputs (SRCNN, EDSR, ESRGAN, Ensembles)"
    try:
        title_font = ImageFont.truetype("arial.ttf", 56)
    except:
        title_font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((A4_WIDTH - tw)//2, 50), title_text, fill=(0,0,0), font=title_font)

    # Label font
    try:
        label_font = ImageFont.truetype("arial.ttf", 24)
    except:
        label_font = ImageFont.load_default()

    # ---------------------------
    # Paste all items
    # ---------------------------
    start_y = TITLE_SPACE
    
    for idx, (img_np, label) in enumerate(grid_imgs):
        
        # Determine Row and Column
        row = idx // COLS
        col = idx % COLS
        
        # Calculate X and Y
        x = MARGIN + col * (scaled_w + GAP)
        y = start_y + row * (scaled_h + TEXT_HEIGHT + ROW_GAP)
        
        # Center the second row (5 images) if desired, or just left align
        # Let's left align for consistency with top row as per standard grid
        
        # Resize and Paste Image
        pil_img = np_to_pil(img_np).resize((scaled_w, scaled_h), Image.BICUBIC)
        canvas.paste(pil_img, (x, y))

        # Draw Text Below Image
        text_y = y + scaled_h + 10
        for i, line in enumerate(label.split("\n")):
            draw.text((x, text_y + i*28), line, fill=(0,0,0), font=label_font)

    # Save
    outname = fname.replace(".", "_A4cmp.")
    comp_path = os.path.join(OUT_ROOT, "comparisons", outname)
    canvas.save(comp_path, dpi=(300,300))
    print(f"✔ A4 comparison saved: {comp_path}")
    return comp_path


def process_image(image_path, models):
    print(f"Processing: {image_path}")
    fname = os.path.basename(image_path)
    
    # 1. Load HR & Prep
    hr = Image.open(image_path).convert("L")
    hr_np = pil_to_np(hr)
    
    # Crop to be divisible by SCALE
    h, w = hr_np.shape
    h = (h // SCALE) * SCALE
    w = (w // SCALE) * SCALE
    hr_np = hr_np[:h, :w]
    
    # 2. Create LR
    lr_np = create_lr_bicubic(hr_np)
    lr_up = cv2.resize(lr_np, (w, h), interpolation=cv2.INTER_CUBIC) # For residual fix & visualization
    
    # 3. Run Models
    t_lr = tensor_from_np(lr_np)
    with torch.no_grad():
        out_e = tensor_to_np(models["edsr"](t_lr))
        out_g = tensor_to_np(models["esrgan"](t_lr))
        out_s = tensor_to_np(models["srcnn"](t_lr))
        
    # 4. Post-process & Fixes
    # Resize to match HR (just in case)
    out_e = cv2.resize(out_e, (w, h), interpolation=cv2.INTER_CUBIC)
    out_g = cv2.resize(out_g, (w, h), interpolation=cv2.INTER_CUBIC)
    out_s = cv2.resize(out_s, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # FIX: EDSR Residual
    out_e = out_e + lr_up
    out_e = np.clip(out_e, 0, 1)
    out_g = np.clip(out_g, 0, 1)
    out_s = np.clip(out_s, 0, 1)
    
    # 5. Ensembles
    
    # A. Pairwise & Global (Simple Average)
    ens_es = (out_e + out_s) / 2
    ens_eg = (out_e + out_g) / 2
    ens_sg = (out_s + out_g) / 2
    ens_global = (out_e + out_s + out_g) / 3
    
    # B. Patch-wise (Variance Based)
    # Calculate variance map on upscaled LR
    patch_size = 32
    ens_patch = np.zeros_like(lr_up)
    
    # We can do this efficiently with sliding window or simple loops
    # Using loops for clarity as in ensemble_final.py
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            patch_lr = lr_up[y:y_end, x:x_end]
            var = np.var(patch_lr)
            
            # Thresholds tuned for MRI (from ensemble_final.py)
            # < 0.001: Smooth -> SRCNN
            # < 0.005: Texture -> EDSR
            # >= 0.005: Edge -> ESRGAN
            if var < 0.001:
                ens_patch[y:y_end, x:x_end] = out_s[y:y_end, x:x_end]
            elif var < 0.005:
                ens_patch[y:y_end, x:x_end] = out_e[y:y_end, x:x_end]
            else:
                ens_patch[y:y_end, x:x_end] = out_g[y:y_end, x:x_end]
                
    # C. Frequency Fusion (FFT)
    # Low Freq: EDSR, High Freq: ESRGAN
    freq_radius = 80 # Larger radius for 4x scale
    
    def apply_fft(img): return np.fft.fftshift(np.fft.fft2(img))
    def apply_ifft(fshift): return np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    
    f_e = apply_fft(out_e)
    f_g = apply_fft(out_g)
    
    rows, cols = h, w
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), freq_radius, 1, -1)
    
    f_fused = f_e * mask + f_g * (1 - mask)
    ens_freq = apply_ifft(f_fused)
    ens_freq = np.clip(ens_freq, 0, 1)
    
    # 6. Metrics
    results = {}
    items = {
        "edsr": out_e, "esrgan": out_g, "srcnn": out_s,
        "ens_es": ens_es, "ens_eg": ens_eg, "ens_sg": ens_sg,
        "ens_global": ens_global,
        "ens_patch": ens_patch,
        "ens_freq": ens_freq
    }
    
    for key, img in items.items():
        p, s = compute_metrics(hr_np, img)
        results[f"psnr_{key}"] = p
        results[f"ssim_{key}"] = s
        
        # Save Image
        save_path = os.path.join(OUT_ROOT, key, fname)
        np_to_pil(img).save(save_path)

    # 7. A4-Ready Comparison Grid (High Quality, With PSNR + SSIM)
    labels = [
        f"LR\n-", 
        f"HR\n-",
        f"SRCNN\nPSNR:{results['psnr_srcnn']:.2f}\nSSIM:{results['ssim_srcnn']:.3f}",
        f"EDSR\nPSNR:{results['psnr_edsr']:.2f}\nSSIM:{results['ssim_edsr']:.3f}",
        f"ESRGAN\nPSNR:{results['psnr_esrgan']:.2f}\nSSIM:{results['ssim_esrgan']:.3f}",
        f"E+S\nPSNR:{results['psnr_ens_es']:.2f}\nSSIM:{results['ssim_ens_es']:.3f}",
        f"E+G\nPSNR:{results['psnr_ens_eg']:.2f}\nSSIM:{results['ssim_ens_eg']:.3f}",
        f"S+G\nPSNR:{results['psnr_ens_sg']:.2f}\nSSIM:{results['ssim_ens_sg']:.3f}",
        f"Global\nPSNR:{results['psnr_ens_global']:.2f}\nSSIM:{results['ssim_ens_global']:.3f}",
        f"Patch\nPSNR:{results['psnr_ens_patch']:.2f}\nSSIM:{results['ssim_ens_patch']:.3f}",
        f"Freq\nPSNR:{results['psnr_ens_freq']:.2f}\nSSIM:{results['ssim_ens_freq']:.3f}"
    ]

    grid = [
        (lr_up, labels[0]),
        (hr_np, labels[1]),
        (out_s, labels[2]),
        (out_e, labels[3]),
        (out_g, labels[4]),
        (ens_es, labels[5]),
        (ens_eg, labels[6]),
        (ens_sg, labels[7]),
        (ens_global, labels[8]),
        (ens_patch, labels[9]),
        (ens_freq, labels[10])
    ]

    create_a4_grid(grid, w, h, fname)

    return results
        
    # 7. A4-Ready Comparison Grid (High Quality, With PSNR + SSIM)






# ------------------ MAIN ------------------------
def main():
    models = {
        "edsr": load_edsr(),
        "esrgan": load_esrgan(),
        "srcnn": load_srcnn()
    }
    
    # Collect files
    files = []
    for root, _, fnames in os.walk(HR_ROOT_FOLDER):
        for f in fnames:
            if f.lower().endswith(VALID_EXT):
                files.append(os.path.join(root, f))
    files.sort()
    
    if not files:
        print(f"No images found in {HR_ROOT_FOLDER}")
        return

    # CSV
    csv_path = os.path.join(OUT_ROOT, "metrics_combinations.csv")
    fieldnames = ["filename"]
    # Add all metric keys dynamically from first result
    # But we know them: psnr_edsr, ssim_edsr, ...
    # Let's just run one to get keys or define them.
    # Defining order is nicer.
    keys = ["edsr", "esrgan", "srcnn", "ens_es", "ens_eg", "ens_sg", "ens_global", "ens_patch", "ens_freq"]
    metric_cols = []
    for k in keys:
        metric_cols.extend([f"psnr_{k}", f"ssim_{k}"])
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + metric_cols)
        writer.writeheader()
        
        for fpath in files:
            res = process_image(fpath, models)
            row = {"filename": os.path.basename(fpath)}
            row.update(res)
            writer.writerow(row)
            
    print(f"\n✔ All Done! Results in: {OUT_ROOT}")

if __name__ == "__main__":
    main()
