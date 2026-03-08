# final_ensemble_best.py
# Selects the BEST result among all individual models and ensembles.
# Generates: Best Image, Comparison Grid, and Metrics CSV.

import os
import csv
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from models.edsr import edsr_x
from models.rrdb import RRDBNet
from models.srcnn import MedicalSRCNN_Plus as SRCNN

# ------------------ CONFIG ---------------------
HR_ROOT_FOLDER = "10_IMAGES" 
OUT_ROOT = "results_best"
SCALE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# Ensure output directories exist
DIRS = [
    "best_images",
    "comparisons",
    "edsr", "esrgan", "srcnn"
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
    lr_up = cv2.resize(lr_np, (w, h), interpolation=cv2.INTER_CUBIC) 
    
    # 3. Run Models
    t_lr = tensor_from_np(lr_np)
    with torch.no_grad():
        out_e = tensor_to_np(models["edsr"](t_lr))
        out_g = tensor_to_np(models["esrgan"](t_lr))
        out_s = tensor_to_np(models["srcnn"](t_lr))
        
    # 4. Post-process & Fixes
    out_e = cv2.resize(out_e, (w, h), interpolation=cv2.INTER_CUBIC)
    out_g = cv2.resize(out_g, (w, h), interpolation=cv2.INTER_CUBIC)
    out_s = cv2.resize(out_s, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # FIX: EDSR Residual
    out_e = out_e + lr_up
    out_e = np.clip(out_e, 0, 1)
    out_g = np.clip(out_g, 0, 1)
    out_s = np.clip(out_s, 0, 1)
    
    # 5. Ensembles
    
    # A. Pairwise & Global
    ens_es = (out_e + out_s) / 2
    ens_eg = (out_e + out_g) / 2
    ens_sg = (out_s + out_g) / 2
    ens_global = (out_e + out_s + out_g) / 3
    
    # B. Patch-wise (Variance Based)
    patch_size = 32
    ens_patch = np.zeros_like(lr_up)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch_lr = lr_up[y:y_end, x:x_end]
            var = np.var(patch_lr)
            if var < 0.001:
                ens_patch[y:y_end, x:x_end] = out_s[y:y_end, x:x_end]
            elif var < 0.005:
                ens_patch[y:y_end, x:x_end] = out_e[y:y_end, x:x_end]
            else:
                ens_patch[y:y_end, x:x_end] = out_g[y:y_end, x:x_end]
                
    # C. Frequency Fusion (FFT)
    freq_radius = 80 
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
    
    # 6. Metrics & Selection
    candidates = {
        "EDSR": out_e, "ESRGAN": out_g, "SRCNN": out_s,
        "Ens_ES": ens_es, "Ens_EG": ens_eg, "Ens_SG": ens_sg,
        "Ens_Global": ens_global,
        "Ens_Patch": ens_patch,
        "Ens_Freq": ens_freq
    }
    
    best_name = ""
    best_psnr = -1.0
    best_ssim = -1.0
    best_img = None
    
    results = {}
    
    for name, img in candidates.items():
        p, s = compute_metrics(hr_np, img)
        results[f"psnr_{name}"] = p
        results[f"ssim_{name}"] = s
        
        if p > best_psnr:
            best_psnr = p
            best_ssim = s
            best_name = name
            best_img = img
            
    # Save Individual Images
    np_to_pil(out_e).save(os.path.join(OUT_ROOT, "edsr", fname))
    np_to_pil(out_g).save(os.path.join(OUT_ROOT, "esrgan", fname))
    np_to_pil(out_s).save(os.path.join(OUT_ROOT, "srcnn", fname))

    # Save Best Image
    best_pil = np_to_pil(best_img)
    best_pil.save(os.path.join(OUT_ROOT, "best_images", fname))
    
    # 7. Comparison Grid (A4 Layout)
    # A4 at 150 DPI is approx 1240 x 1754 pixels
    # We have 6 images: LR, HR, EDSR, ESRGAN, SRCNN, BEST
    # Layout: 3 rows x 2 columns
    # Row 1: LR, HR
    # Row 2: EDSR, ESRGAN
    # Row 3: SRCNN, BEST
    
    # Define A4 dimensions (Portrait)
    A4_W, A4_H = 1240, 1754
    comp = Image.new("RGB", (A4_W, A4_H), (255, 255, 255)) # White bg
    draw = ImageDraw.Draw(comp)
    
    try:
        # Load a larger font for A4
        font = ImageFont.truetype("arial.ttf", 40)
        title_font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Prepare images and labels
    # Resize images to fit in the grid cells
    # Grid cell size
    cell_w = A4_W // 2
    cell_h = A4_H // 3
    
    # Padding
    pad = 20
    img_max_w = cell_w - 2 * pad
    img_max_h = cell_h - 100 # Space for text
    
    items = [
        (lr_up, "LR (Bicubic)"),
        (hr_np, "HR (Ground Truth)"),
        (out_e, f"EDSR\nPSNR: {results['psnr_EDSR']:.2f} | SSIM: {results['ssim_EDSR']:.4f}"),
        (out_g, f"ESRGAN\nPSNR: {results['psnr_ESRGAN']:.2f} | SSIM: {results['ssim_ESRGAN']:.4f}"),
        (out_s, f"SRCNN\nPSNR: {results['psnr_SRCNN']:.2f} | SSIM: {results['ssim_SRCNN']:.4f}"),
        (best_img, f"BEST: {best_name}\nPSNR: {best_psnr:.2f} | SSIM: {best_ssim:.4f}")
    ]
    
    for idx, (img, label) in enumerate(items):
        # Convert to PIL
        pil_img = np_to_pil(img)
        
        # Resize to fit while maintaining aspect ratio
        pil_img.thumbnail((img_max_w, img_max_h), Image.Resampling.LANCZOS)
        
        # Calculate position
        row = idx // 2
        col = idx % 2
        
        x_center = col * cell_w + cell_w // 2
        y_center = row * cell_h + cell_h // 2
        
        # Paste image centered in cell (shifted up slightly for text)
        img_w, img_h = pil_img.size
        x_pos = x_center - img_w // 2
        y_pos = y_center - img_h // 2 - 30 
        
        comp.paste(pil_img, (x_pos, y_pos))
        
        # Draw Label
        # Calculate text size roughly to center it
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_center - text_w // 2
        text_y = y_pos + img_h + 10
        
        draw.multiline_text((text_x, text_y), label, fill=(0, 0, 0), font=font, align="center")

    # Add Title
    draw.text((20, 20), f"Comparison: {fname}", fill=(0, 0, 0), font=title_font)

    comp_path = os.path.join(OUT_ROOT, "comparisons", fname.replace(".", "_cmp."))
    comp.save(comp_path)
    
    return {
        "filename": fname,
        "best_method": best_name,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        **results
    }

# ------------------ MAIN ------------------------
def main():
    models = {
        "edsr": load_edsr(),
        "esrgan": load_esrgan(),
        "srcnn": load_srcnn()
    }
    
    files = []
    for root, _, fnames in os.walk(HR_ROOT_FOLDER):
        for f in fnames:
            if f.lower().endswith(VALID_EXT):
                files.append(os.path.join(root, f))
    files.sort()
    
    if not files:
        print(f"No images found in {HR_ROOT_FOLDER}")
        return

    csv_path = os.path.join(OUT_ROOT, "metrics_best.csv")
    
    # Determine fieldnames from first result
    # We can just run one to get keys, but let's define them to be safe
    # Actually, let's process the first one and then write header
    
    all_results = []
    print("Starting processing...")
    
    for fpath in files:
        res = process_image(fpath, models)
        all_results.append(res)
        
    if not all_results:
        return
        
    fieldnames = list(all_results[0].keys())
    # Ensure filename is first, bests are next
    priority = ["filename", "best_method", "best_psnr", "best_ssim"]
    fieldnames = priority + [k for k in fieldnames if k not in priority]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
            
    print(f"\n✔ All Done! Results in: {OUT_ROOT}")

if __name__ == "__main__":
    main()
