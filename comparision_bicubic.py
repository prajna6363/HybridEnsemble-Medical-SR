# comparison_final_bicubic.py
# Final PSNR-boosted evaluation script (bicubic LR only)

import os
import csv
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from skimage.filters import sobel

from models.edsr import edsr_x
from models.rrdb import RRDBNet
from models.srcnn import MedicalSRCNN_Plus as SRCNN

# 
# CONFIG
# 
HR_ROOT_FOLDER = "10_IMAGES"
OUT_ROOT = "comparison_final_results"

SCALE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")
PATCH_SIZE = 64 # Define patch size

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "edsr"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "esrgan"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "srcnn"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "ensemble"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "ensemble_edsr_srcnn"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "ensemble_esrgan_srcnn"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "ensemble_all"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "comparisons"), exist_ok=True)


# 
# LOAD MODELS (robust)
# -------------------------------------------
def load_edsr(pth="weights/edsr.pth"):
    model = edsr_x(scale=SCALE, n_resblocks=16, n_feats=64, n_colors=1).to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"✔ EDSR loaded. SCALE={SCALE}, model.scale={model.scale}")
    # Check upsampler length
    print(f"  Upsampler length: {len(model.upsampler.upsample)}")
    return model

def load_esrgan(pth="weights/esrgan.pth"):
    model = RRDBNet(in_nc=1, out_nc=1, nf=16, nb=1, gc=8, scale=SCALE).to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("✔ ESRGAN loaded")
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


# -------------------------------------------
# IMAGE HELPERS
# -------------------------------------------
def pil_to_np(img):
    return np.array(img).astype(np.float32) / 255.0

def np_to_pil(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))

def tensor_from_np(x):
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return t

def tensor_to_np(t):
    arr = t.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(arr, 0, 1)


# -------------------------------------------
# PATCH PROCESSING HELPERS
# -------------------------------------------
def split_image_into_patches(image_np, patch_size):
    h, w = image_np.shape
    patches = []
    coords = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_np[i:min(i + patch_size, h), j:min(j + patch_size, w)]
            patches.append(patch)
            coords.append((i, j, min(i + patch_size, h), min(j + patch_size, w)))
    return patches, coords

def combine_patches_into_image(patches, coords, original_shape):
    h, w = original_shape
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    for patch, (y1, x1, y2, x2) in zip(patches, coords):
        reconstructed_image[y1:y2, x1:x2] = patch
    return reconstructed_image


def calculate_quality_score(image_patch):
    # Use Sobel filter to calculate gradient magnitude as a quality score
    # Higher gradient magnitude often indicates sharper edges/details
    return np.mean(np.abs(sobel(image_patch)))


# CORRECT LR GENERATION (KEY FIX!)
def create_lr_bicubic(hr_np):
    """Correct LR generation exactly matching training pipeline."""
    h, w = hr_np.shape
    lr = cv2.resize(hr_np, (w//SCALE, h//SCALE), interpolation=cv2.INTER_CUBIC)
    return lr


# -------------------------------------------
# METRICS
# -------------------------------------------
def compute_metrics(hr, pred):
    hr = np.clip(hr, 0, 1)
    pred = np.clip(pred, 0, 1)
    psnr = sk_psnr(hr, pred, data_range=1.0)
    ssim = sk_ssim(hr, pred, data_range=1.0)
    # guard against NaN
    if np.isnan(psnr): psnr = 0.0
    if np.isnan(ssim): ssim = 0.0
    return float(psnr), float(ssim)


def process_image(image_path, models):
    print("Processing:", image_path)

    # Load HR
    hr = Image.open(image_path).convert("L")
    hr_np = pil_to_np(hr)

    # Make HR divisible by scale
    h4 = (hr_np.shape[0] // SCALE) * SCALE
    w4 = (hr_np.shape[1] // SCALE) * SCALE
    hr_np = hr_np[:h4, :w4]

    # ---- Correct LR generation ----
    lr_np = create_lr_bicubic(hr_np)

    # Upsampled for visual
    lr_up = cv2.resize(lr_np, (w4, h4), interpolation=cv2.INTER_CUBIC)

    # ---- Run SR models ----
    t_lr = tensor_from_np(lr_np)
    with torch.no_grad():
        sr_e = tensor_to_np(models["edsr"](t_lr))
        sr_g = tensor_to_np(models["esrgan"](t_lr))
        sr_s = tensor_to_np(models["srcnn"](t_lr))

    # ---- Ensuring correct size ----
    sr_e = cv2.resize(sr_e, (w4, h4), interpolation=cv2.INTER_CUBIC)
    
    # FIX: EDSR outputs residual
    sr_e = sr_e + lr_up
    sr_e = np.clip(sr_e, 0, 1)

    sr_g = cv2.resize(sr_g, (w4, h4), interpolation=cv2.INTER_CUBIC)
    sr_g = cv2.resize(sr_g, (w4, h4), interpolation=cv2.INTER_CUBIC)
    sr_s = cv2.resize(sr_s, (w4, h4), interpolation=cv2.INTER_CUBIC)

    # ---- Patch-wise Model Selection Ensemble ----
    # Split LR image into patches
    lr_patches, lr_coords = split_image_into_patches(lr_np, PATCH_SIZE)

    # Prepare lists to store SR patches from each model
    edsr_sr_patches = []
    esrgan_sr_patches = []
    srcnn_sr_patches = []

    # Process each patch
    for lr_patch_np in lr_patches:
        # Convert patch to tensor for model input
        lr_patch_tensor = tensor_from_np(lr_patch_np)

        # Run each model on the patch
        with torch.no_grad():
            edsr_sr_patch_tensor = models["edsr"](lr_patch_tensor)
            esrgan_sr_patch_tensor = models["esrgan"](lr_patch_tensor)
            srcnn_sr_patch_tensor = models["srcnn"](lr_patch_tensor)
            
            # FIX: SRCNN might be 2x, resize to match others (4x) if needed
            if srcnn_sr_patch_tensor.shape[2] != edsr_sr_patch_tensor.shape[2]:
                srcnn_sr_patch_tensor = torch.nn.functional.interpolate(srcnn_sr_patch_tensor, size=edsr_sr_patch_tensor.shape[2:], mode='bicubic', align_corners=False)

        # Convert SR patches back to numpy
        edsr_sr_patches.append(tensor_to_np(edsr_sr_patch_tensor))
        esrgan_sr_patches.append(tensor_to_np(esrgan_sr_patch_tensor))
        srcnn_sr_patches.append(tensor_to_np(srcnn_sr_patch_tensor))

    # Initialize list for selected patches
    selected_sr_patches = [np.zeros_like(p) for p in edsr_sr_patches] # Use any patch list for shape

    # Select best patch for each coordinate
    for i, (edsr_patch, esrgan_patch, srcnn_patch) in enumerate(zip(edsr_sr_patches, esrgan_sr_patches, srcnn_sr_patches)):
        edsr_score = calculate_quality_score(edsr_patch)
        esrgan_score = calculate_quality_score(esrgan_patch)
        srcnn_score = calculate_quality_score(srcnn_patch)

        if edsr_score >= esrgan_score and edsr_score >= srcnn_score:
            selected_sr_patches[i] = edsr_patch
        elif esrgan_score >= edsr_score and esrgan_score >= srcnn_score:
            selected_sr_patches[i] = esrgan_patch
        else:
            selected_sr_patches[i] = srcnn_patch

    # Scale coords for SR
    sr_coords = []
    for (y1, x1, y2, x2) in lr_coords:
        sr_coords.append((y1*SCALE, x1*SCALE, y2*SCALE, x2*SCALE))

    # Combine selected patches into final SR image
    ensemble_all = combine_patches_into_image(selected_sr_patches, sr_coords, hr_np.shape)
    ensemble_all = np.clip(ensemble_all, 0.0, 1.0)

    # ---- Frequency Fusion (FFT) ----
    # LF from EDSR, HF from ESRGAN
    # Radius needs to be larger for 4x scale (images are ~1000x1000?)
    # If 2x was 40 for 512, 4x might be 80 for 1024.
    freq_radius = 80
    
    def apply_fft(img):
        return np.fft.fftshift(np.fft.fft2(img))
        
    def apply_ifft(fshift):
        return np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
        
    f_e = apply_fft(sr_e)
    f_g = apply_fft(sr_g)
    
    rows, cols = sr_e.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), freq_radius, 1, -1)
    
    f_fused = f_e * mask + f_g * (1 - mask)
    ensemble_freq = apply_ifft(f_fused)
    ensemble_freq = np.clip(ensemble_freq, 0, 1)

    # Process full images for individual models (for comparison)
    # These are already computed as sr_e, sr_g, sr_s
    # No need to re-run models on full image if we are doing patch-wise for ensemble_all

    # For other ensembles, we can still use simple averaging of full images
    ensemble_edsr_esrgan = 0.5 * sr_e + 0.5 * sr_g
    ensemble_edsr_esrgan = np.clip(ensemble_edsr_esrgan, 0.0, 1.0)

    ensemble_edsr_srcnn = 0.5 * sr_e + 0.5 * sr_s
    ensemble_edsr_srcnn = np.clip(ensemble_edsr_srcnn, 0.0, 1.0)

    ensemble_esrgan_srcnn = 0.5 * sr_g + 0.5 * sr_s
    ensemble_esrgan_srcnn = np.clip(ensemble_esrgan_srcnn, 0.0, 1.0)

    # ---- Metrics ----
    ps_e, ss_e = compute_metrics(hr_np, sr_e)
    ps_g, ss_g = compute_metrics(hr_np, sr_g)
    ps_s, ss_s = compute_metrics(hr_np, sr_s)
    ps_ees, ss_ees = compute_metrics(hr_np, ensemble_edsr_esrgan)
    ps_es, ss_es = compute_metrics(hr_np, ensemble_edsr_srcnn)
    ps_gs, ss_gs = compute_metrics(hr_np, ensemble_esrgan_srcnn)
    ps_all, ss_all = compute_metrics(hr_np, ensemble_all)
    ps_freq, ss_freq = compute_metrics(hr_np, ensemble_freq)

    # Save images
    # Extract relative path for output directories
    relative_path = os.path.relpath(image_path, HR_ROOT_FOLDER)
    output_dir_edsr = os.path.join(OUT_ROOT, "edsr", os.path.dirname(relative_path))
    output_dir_esrgan = os.path.join(OUT_ROOT, "esrgan", os.path.dirname(relative_path))
    output_dir_srcnn = os.path.join(OUT_ROOT, "srcnn", os.path.dirname(relative_path))
    output_dir_ens_ees = os.path.join(OUT_ROOT, "ensemble", os.path.dirname(relative_path))
    output_dir_ens_es = os.path.join(OUT_ROOT, "ensemble_edsr_srcnn", os.path.dirname(relative_path))
    output_dir_ens_gs = os.path.join(OUT_ROOT, "ensemble_esrgan_srcnn", os.path.dirname(relative_path))
    output_dir_ens_all = os.path.join(OUT_ROOT, "ensemble_all", os.path.dirname(relative_path))
    output_dir_ens_freq = os.path.join(OUT_ROOT, "ensemble_freq", os.path.dirname(relative_path))
    output_dir_comparisons = os.path.join(OUT_ROOT, "comparisons", os.path.dirname(relative_path))

    os.makedirs(output_dir_edsr, exist_ok=True)
    os.makedirs(output_dir_esrgan, exist_ok=True)
    os.makedirs(output_dir_srcnn, exist_ok=True)
    os.makedirs(output_dir_ens_ees, exist_ok=True)
    os.makedirs(output_dir_ens_es, exist_ok=True)
    os.makedirs(output_dir_ens_gs, exist_ok=True)
    os.makedirs(output_dir_ens_all, exist_ok=True)
    os.makedirs(output_dir_ens_freq, exist_ok=True)
    os.makedirs(output_dir_comparisons, exist_ok=True)

    out_edsr = os.path.join(output_dir_edsr, os.path.basename(image_path))
    out_esr = os.path.join(output_dir_esrgan, os.path.basename(image_path))
    out_srcnn = os.path.join(output_dir_srcnn, os.path.basename(image_path))
    out_ens_ees = os.path.join(output_dir_ens_ees, os.path.basename(image_path))
    out_ens_es = os.path.join(output_dir_ens_es, os.path.basename(image_path))
    out_ens_gs = os.path.join(output_dir_ens_gs, os.path.basename(image_path))
    out_ens_all = os.path.join(output_dir_ens_all, os.path.basename(image_path))
    out_ens_freq = os.path.join(output_dir_ens_freq, os.path.basename(image_path))

    np_to_pil(sr_e).save(out_edsr)
    np_to_pil(sr_g).save(out_esr)
    np_to_pil(sr_s).save(out_srcnn)
    np_to_pil(ensemble_edsr_esrgan).save(out_ens_ees)
    np_to_pil(ensemble_edsr_srcnn).save(out_ens_es)
    np_to_pil(ensemble_esrgan_srcnn).save(out_ens_gs)
    np_to_pil(ensemble_all).save(out_ens_all)
    np_to_pil(ensemble_freq).save(out_ens_freq)

    # ---- Build comparison grid ----
    w = w4
    h = h4
    gap = 8
    comp = Image.new("L", (w*9 + gap*8, h), 0)
    imgs = [lr_up, hr_np, sr_e, sr_g, sr_s, ensemble_edsr_esrgan, ensemble_edsr_srcnn, ensemble_esrgan_srcnn, ensemble_all, ensemble_freq]
    captions = [
        "LR (Bicubic)",
        "HR",
        f"EDSR\nPSNR={ps_e:.2f}\nSSIM={ss_e:.3f}",
        f"ESRGAN\nPSNR={ps_g:.2f}\nSSIM={ss_g:.3f}",
        f"SRCNN\nPSNR={ps_s:.2f}\nSSIM={ss_s:.3f}",
        f"ENS (E+G)\nPSNR={ps_ees:.2f}\nSSIM={ss_ees:.3f}",
        f"ENS (E+S)\nPSNR={ps_es:.2f}\nSSIM={ss_es:.3f}",
        f"ENS (G+S)\nPSNR={ps_gs:.2f}\nSSIM={ss_gs:.3f}",
        f"ENS (ALL)\nPSNR={ps_all:.2f}\nSSIM={ss_all:.3f}",
        f"ENS (FREQ)\nPSNR={ps_freq:.2f}\nSSIM={ss_freq:.3f}"
    ]

    x = 0
    for img_np, label in zip(imgs, captions):
        im = np_to_pil(img_np)
        comp.paste(im, (x, 0))
        drawer = ImageDraw.Draw(comp)
        drawer.text((x+5, 5), label, fill=255)
        x += w + gap

    # ---------- FIXED SAVE: construct valid file name ----------
    base = os.path.splitext(os.path.basename(image_path))[0]
    comp_out = os.path.join(output_dir_comparisons, base + "_cmp.png")
    comp.save(comp_out)

    print(f"Saved comparison for {image_path} -> {comp_out}")

    return ps_e, ss_e, ps_g, ss_g, ps_s, ss_s, ps_ees, ss_ees, ps_es, ss_es, ps_gs, ss_gs, ps_all, ss_all, ps_freq, ss_freq


# -------------------------------------------
# MAIN
# -------------------------------------------
def main():
    models = {
        "edsr": load_edsr(),
        "esrgan": load_esrgan(),
        "srcnn": load_srcnn()
    }

    files = []
    for root, _, fnames in os.walk(HR_ROOT_FOLDER):
        for fname in fnames:
            if fname.lower().endswith(VALID_EXT):
                files.append(os.path.join(root, fname))
    files.sort()
    if not files:
        print("No images found in", HR_ROOT_FOLDER)
        return

    csv_path = os.path.join(OUT_ROOT, "metrics_final_bicubic.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename","psnr_edsr","ssim_edsr","psnr_esrgan","ssim_esrgan","psnr_srcnn","ssim_srcnn","psnr_ensemble_edsr_esrgan","ssim_ensemble_edsr_esrgan","psnr_ensemble_edsr_srcnn","ssim_ensemble_edsr_srcnn","psnr_ensemble_esrgan_srcnn","ssim_ensemble_esrgan_srcnn","psnr_ensemble_all","ssim_ensemble_all","psnr_freq","ssim_freq"])

        for fname in files:
            ps_e, ss_e, ps_g, ss_g, ps_s, ss_s, ps_ees, ss_ees, ps_es, ss_es, ps_gs, ss_gs, ps_all, ss_all, ps_freq, ss_freq = process_image(fname, models)
            writer.writerow([fname, ps_e, ss_e, ps_g, ss_g, ps_s, ss_s, ps_ees, ss_ees, ps_es, ss_es, ps_gs, ss_gs, ps_all, ss_all, ps_freq, ss_freq])

    print("All done! Check:", OUT_ROOT)


if __name__ == "__main__":
    main()
