import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --------------------------------------
# FOLDER PATHS
# --------------------------------------
# Adjust these if your folder structure is different
GT_POS = "brain_mri_scan_images"
# We will evaluate results from both Scale 2 (results_x2) and Scale 4 (comparison_final_results)
# For simplicity, let's focus on the Scale 2 results first as they are the most recent "perfect" run.
# If you want to evaluate Scale 4, change the root folder.

# Let's make this script flexible to evaluate any root folder
ROOT_RESULTS = "results_combinations" 

FOLDERS_TO_EVAL = {
    "EDSR": f"{ROOT_RESULTS}/edsr",
    "ESRGAN": f"{ROOT_RESULTS}/esrgan",
    "SRCNN": f"{ROOT_RESULTS}/srcnn",
    "Ensemble (E+S)": f"{ROOT_RESULTS}/ens_es",
    "Ensemble (E+G)": f"{ROOT_RESULTS}/ens_eg",
    "Ensemble (S+G)": f"{ROOT_RESULTS}/ens_sg",
    "Ensemble (Global)": f"{ROOT_RESULTS}/ens_global",
    "Ensemble (Patch)": f"{ROOT_RESULTS}/ens_patch",
    "Ensemble (Freq)": f"{ROOT_RESULTS}/ens_freq",
}

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# --------------------------------------
# FUNCTION: LOAD & SORT FILES
# --------------------------------------
def load_images_sorted(folder):
    if not os.path.exists(folder):
        print(f"⚠ Warning: Folder not found: {os.path.abspath(folder)}")
        return [], []
        
    all_files = []
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            if fname.lower().endswith(VALID_EXT):
                all_files.append(os.path.join(root, fname))
    all_files.sort()

    images = []
    names = []
    for f_path in all_files:
        img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        images.append(img)
        names.append(os.path.basename(f_path))
        
    return images, names

# --------------------------------------
# FUNCTION: COMPUTE METRICS
# --------------------------------------
def evaluate(gt_folder, sr_folder, label):
    print(f"Evaluating {label}...")
    
    # Load GT
    # We need to find matching GT for each SR file
    # Since GT folder structure might be recursive (positive/negative), and SR might be flat or recursive
    # The safest way is to map filename -> full path
    
    gt_map = {}
    for root, _, fnames in os.walk(gt_folder):
        for fname in fnames:
            if fname.lower().endswith(VALID_EXT):
                gt_map[fname] = os.path.join(root, fname)
                
    sr_imgs, sr_names = load_images_sorted(sr_folder)
    
    if not sr_imgs:
        print(f"  -> No images found in {sr_folder}")
        return 0.0, 0.0

    psnr_list = []
    ssim_list = []
    
    for i, name in enumerate(sr_names):
        if name not in gt_map:
            # Try finding it in 10_IMAGES if not in brain_mri_scan_images
            # Or just skip
            continue
            
        gt_path = gt_map[name]
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = gt.astype(np.float32) / 255.0
        
        sr = sr_imgs[i]
        
        # Ensure sizes match
        # If SR is smaller (due to cropping for scale), crop GT to match
        h_sr, w_sr = sr.shape
        h_gt, w_gt = gt.shape
        
        if h_sr != h_gt or w_sr != w_gt:
            gt = gt[:h_sr, :w_sr]
            
        p = psnr(gt, sr, data_range=1.0)
        s = ssim(gt, sr, data_range=1.0)
        
        psnr_list.append(p)
        ssim_list.append(s)
        
    if not psnr_list:
        print("  -> No matching GT files found.")
        return 0.0, 0.0
        
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"  -> PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

# --------------------------------------
# MAIN
# --------------------------------------
def main():
    # We need to know which GT folder to use.
    # For results_x2, the GT was 10_IMAGES
    # For comparison_final_results, the GT was brain_mri_scan_images
    
    # Let's try to detect based on ROOT_RESULTS
    if "x2" in ROOT_RESULTS or "combinations" in ROOT_RESULTS:
        GT_FOLDER = "10_IMAGES"
    else:
        GT_FOLDER = "brain_mri_scan_images"
        
    print(f"Using GT Folder: {GT_FOLDER}")
    print(f"Using Results Folder: {ROOT_RESULTS}\n")
    
    results = []
    
    for label, folder in FOLDERS_TO_EVAL.items():
        p, s = evaluate(GT_FOLDER, folder, label)
        results.append((label, p, s))
        
    print("\n========== FINAL SUMMARY ==========")
    print(f"{'Method':<20} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 46)
    for label, p, s in results:
        print(f"{label:<20} | {p:<10.4f} | {s:<10.4f}")
    print("===================================")

if __name__ == "__main__":
    main()
