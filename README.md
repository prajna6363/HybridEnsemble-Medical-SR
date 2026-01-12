# Advanced MRI Super-Resolution Ensemble

This project implements an advanced ensemble of Super-Resolution models (SRCNN, EDSR, ESRGAN) for MRI images, featuring **Patch-wise Model Selection** and **Frequency-domain Fusion**.

## Ensemble Strategies Explained

This project employs three distinct ensemble strategies to maximize image quality by leveraging the strengths of different models.

### 1. Patch-wise Model Selection (Variance-Based)
No single model is perfect for every region of an image. This strategy dynamically selects the best model for each local region based on its texture complexity.

*   **Logic**:
    *   **Smooth Areas (Background/Soft Tissue)**: These regions are prone to noise amplification by aggressive models. We use **SRCNN**, which provides clean, smooth reconstruction.
    *   **Medium Texture**: Areas with some detail but no sharp edges. We use **EDSR**, which offers high fidelity and structural accuracy.
    *   **High Texture / Edges**: Areas requiring sharp definition. We use **ESRGAN**, which excels at hallucinating fine high-frequency details.
*   **Process Workflow**:
    1.  **Initialization**: Create an empty canvas `ens_patch` of the same dimensions as the upscaled image.
    2.  **Patch Iteration**: Loop through the image in **32x32 pixel blocks** (rows `y`, columns `x`).
    3.  **Variance Calculation**: For each patch, extract the corresponding region from the upscaled Low-Resolution (LR) image and calculate its **variance** (a statistical measure of contrast/texture).
    4.  **Selection & Assignment**:
        *   **If Variance < 0.001 (Smooth)**: Copy the patch from the **SRCNN** result.
        *   **If 0.001 <= Variance < 0.005 (Medium)**: Copy the patch from the **EDSR** result.
        *   **If Variance >= 0.005 (Textured)**: Copy the patch from the **ESRGAN** result.
    5.  **Reconstruction**: Repeat until the entire image is filled.

### 2. Frequency Domain Fusion (FFT-Based)
This method fuses images in the frequency domain rather than the spatial domain, combining the structural stability of one model with the sharp details of another.

*   **Logic**:
    *   **Low Frequencies (Structure/Color/Shapes)**: Taken from **EDSR**. EDSR has high PSNR and preserves the correct underlying structure of the image.
    *   **High Frequencies (Fine Detail/Noise/Edges)**: Taken from **ESRGAN**. ESRGAN generates realistic high-frequency textures that EDSR might blur out.
*   **Process Workflow**:
    1.  **FFT Conversion**: Convert both **EDSR** and **ESRGAN** output images into the frequency domain using the 2D Fast Fourier Transform (FFT).
    2.  **Shift**: Shift the zero-frequency component to the center of the spectrum for easier filtering.
    3.  **Mask Creation**: Generate a **Low Pass Filter (LPF)** mask, which is a white circle (value 1) of radius **80** on a black background (value 0).
    4.  **Fusion**:
        *   **Low Frequency Component**: Multiply the **EDSR** spectrum by the mask (keeping only the center).
        *   **High Frequency Component**: Multiply the **ESRGAN** spectrum by the inverse of the mask (keeping only the outer edges).
        *   **Combine**: Add the two filtered spectra together.
    5.  **Reconstruction**: Apply the Inverse FFT (IFFT) to convert the fused spectrum back into a spatial image.

### 3. Global Weighted Ensemble
A simple but effective baseline that averages the pixel values of multiple models to cancel out individual model errors.

*   **Process Workflow**:
    1.  **Pixel-wise Summation**: For every pixel at position `(x, y)`, sum the intensity values from all contributing models (e.g., `Value = SRCNN(x,y) + EDSR(x,y) + ESRGAN(x,y)`).
    2.  **Averaging**: Divide the sum by the number of models (e.g., `Result = Value / 3`).
    3.  **Normalization**: Ensure the final value is within the valid image range [0, 255].
*   **Benefit**: Often achieves the highest PSNR by smoothing out random noise from individual models, though it may be slightly blurrier than the patch-based or frequency-based methods.

## Scripts

### 1. `ensemble_final.py` (Scale x2)
Runs the full pipeline on the sample images in `10_IMAGES/`.
- **Input**: `10_IMAGES/`
- **Output**: `results_x2/`
- **Usage**:
  ```bash
  python ensemble_final.py
  ```

### 2. `comparision_bicubic.py` (Scale x4)
Runs the full pipeline on the full dataset in `brain_mri_scan_images/`.
- **Input**: `brain_mri_scan_images/`
- **Output**: `comparison_final_results/`
- **Usage**:
  ```bash
  python comparision_bicubic.py
  ```

### 3. `ensemble_combinations.py` (Comprehensive)
**Best for Research/Analysis**. Runs ALL methods (Individual, Pairwise, Global, Patch, Freq) and generates a mega-comparison grid.
- **Input**: `10_IMAGES/`
- **Output**: `results_combinations/`
- **Usage**:
  ```bash
  python ensemble_combinations.py
  ```

### 4. `metrics.py`
Evaluates the results against Ground Truth (GT).
- **Usage**:
  ```bash
  python metrics.py
  ```
  *Note: By default, it evaluates `results_x2`. Edit `ROOT_RESULTS` in the script to evaluate `comparison_final_results`.*

## Results Structure
The output directories contain subfolders for each method:
- `srcnn/`, `edsr/`, `esrgan/`: Individual model outputs.
- `ensemble/`: Global average ensemble.
- `ensemble_patch/`: Patch-wise selection result.
- `ensemble_freq/`: Frequency fusion result.
- `comparisons/`: Side-by-side comparison grids.
