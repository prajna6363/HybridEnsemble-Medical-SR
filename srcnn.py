import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Enhanced Channel Attention for Medical Images ---
class MedicalChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):  # Reduced reduction for medical features
        super(MedicalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling
        
        # Enhanced FC layers with medical-specific design (removed BatchNorm for inference)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out  # Combined attention
        return x * self.sigmoid(out)

# --- Enhanced Spatial Attention for Medical Details ---
class MedicalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(MedicalSpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

# --- Enhanced CBAM for Medical Applications ---
class MedicalCBAM(nn.Module):
    def __init__(self, in_channels):
        super(MedicalCBAM, self).__init__()
        self.ca = MedicalChannelAttention(in_channels)
        self.sa = MedicalSpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# --- Medical Image Preprocessing Layer ---
class MedicalPreprocess(nn.Module):
    def __init__(self):
        super(MedicalPreprocess, self).__init__()
        # Normalization for medical images (typically 0-1 range)
        self.normalize = nn.BatchNorm2d(1)
        
    def forward(self, x):
        # Ensure proper normalization for medical images
        return self.normalize(x)

# --- Enhanced SRCNN_Plus for Medical Applications ---
class MedicalSRCNN_Plus(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1):
        super(MedicalSRCNN_Plus, self).__init__()
        
        # Medical image preprocessing (simplified for inference)
        # self.preprocess = MedicalPreprocess()  # Commented out for inference
        
        # Enhanced upsampling with medical-specific interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        
        # Enhanced feature extraction with medical focus
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional conv layer
            nn.ReLU(inplace=True),
        )

        # Medical CBAM attention
        self.cbam = MedicalCBAM(64)

        # Enhanced mapping layers
        self.mapping = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv layer
            nn.ReLU(inplace=True),
        )

        # Enhanced reconstruction with medical focus
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_channels, kernel_size=5, padding=2),
        )
        
        # Initialize weights for better medical performance
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for medical images
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Medical preprocessing (simplified)
        # x = self.preprocess(x)  # Commented out for inference

        # Upsampling
        x = self.upsample(x)
        residual = x

        # Feature extraction
        x = self.feature_extraction(x)

        # Attention mechanism
        x = self.cbam(x)

        # Mapping
        x = self.mapping(x)

        # Reconstruction
        x = self.reconstruction(x)

        # Residual (skip) connection to improve identity mapping and PSNR
        return residual + x

# --- Medical Image Quality Metrics ---
class MedicalImageMetrics:
    @staticmethod
    def calculate_psnr(img1, img2, max_val=1.0):
        """Calculate PSNR for medical images"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(img1, img2, window_size=11):
        """Calculate SSIM for medical images"""
        # Simplified SSIM calculation for medical images
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

# --- Medical Image Loss Functions ---
class MedicalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(MedicalLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # SSIM weight
        self.gamma = gamma  # Edge preservation weight
        
    def forward(self, sr, hr):
        # MSE Loss
        mse_loss = F.mse_loss(sr, hr)
        
        # SSIM Loss
        ssim_loss = 1 - MedicalImageMetrics.calculate_ssim(sr, hr)
        
        # Edge preservation loss for medical details
        edge_loss = self._edge_loss(sr, hr)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * edge_loss
        
        return total_loss
    
    def _edge_loss(self, sr, hr):
        """Edge preservation loss for medical image details"""
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(sr.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(sr.device)
        
        # Edge maps
        sr_edge_x = F.conv2d(sr, sobel_x, padding=1)
        sr_edge_y = F.conv2d(sr, sobel_y, padding=1)
        sr_edge = torch.sqrt(sr_edge_x**2 + sr_edge_y**2)
        
        hr_edge_x = F.conv2d(hr, sobel_x, padding=1)
        hr_edge_y = F.conv2d(hr, sobel_y, padding=1)
        hr_edge = torch.sqrt(hr_edge_x**2 + hr_edge_y**2)
        
        return F.mse_loss(sr_edge, hr_edge)

# --- Medical Image Training Configuration ---
class MedicalTrainingConfig:
    def __init__(self):
        self.learning_rate = 5e-5  # Lower LR for high-resolution training stability
        self.batch_size = 4        # Smaller batch for high-resolution images (1024x1024)
        self.epochs = 150          # More epochs for high-resolution medical accuracy
        self.loss_weights = {'mse': 0.6, 'ssim': 0.3, 'edge': 0.1}  # Enhanced SSIM weight
        self.scheduler_step = 25   # Adjusted for longer training
        self.scheduler_gamma = 0.7 # Gentler decay
        
    def get_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    
    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)

if __name__ == "__main__":
    # Test the medical model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MedicalSRCNN_Plus(scale_factor=2, num_channels=1).to(device)
    
    # Test input
    test_input = torch.randn(1, 1, 64, 64).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss function with properly sized tensors
    loss_fn = MedicalLoss()
    # Resize test_input to match output size for loss calculation
    test_input_resized = torch.nn.functional.interpolate(test_input, size=output.shape[-2:], mode='bilinear', align_corners=False)
    loss = loss_fn(output, test_input_resized)
    print(f"Test loss: {loss.item():.6f}")
    
    print("✅ Medical SRCNN+ model test completed successfully!")
