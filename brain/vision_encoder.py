import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class CognitiveVisionEncoder(nn.Module):
    def __init__(self, d_model=512):
        super(CognitiveVisionEncoder, self).__init__()
        # Use lightweight MobileNetV2 for CPU-friendly extraction
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Extract features (shape: [B, 1280, H/32, W/32])
        self.feature_extractor = mobilenet.features
        
        # Project 1280 channels to our cognitive d_model (512)
        self.visual_projection = nn.Linear(1280, d_model)
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images):
        """
        images: A batch of PIL Images or a Tensor of shape [B, 3, H, W]
        returns: visual_tokens of shape [B, num_patches, d_model]
        """
        if isinstance(images, list):
            device = next(self.parameters()).device
            tensors = [self.transform(img) for img in images]
            images = torch.stack(tensors).to(device)
            
        with torch.no_grad():
            # MobileNet features: [B, 1280, 7, 7]
            features = self.feature_extractor(images)
            
        B, C, H, W = features.shape
        # Flatten spatial dimensions: [B, C, H*W] -> [B, H*W, C]
        features = features.view(B, C, -1).transpose(1, 2)
        
        # Project to cognitive dimension: [B, 49, 512]
        visual_tokens = self.visual_projection(features)
        return visual_tokens

if __name__ == "__main__":
    encoder = CognitiveVisionEncoder()
    # Create a dummy white image to test
    dummy_img = Image.new('RGB', (224, 224), color = 'white')
    tokens = encoder([dummy_img])
    print(f"Visual Tokens Shape (Batch, NumPatches, d_model): {tokens.shape}")
