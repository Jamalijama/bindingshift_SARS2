import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pandas as pd
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from data_load_SARS2S1_SARSMERSesm2v2 import loading_data
import os


# Custom Dataset Class for numpy array data
class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array of shape (n, 560, 560) or (n, 560, 560, 3)
            labels: numpy array of shape (n,)
            transform: data augmentation transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Ensure images are 3-channel
        if len(images.shape) == 3:
            self.images = np.stack([images, images, images], axis=-1)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transformations
        if self.transform:
            # Ensure image data is in 0-255 range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        else:
            # If no transform, convert directly to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            # Normalize to [0,1]
            if image.max() > 1:
                image = image / 255.0
        
        return image, label


# Enhanced ResNet Binary Classification Model
class EnhancedResNetBinaryClassifier(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1):
        super(EnhancedResNetBinaryClassifier, self).__init__()
        
        # Select backbone network
        backbones = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if backbone not in backbones:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone = backbones[backbone](pretrained=pretrained)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace classification head - more complex structure
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.backbone(x)


# Data preprocessing transforms
def get_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((560, 560)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((560, 560)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_model_simple(model_path, backbone='resnet50', device='cpu'):
    """
    Simple model loading method
    
    Args:
        model_path: path to model file
        backbone: backbone network type
        device: device to load model on
    """
    # For PyTorch >= 1.13, use weights_only=False
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = EnhancedResNetBinaryClassifier(backbone=backbone, pretrained=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

# Method 3: Load only model weights
def load_model_weights_only(model_path, backbone='resnet50', device='cpu'):
    """
    Load only model weights (most compatible method)
    
    Args:
        model_path: path to model file
        backbone: backbone network type
        device: device to load model on
    """
    # Create model first
    model = EnhancedResNetBinaryClassifier(backbone=backbone, pretrained=False)
    model = model.to(device)
    
    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        
        # Alternative method: load with pickle
        try:
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            raise
    
    model.eval()
    print("Model weights loaded successfully!")
    return model, checkpoint

# Batch Prediction Function
def predict_batch(model, images, device='cpu'):
    """
    Batch prediction for new images
    
    Args:
        model: trained model
        images: input images of shape (m, 560, 560) or (m, 560, 560, 3)
        device: device to run inference on
    
    Returns:
        predictions: predicted classes
        probabilities: prediction probabilities
    """
    model.eval()
    transform = get_transforms(augment=False)
    
    # Preprocess images
    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img_pil)
        processed_images.append(img_tensor)
    
    batch_tensor = torch.stack(processed_images).to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = outputs.cpu().numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities

# Single Image Prediction
def predict_single_image(model, image_path, device='cpu'):
    """
    Predict a single image
    
    Args:
        model: trained model
        image_path: path to image file
        device: device to run inference on
    
    Returns:
        prediction: predicted class (0 or 1)
        probability: prediction probability
    """
    model.eval()
    transform = get_transforms(augment=False)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

data = loading_data()
test_X, indexlst = data[0], data[1]
print(test_X.shape)
path_model = './'

for file in os.listdir(path_model):
    if file.endswith ('40.pth'):
        
        model_name = path_model + file    
        # model_name = 'esm2wt_110.pth'
        model, checkpoint = load_model_simple(model_name, backbone='resnet50')
        
        new_images = np.random.random((10, 560, 560))
        
        
        # Method 2: Batch prediction
        predictions, probabilities = predict_batch(model, test_X)
        
        probabilities = [round(i, 3) for i in probabilities]
        
        # print (file, predictions, probabilities, model_name)
        
        # print (file, indexlst, predictions, probabilities)
        
        df_predicted = pd.DataFrame()
        # df_predicted['seqID'] = indexlst
        df_predicted['prediction'] = predictions
        df_predicted['probability'] = probabilities
        
        df_predicted.to_csv (path_model + 'df_predicted_S1SARS2_' + file[:-4] + '_30000.csv')
