import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
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


# Enhanced ResNet Multi-class Classification Model (3 classes)
class EnhancedResNetMultiClassifier(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=3):
        super(EnhancedResNetMultiClassifier, self).__init__()
        
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
            nn.Softmax(dim=1)  # Softmax for multi-class
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


def load_model_simple(model_path, backbone='resnet50', num_classes=3, device='cuda'):
    """
    Simple model loading method with configurable num_classes
    
    Args:
        model_path: path to model file
        backbone: backbone network type
        num_classes: number of output classes (3 for multi-class)
        device: device to load model on
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with specified num_classes
    model = EnhancedResNetMultiClassifier(backbone=backbone, 
                                          pretrained=False, 
                                          num_classes=num_classes)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_batch_3class(model, images, device='cuda', batch_size=64):
    """
    Batch prediction for 3-class classification
    
    Args:
        model: trained model (with 3 output classes)
        images: input images of shape (m, 560, 560) or (m, 560, 560, 3)
        device: device to run inference on
        batch_size: internal batch size for processing
    
    Returns:
        predictions: predicted class indices (0, 1, or 2)
        probabilities: prediction probabilities for all 3 classes (shape: m x 3)
    """
    model.eval()
    transform = get_transforms(augment=False)
    
    all_predictions = []
    all_probabilities = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
        batch_images = images[i:i+batch_size]
        processed_images = []
        
        for img in batch_images:
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
            # outputs are already probabilities due to Softmax
            probs = outputs.cpu().numpy()
            # print(probs)
            predictions = np.argmax(probs, axis=1)
            
        all_predictions.extend(predictions)
        all_probabilities.extend(probs)
    
    return np.array(all_predictions), np.array(all_probabilities)


# Main prediction code
def main():
    # Configuration
    model_name = 'classifier3_checkpoint-330_SM229Eesm2_40.pth'
    n0 = 30000
    batch_size = 64    # Processing batch size
    
    # Paths
    path1 = '../data/'
    path2 = '../embedding_SM229E/'
    path3 = '../model_training3/'
    label_file2 = 'df_metadata_forS1_SARS2_testing.csv'
    
    # Load metadata
    print("Loading metadata...")
    df_test = pd.read_csv(path1 + label_file2, index_col='seqID')
    
    if n0 + 10000 < df_test.shape[0]:
        n_samples = n0 + 10000  # Number of samples to process
    else:
        n_samples = df_test.shape[0]
    

    indexlst = df_test.index.tolist()[n0:n_samples]
    labellst = df_test.label.tolist()[n0:n_samples]
    label1lst = df_test.label1.tolist()[n0:n_samples]
    label0lst = df_test.label0.tolist()[n0:n_samples]
    
    # Load embeddings
    print("Loading embeddings...")
    file = path2 + 'esm2finetuned_SMS229E_checkpoint-330_parsed_SpikeS1_allv2_embeddingtesting32204.pkl'
    
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            array = np.array(data)[n0:n_samples, :245, :]  # Only load needed samples
            sample_num = array.shape[0]
            print(f"Loaded {sample_num} samples")
            
            # Reshape to image format
            reshaped_array = array.reshape(sample_num, -1)
            X_all = reshaped_array.reshape((sample_num, 560, 560))
            print(f"Image shape: {X_all.shape}")
            
    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load model with 3 classes
    print("Loading model...")
    model_path = path3 + model_name
    print(f"Model path: {model_path}")
    
    try:
        # Use num_classes=3 for multi-class classification
        model, checkpoint = load_model_simple(model_path, backbone='resnet50', num_classes=3)
        print("Model loaded successfully!")
        
        # Check output classes
        final_layer = list(model.backbone.fc.children())[-2]
        print(f"Model has {final_layer.out_features} output classes")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run predictions
    print("Running predictions...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        predictions, probabilities = predict_batch_3class(
            model, 
            X_all, 
            device=device,
            batch_size=batch_size
        )
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Unique predictions: {np.unique(predictions)}")
        
        # Create results DataFrame with 3-class outputs
        df_results = pd.DataFrame({
            'seqID': indexlst,
            'label': labellst,  # Original binary labels (0 or 1)
            'label1': label1lst,
            'label0': label0lst,
            'predicted_class': predictions,  # 0, 1, or 2
            'prob_class0': probabilities[:, 0],
            'prob_class1': probabilities[:, 1],
            'prob_class2': probabilities[:, 2]
        })
        
        # Optionally, convert 3-class predictions to binary
        # If class 0 = negative, classes 1 and 2 = positive
        df_results['predicted_binary'] = (df_results['predicted_class'] > 0).astype(int)
        
        # Save results
        output_file = f'df_predicted_3class_{model_name[:-4]}_{n_samples}' + '.csv'
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Calculate and print statistics
        print("\n" + "="*50)
        print("PREDICTION STATISTICS")
        print("="*50)
        
        # Class distribution
        print("\nPredicted class distribution:")
        class_counts = df_results['predicted_class'].value_counts().sort_index()
        for cls in range(3):
            count = class_counts.get(cls, 0)
            print(f"  Class {cls}: {count} samples ({count/len(df_results)*100:.2f}%)")
        
        # Binary accuracy (if you have binary labels)
        binary_accuracy = (df_results['label'] == df_results['predicted_binary']).mean()
        print(f"\nBinary classification accuracy: {binary_accuracy:.4f}")
        
        # Confusion matrix for binary classification
        print("\nConfusion Matrix (Binary):")
        cm = pd.crosstab(df_results['label'], df_results['predicted_binary'], 
                        rownames=['True'], colnames=['Predicted'])
        print(cm)
                
        # Save summary
        summary = {
            'total_samples': len(df_results),
            'class_0_predicted': (df_results['predicted_class'] == 0).sum(),
            'class_1_predicted': (df_results['predicted_class'] == 1).sum(),
            'class_2_predicted': (df_results['predicted_class'] == 2).sum(),
            'binary_accuracy': binary_accuracy
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = f'prediction_summary_3class_{model_name[:-4]}_{n_samples}' + '.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")
        
        # Show sample predictions
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS (first 10 samples)")
        print("="*50)
        print(df_results[['seqID', 'label', 'predicted_class', 'predicted_binary', 
                         'prob_class0', 'prob_class1', 'prob_class2']].head(10))
        
        print(df_results[df_results['predicted_class'] ==1].shape)
        print(df_results[df_results['predicted_class'] ==2].shape)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

###########################################

        
if __name__ == "__main__":
    main()
                