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
import os
import pickle
from data_load_SARSMERSS1_SARSMERSesm2 import loading_data

warnings.filterwarnings('ignore')

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

# Trainer Class
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            preds = (outputs > 0.5).float()
            batch_corrects = torch.sum(preds == labels.data)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += batch_corrects
            total_samples += inputs.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(batch_corrects/inputs.size(0)):.4f}'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                preds = (outputs > 0.5).float()
                running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc.cpu())
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc.cpu())
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f'New best model saved with val_acc: {val_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_state)
        
        return self.model

# Evaluator Class
class Evaluator:
    def __init__(self, model, plt_path, device='cuda'):
        self.model = model
        self.device = device
        self.plt_path = plt_path
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return np.array(all_probs), np.array(all_preds), np.array(all_labels)
    
    def calculate_metrics(self, probs, preds, labels):
        accuracy = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=['Class 0', 'Class 1'])
        
        return accuracy, auc, cm, report
    
    def roc_ (self, probs, labels): #, plt_path
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc_score = roc_auc_score(labels, probs)
       
        return auc_score, fpr, tpr

# Main Training Workflow
def main_training_workflow(X, y, \
                           test_size=0.2,\
                           val_size=0.2,\
                           batch_size=16,\
                           num_epochs=50,\
                           backbone ='resnet50',\
                           model_name = 'my_resnet.pth'):
    """
    Main training workflow for binary classification
    
    Args:
        X: input data of shape (n, 560, 560) or (n, 560, 560, 3)
        y: labels of shape (n,)
        test_size: test set ratio
        val_size: validation set ratio
        batch_size: batch size
        num_epochs: number of training epochs
        backbone: backbone network type
    """
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data splitting
    n_total = len(X)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    indices = np.random.permutation(n_total)
    train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create datasets and data loaders
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    train_dataset = ArrayDataset(X_train, y_train, transform=train_transform)
    val_dataset = ArrayDataset(X_val, y_val, transform=val_transform)
    test_dataset = ArrayDataset(X_test, y_test, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = EnhancedResNetBinaryClassifier(backbone=backbone, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Train model
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    trained_model = trainer.train(num_epochs)
    
    # Evaluate model
    evaluator = Evaluator(trained_model, model_name)
    probs, preds, labels = evaluator.evaluate(test_loader)
    accuracy, auc, cm, report = evaluator.calculate_metrics(probs, preds, labels)
    
    # Print results
    print('\n' + '='*60)
    print('FINAL EVALUATION RESULTS')
    print('='*60)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test AUC: {auc:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(report)
    path = 'roc_curve_' + model_name[:4] + '.svg'
    # Plot ROC curve
    roc_results = evaluator.roc_(probs, labels)
    auc_score, fpr, tpr = roc_results[0], roc_results[1], roc_results[2]
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.grid(False)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    plt.grid(False)
    plt.plot(trainer.train_accuracies, label='Train Acc')
    plt.plot(trainer.val_accuracies, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    
    plt.tight_layout()
    plt.savefig('training_results_' + model_name[:-4] +'.svg', dpi=600, bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(8, 6))        
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.grid(False)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_' + model_name[:-4] + '.svg', dpi=600, bbox_inches='tight')
    # plt.show()



    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies,
        'test_accuracy': accuracy,
        'test_auc': auc,
        'config': {
            'backbone': backbone,
            'batch_size': batch_size,
            'num_epochs': num_epochs
        }
    }, model_name)
    
    print('Model saved successfully!')
    
    return trained_model, trainer, evaluator

# Batch Prediction Function
def predict_batch(model, images, device='cuda'):
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

# Load Saved Model
def load_model(model_path, backbone='resnet50', device='cuda'):
    """
    Load a saved model from checkpoint
    
    Args:
        model_path: path to model file
        backbone: backbone network type
        device: device to load model on
    
    Returns:
        model: loaded model
        checkpoint_info: checkpoint information
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only = 'True')
    
    model = EnhancedResNetBinaryClassifier(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

# Example Usage
def trainmingModel(X, y, name, epoch):
    """
    Example usage - generate synthetic data and train model
    """
    # Generate synthetic data
    # n_samples = 1000
    img_height, img_width = 560, 560
    n_samples = X.shape[0]
    
    # Generate random image data
    # X = np.random.rand(n_samples, img_height, img_width).astype(np.float32)
    # Generate random labels
    # y = np.random.randint(0, 2, n_samples)
    
    print(f"Generated data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train model
    model, trainer, evaluator = main_training_workflow(
        X=X,
        y=y,
        test_size = 0.15,
        val_size = 0.15,
        batch_size = 100,
        num_epochs = epoch,
        backbone='resnet50',
        model_name = name
    )
    
    return model, trainer, evaluator

# loading data

for epoch in range(100, 210, 10):
    
    
    data = loading_data()
    X, y, name = data[0], data[1], data[2] + '_' + str(epoch) + '.pth'
                    
                
    
    if __name__ == '__main__':
        # Required packages installation:
        # pip install torch torchvision scikit-learn matplotlib tqdm pandas pillow
        
        # First run quick test
        # quick_test()
        model, trainer, evaluator = trainmingModel(X, y, name, epoch)
 