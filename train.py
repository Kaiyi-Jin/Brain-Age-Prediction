#!/usr/bin/env python
# coding: utf-8

import warnings
import datetime
import logging
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants and configuration
DATA_ROOT = Path('/home/tione/notebook/taop-2021/100002/')
SAVE_ROOT = Path('/home/tione/notebook/')
OUTPUT_DIR = SAVE_ROOT / 'output'
MODEL_DIR = OUTPUT_DIR / 'model'
LOSS_DIR = OUTPUT_DIR / 'loss'

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, LOSS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 3D ResNet for brain age prediction')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--test_split', type=float, default=0.167, help='Test set proportion')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def setup_logging(args):
    """Set up logging configuration."""
    # Create a descriptive name from args
    args_str = '-'.join([f"{k}:{v}" for k, v in vars(args).items()])
    log_dir = Path('logging')
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging configuration
    log_file = log_dir / f"{args_str}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f'Starting training with parameters: {args_str}')
    return args_str

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
class BasicBlock(nn.Module):
    """Basic 3D ResNet block with two convolutions and a residual connection."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BrainAgePredictor(nn.Module):
    """
    Brain age prediction model based on 3D ResNet.
    Processes three modalities: T1 brain, GM, and WM.
    """
    def __init__(self, block, layers, dropout_rate=0.5):
        super(BrainAgePredictor, self).__init__()
        self.inplanes = 64
        
        # Define modality-specific pathways
        self.modality_pathways = nn.ModuleList([
            self._create_pathway() for _ in range(3)
        ])
        
        # Feature fusion and prediction
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(128 * 3, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _create_pathway(self):
        """Create a single modality processing pathway."""
        pathway = nn.ModuleDict({
            'conv1': nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            'bn1': nn.BatchNorm3d(64),
            'relu': nn.ReLU(inplace=True),
            'maxpool': nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            'layer1': self._make_layer(BasicBlock, 64, 1, reset_inplanes=True),
            'layer2': self._make_layer(BasicBlock, 128, 1, stride=2),
            'avgpool': nn.AdaptiveAvgPool3d((1, 1, 1))  # Use adaptive pooling for flexibility
        })
        return pathway

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reset_inplanes=False):
        if reset_inplanes:
            self.inplanes = 64
            
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _process_modality(self, x, pathway):
        """Process a single modality through its pathway."""
        x = pathway['conv1'](x)
        x = pathway['bn1'](x)
        x = pathway['relu'](x)
        x = pathway['maxpool'](x)
        
        x = pathway['layer1'](x)
        x = pathway['layer2'](x)
        x = pathway['avgpool'](x)
        
        return torch.flatten(x, 1)

    def forward(self, data):
        """
        Forward pass through the network.
        Args:
            data: List of three tensors [T1_brain, GM, WM]
        """
        # Process each modality
        features = []
        for i, x in enumerate(data):
            features.append(self._process_modality(x, self.modality_pathways[i]))
        
        # Concatenate features
        combined_features = torch.cat(features, dim=1)
        
        # Apply dropout for regularization
        combined_features = self.dropout(combined_features)
        
        # Final prediction
        output = self.output_layer(combined_features)
        
        return output

class BrainAgeDataset(Dataset):
    """Dataset for brain age prediction."""
    
    def __init__(self, data_root, save_root):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to the raw data directory
            save_root: Path to the processed data directory
        """
        self.data_root = data_root
        self.save_root = save_root
        
        # Load patient information
        df = pd.read_csv(data_root / 'train1_data_info.csv', encoding="gbk")
        
        # Get unique patients and their demographic info
        self.patient_list = df['id_patient'].drop_duplicates().reset_index()
        self.patient_list = self.patient_list.rename(columns={'index': 'col'})
        
        # Extract demographics
        self.demo = df.iloc[self.patient_list['col']][['id_gender', 'id_age']].reset_index(drop=True)
        self.demo = self.demo.replace({'M': 0, 'F': 1})  # Convert gender to binary
        
        self.labels = np.array(self.demo['id_age'])
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate that all required data files exist."""
        missing_files = []
        for idx, patient_id in enumerate(self.patient_list['id_patient']):
            t1_path = self.save_root / 'data/Total/T1brain' / f'{patient_id}_T1_brain.npy'
            gm_path = self.save_root / 'data/Total/GM' / f'{patient_id}_GM_mask.npy'
            wm_path = self.save_root / 'data/Total/WM' / f'{patient_id}_WM_mask.npy'
            
            if not all(path.exists() for path in [t1_path, gm_path, wm_path]):
                missing_files.append(patient_id)
        
        if missing_files:
            logging.warning(f"Missing data for {len(missing_files)} patients: {missing_files[:5]}...")
 
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        patient_id = self.patient_list['id_patient'][index]
        label = self.labels[index]
        
        # Load brain imaging data
        t1_brain = np.load(self.save_root / 'data/Total/T1brain' / f'{patient_id}_T1_brain.npy')
        gm_mask = np.load(self.save_root / 'data/Total/GM' / f'{patient_id}_GM_mask.npy')
        wm_mask = np.load(self.save_root / 'data/Total/WM' / f'{patient_id}_WM_mask.npy')
        
        # Convert to tensors
        t1_brain_tensor = torch.from_numpy(t1_brain.astype(np.float32))
        gm_mask_tensor = torch.from_numpy(gm_mask.astype(np.float32))
        wm_mask_tensor = torch.from_numpy(wm_mask.astype(np.float32))
        
        return torch.tensor(label, dtype=torch.float32), [t1_brain_tensor, gm_mask_tensor, wm_mask_tensor]
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

def prepare_dataloaders(dataset, batch_size, test_split, seed):
    """Prepare training and testing data loaders."""
    # Split dataset into training and testing
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_split,
        random_state=seed
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset)

def load_model(device, args):
    """Create and load the model."""
    # Create model
    model = BrainAgePredictor(BasicBlock, [1, 1, 1]).to(device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        logging.info(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Handle DataParallel models
        if isinstance(checkpoint, nn.DataParallel):
            checkpoint = checkpoint.module
            
        # Load state dict
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            # Clean keys if needed (remove 'module.' prefix)
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
    
    # Use DataParallel for multi-GPU training
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    return model

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for labels, inputs in tqdm(train_loader, desc="Training"):
        # Prepare inputs
        inputs = [x.unsqueeze(1).to(device) for x in inputs]
        labels = labels.unsqueeze(1).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        
        # Collect predictions and labels for metrics
        all_predictions.extend(outputs.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
    
    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    
    return running_loss / len(train_loader.dataset), mae

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for labels, inputs in tqdm(test_loader, desc="Evaluating"):
            # Prepare inputs
            inputs = [x.unsqueeze(1).to(device) for x in inputs]
            labels = labels.unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item()
            
            # Collect predictions and labels for metrics
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    
    return running_loss / len(test_loader.dataset), mae, all_predictions, all_labels

def save_model(model, path):
    """Save model checkpoint."""
    # If model is DataParallel, save the module
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
    }, path)

def plot_training_history(train_losses, train_maes, val_losses, val_maes, args_str):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot losses
    ax1.plot(train_losses, 'r-', label='Train Loss')
    ax1.plot(val_losses, 'b-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAEs
    ax2.plot(train_maes, 'r-', label='Train MAE')
    ax2.plot(val_maes, 'b-', label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (years)')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"training_history_{args_str}.png")
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    args_str = setup_logging(args)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Prepare dataset
    dataset = BrainAgeDataset(DATA_ROOT, SAVE_ROOT)
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Create data loaders
    train_loader, test_loader, train_size, test_size = prepare_dataloaders(
        dataset, args.batch_size, args.test_split, args.seed
    )
    logging.info(f"Training set size: {train_size}, Test set size: {test_size}")
    
    # Initialize model
    model = load_model(device, args)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss(reduction='sum')
    
    # Training history tracking
    train_losses = []
    train_maes = []
    val_losses = []
    val_maes = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    start_time = datetime.datetime.now()
    logging.info(f"Training started at {start_time}")
    
    for epoch in range(args.epochs):
        epoch_start = datetime.datetime.now()
        
        # Train for one epoch
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Evaluate on test set
        val_loss, val_mae, predictions, true_values = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Log results
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        logging.info(f"Epoch {epoch+1}/{args.epochs} - "
                     f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} years, "
                     f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f} years, "
                     f"Time: {epoch_time:.2f}s")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == 0:
            checkpoint_path = MODEL_DIR / f"model_{args_str}_epoch{epoch+1}.pt"
            save_model(model, checkpoint_path)
            logging.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = MODEL_DIR / f"best_model_{args_str}.pt"
            save_model(model, best_model_path)
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Training complete
    training_time = (datetime.datetime.now() - start_time).total_seconds() / 60
    logging.info(f"Training completed in {training_time:.2f} minutes")
    
    # Save final model
    final_model_path = MODEL_DIR / f"final_model_{args_str}.pt"
    save_model(model, final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(train_losses, train_maes, val_losses, val_maes, args_str)
    logging.info(f"Training history plot saved to {OUTPUT_DIR / f'training_history_{args_str}.png'}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_mae': train_maes,
        'val_loss': val_losses,
        'val_mae': val_maes
    }
    np.savez(LOSS_DIR / f"training_history_{args_str}.npz", **history)
    
    # Final evaluation
    final_val_loss, final_val_mae, _, _ = evaluate(model, test_loader, criterion, device)
    logging.info(f"Final evaluation - Val Loss: {final_val_loss:.4f}, Val MAE: {final_val_mae:.2f} years")

if __name__ == "__main__":
    main()
