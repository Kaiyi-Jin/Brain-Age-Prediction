#!/usr/bin/env python
# coding: utf-8

import warnings
import datetime
import logging
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test brain age prediction model')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='/home/tione/notebook/taop-2021-result/', 
                       help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path('logging')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"testing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class BrainAgePredictor(nn.Module):
    """Brain age prediction model with three modality pathways."""
    def __init__(self, block, layer_nums):
        super(BrainAgePredictor, self).__init__()
        # modality_1 - T1 brain
        self.inplanes = 64
        self.conv11 = nn.Conv3d(1, 64, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn11 = nn.BatchNorm3d(64)
        self.relu11 = nn.ReLU(inplace=True)
        self.maxpool11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer11 = self._make_layer(block, 64, layer_nums[0], flag=1)
        self.layer12 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.pooling_layer1 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Use adaptive pooling for flexibility
        
        # modality_2 - Gray Matter
        self.conv21 = nn.Conv3d(1, 64, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn21 = nn.BatchNorm3d(64)
        self.relu21 = nn.ReLU(inplace=True)
        self.maxpool21 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer21 = self._make_layer(block, 64, layer_nums[0], flag=1)
        self.layer22 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.pooling_layer2 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # modality_3 - White Matter
        self.conv31 = nn.Conv3d(1, 64, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn31 = nn.BatchNorm3d(64)
        self.relu31 = nn.ReLU(inplace=True)
        self.maxpool31 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer31 = self._make_layer(block, 64, layer_nums[0], flag=1)
        self.layer32 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.pooling_layer3 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final prediction layer
        self.output_layer = nn.Linear(128*3, 1)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, flag=0):
        """Create a layer of blocks."""
        if flag:
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
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    def forward(self, data):
        """Forward pass through the network."""
        x1, x2, x3 = data
        
        # Process T1 brain images
        x1 = self.conv11(x1)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.maxpool11(x1)
        x1 = self.layer11(x1)
        x1 = self.layer12(x1)
        x1 = self.pooling_layer1(x1)
        x1 = torch.flatten(x1, 1)
        
        # Process Gray Matter images
        x2 = self.conv21(x2)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.maxpool21(x2)
        x2 = self.layer21(x2)
        x2 = self.layer22(x2)
        x2 = self.pooling_layer2(x2)
        x2 = torch.flatten(x2, 1)
        
        # Process White Matter images
        x3 = self.conv31(x3)
        x3 = self.bn31(x3)
        x3 = self.relu31(x3)
        x3 = self.maxpool31(x3)
        x3 = self.layer31(x3)
        x3 = self.layer32(x3)
        x3 = self.pooling_layer3(x3)
        x3 = torch.flatten(x3, 1)
        
        # Concatenate features from all three modalities
        final_feature_vector = torch.cat((x1, x2, x3), 1)
        
        # Final prediction
        result = self.output_layer(final_feature_vector)
        
        return result

class TestDataset(Dataset):
    """Dataset for testing brain age prediction."""
    
    def __init__(self, data_root_path, save_root_path):
        """
        Initialize the test dataset.
        
        Args:
            data_root_path: Path to the root directory containing data files
            save_root_path: Path to the root directory for saving processed data
        """
        self.data_root = data_root_path
        self.save_root = save_root_path
        
        # Read test data information
        df = pd.read_csv(self.data_root / 'test1_data_info.csv', encoding="gbk")
        
        # Get unique patient IDs
        self.patient_list = df['id_patient'].drop_duplicates().reset_index()
        self.patient_list = self.patient_list.rename(columns={'index': 'col'})
        
        # Validate data availability
        self._validate_data()
        
    def _validate_data(self):
        """Validate that required data files exist."""
        missing_files = []
        for idx, patient_id in enumerate(self.patient_list['id_patient']):
            t1_path = self.save_root / 'data/Total/T1brain' / f'{patient_id}_T1_brain.npy'
            gm_path = self.save_root / 'data/Total/GM' / f'{patient_id}_GM_mask.npy'
            wm_path = self.save_root / 'data/Total/WM' / f'{patient_id}_WM_mask.npy'
            
            if not all(path.exists() for path in [t1_path, gm_path, wm_path]):
                missing_files.append(patient_id)
        
        if missing_files:
            logging.warning(f"Missing data for {len(missing_files)} patients")
            logging.warning(f"First few missing patient IDs: {missing_files[:5]}")
 
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        patient_id = self.patient_list['id_patient'][index]
        
        # Load brain imaging data
        try:
            img_t1_brain = np.load(self.save_root / 'data/Total/T1brain' / f'{patient_id}_T1_brain.npy')
            img_gm = np.load(self.save_root / 'data/Total/GM' / f'{patient_id}_GM_mask.npy')
            img_wm = np.load(self.save_root / 'data/Total/WM' / f'{patient_id}_WM_mask.npy')
            
            # Convert to tensors with appropriate types
            return [
                torch.from_numpy(img_t1_brain.astype(np.float32)), 
                torch.from_numpy(img_gm.astype(np.float32)), 
                torch.from_numpy(img_wm.astype(np.float32))
            ], patient_id
        except Exception as e:
            logging.error(f"Error loading data for patient {patient_id}: {e}")
            # Return zeros as fallback
            shape = (128, 128, 128)  # Assuming this is the expected shape
            return [
                torch.zeros(shape, dtype=torch.float32),
                torch.zeros(shape, dtype=torch.float32),
                torch.zeros(shape, dtype=torch.float32)
            ], patient_id
 
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.patient_list)

def load_model(model_path, device):
    """Load the trained model."""
    try:
        logging.info(f"Loading model from {model_path}")
        
        # Handle different model saving formats
        if model_path.endswith('.pt'):
            # New format with state dict
            checkpoint = torch.load(model_path, map_location=device)
            model = BrainAgePredictor(BasicBlock, [1, 1, 1]).to(device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
        else:
            # Old format with full model
            model = torch.load(model_path, map_location=device)
            
            # Handle DataParallel models
            if isinstance(model, nn.DataParallel):
                model = model.module
                
        logging.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def run_inference(model, test_loader, device):
    """Run inference on the test dataset."""
    model.eval()
    predictions = {}
    
    start_time = datetime.datetime.now()
    logging.info("Starting inference...")
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            inputs, patient_ids = data
            
            # Prepare inputs
            inputs = [x.unsqueeze(1).to(device) for x in inputs]
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions with patient IDs
            for i, patient_id in enumerate(patient_ids):
                predictions[patient_id] = outputs[i].item()
    
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    logging.info(f"Inference completed in {elapsed_time:.2f} seconds")
    
    return predictions

def save_results(predictions, data_root, output_dir, timestamp):
    """Save prediction results to CSV."""
    try:
        # Create output directory
        result_dir = Path(output_dir) / '02_predicts' / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data info
        df_test = pd.read_csv(data_root / 'test1_data_info.csv', encoding="gbk")
        
        # Filter results as per requirements
        df_result = df_test[
            (df_test.label_type == 'MR') & 
            (df_test.id_area == 1.0)
        ][['id_type', 'id_project', 'id_patient', 'id_exam', 'id_series', 'id_image', 'id_doctor']]
        
        # Add predictions
        df_result['id_age'] = df_result['id_patient'].map(predictions)
        df_result['id_series'] = -1  # As per original code
        
        # Save results
        result_file = result_dir / '02_predicts.csv'
        df_result.to_csv(result_file, index=None)
        logging.info(f"Results saved to {result_file}")
        
        # Create summary statistics
        age_predictions = list(predictions.values())
        stats = {
            'Mean Age': np.mean(age_predictions),
            'Median Age': np.median(age_predictions),
            'Min Age': np.min(age_predictions),
            'Max Age': np.max(age_predictions),
            'Std Dev': np.std(age_predictions)
        }
        
        # Log statistics
        logging.info("Prediction Statistics:")
        for stat_name, stat_value in stats.items():
            logging.info(f"  {stat_name}: {stat_value:.2f}")
            
        # Optional: Create a histogram of predicted ages
        plt.figure(figsize=(10, 6))
        plt.hist(age_predictions, bins=20, alpha=0.7)
        plt.xlabel('Predicted Age (years)')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Ages')
        plt.grid(True, alpha=0.3)
        
        plot_file = result_dir / 'age_distribution.png'
        plt.savefig(plot_file)
        plt.close()
        logging.info(f"Age distribution plot saved to {plot_file}")
        
        return result_file
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def main():
    """Main function for testing."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    timestamp = setup_logging()
    
    # Define paths
    data_root = Path('/home/tione/notebook/taop-2021/100002/')
    save_root = Path('/home/tione/notebook/')
    
    # If model path not provided, construct it from default pattern
    if not args.model_path:
        model_base_name = '-'.join([
            f"GPU_num:{torch.cuda.device_count()}",
            f"batch_size:{args.batch_size}",
            f"epoch:{20}",  # Default value
            f"pretrain:{True}",
            f"lr:{0.0001}",
            f"split_num:{6}"
        ])
        args.model_path = save_root / 'output/model' / f"model{model_base_name}.pkl"
    
    # Log configuration
    logging.info(f"Test configuration:")
    logging.info(f"  Model path: {args.model_path}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Output directory: {args.output_dir}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    try:
        # Load model
        model = load_model(args.model_path, device)
        model.to(device)
        
        # Create dataset and data loader
        test_dataset = TestDataset(data_root, save_root)
        logging.info(f"Test dataset size: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,  # No need to shuffle for inference
            num_workers=4,  # Use multiple workers for faster data loading
            pin_memory=torch.cuda.is_available()  # Pin memory if using GPU
        )
        
        # Run inference
        predictions = run_inference(model, test_loader, device)
        logging.info(f"Generated predictions for {len(predictions)} patients")
        
        # Save results
        result_file = save_results(predictions, data_root, args.output_dir, timestamp)
        logging.info("Testing completed successfully")
        
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
