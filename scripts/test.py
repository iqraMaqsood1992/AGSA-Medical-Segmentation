import torch
import os
import argparse
import numpy as np
from models.models import get_model
from utils.config import get_config
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import nibabel as nib
from utils.utils_gray import correct_dims, to_long_tensor
from torchvision import transforms as T
from torchvision.transforms import functional as F
from medpy.metric.binary import dc
import glob
from PIL import Image

class TestDataset(Dataset):
    """Dataset for testing with the specific directory structure"""
    def __init__(self, data_path, img_size=256):
        self.img_size = img_size
        self.img_path = os.path.join(data_path, 'Training-Testing/img')
        self.label_path = os.path.join(data_path, 'Training-Testing/label')
        
        # Find all image files
        self.img_files = []
        for subdir in os.listdir(self.img_path):
            subdir_path = os.path.join(self.img_path, subdir)
            if os.path.isdir(subdir_path):
                for img_file in glob.glob(os.path.join(subdir_path, '*.nii.gz')):
                    self.img_files.append(img_file)
        
        print(f"Found {len(self.img_files)} test images")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get image file path
        img_file = self.img_files[idx]
        img_name = os.path.basename(img_file)
        img_subdir = os.path.basename(os.path.dirname(img_file))
        
        # Extract image number from filename (like "0001" from "img0001-0005.nii.gz")
        img_number = img_name.split('-')[0][3:]  # Remove "img" prefix and get number before "-"
        
        # Load image using nibabel
        try:
            nii_img = nib.load(img_file)
            image = nii_img.get_fdata()
            
            # Convert to 2D if 3D (taking middle slice)
            if len(image.shape) > 2:
                mid_slice = image.shape[2] // 2
                image = image[:, :, mid_slice]
            
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min() + 1e-5) * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Try multiple possible mask locations
        mask = None
        possible_mask_paths = [
            # Try with label folder 0061
            os.path.join(self.label_path, '0061', f"label{img_number}-0061.nii.gz"),
            
            # Try with the image subdirectory 
            os.path.join(self.label_path, img_subdir, f"label{img_number}-{img_subdir}.nii.gz"),
            
            # Try with original format but different directory
            os.path.join(self.label_path, '0001', f"label{img_number}-0001.nii.gz"),
        ]
        
        for mask_path in possible_mask_paths:
            if os.path.exists(mask_path):
                try:
                    nii_mask = nib.load(mask_path)
                    mask_data = nii_mask.get_fdata()
                    
                    # Convert to 2D if 3D (taking middle slice)
                    if len(mask_data.shape) > 2:
                        mid_slice = mask_data.shape[2] // 2
                        mask_data = mask_data[:, :, mid_slice]
                        
                    mask = mask_data.astype(np.uint8)
                    print(f"Found mask at: {mask_path}")
                    break
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
        
        # If no mask found, create a dummy mask
        if mask is None:
            print(f"Warning: No mask found for image {img_name}")
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Process image and mask to tensors
        image = correct_dims(image)
        mask = correct_dims(mask)
        
        # Convert to PIL for transformations
        image_pil = Image.fromarray(image.squeeze())
        mask_pil = Image.fromarray(mask.squeeze())
        
        # Resize to model input size
        image_pil = image_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to tensor
        image_tensor = T.ToTensor()(image_pil)
        mask_tensor = to_long_tensor(mask_pil)
        
        # Create a smaller version for deep supervision if needed
        mask_mini = F.resize(mask_pil, (self.img_size//8, self.img_size//8), Image.NEAREST)
        mask_mini_tensor = to_long_tensor(mask_mini)
        
        return image_tensor, mask_tensor, mask_mini_tensor, img_name
def test_model_on_data():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test model on real data')
    parser.add_argument('--modelname', default='SETR_DSFormer', type=str, help='type of model')
    parser.add_argument('--task', default='Synapse', help='task or dataset name')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint file')
    parser.add_argument('--data_path', default='./data/Synapse', help='path to test data')
    parser.add_argument('--save_results', action='store_true', help='save prediction results')
    parser.add_argument('--output_dir', default='./results', help='directory to save results')
    args = parser.parse_args()
    
    # Get configuration
    opt = get_config(args.task)
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    print(f"Initializing {args.modelname} model...")
    model = get_model(modelname=args.modelname, img_size=opt.img_size, 
                     img_channel=opt.img_channel, classes=opt.classes)
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("✓ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Create test dataset and dataloader
    print("Preparing test dataset...")
    test_dataset = TestDataset(args.data_path, img_size=opt.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create output directory if saving results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test loop
    dice_scores = []
    print("\nRunning inference on test data...")
    
    with torch.no_grad():
        for i, (image, mask, mask_mini, filename) in enumerate(test_loader):
            # Move inputs to device
            image = image.to(device)
            mask = mask.to(device)
            
            # Forward pass
            outputs = model(image)
            
            # Get predictions
            if isinstance(outputs, tuple):
                output = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs
            else:
                output = outputs
            
            # Convert to binary predictions for dice calculation
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt = mask.squeeze(0).cpu().numpy()
            
            # Calculate Dice coefficient for this image
            dice = 0
            valid_classes = 0
            for c in range(1, opt.classes):  # Skip background class
                if np.sum(gt == c) > 0:  # Only calculate for classes present in ground truth
                    class_dice = dc(pred == c, gt == c)
                    dice += class_dice
                    valid_classes += 1
                    print(f"  Image {i+1}/{len(test_dataset)}, Class {c}: Dice = {class_dice:.4f}")
            
            # Average Dice over present classes
            avg_dice = dice / max(1, valid_classes)
            dice_scores.append(avg_dice)
            print(f"  Image {i+1}/{len(test_dataset)}: Average Dice = {avg_dice:.4f}")
            
            # Save results if requested
            if args.save_results:
                # Create a colored visualization
                plt.figure(figsize=(12, 4))
                
                plt.subplot(131)
                plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
                plt.title('Input Image')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(gt, cmap='tab20', vmin=0, vmax=opt.classes-1)
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(pred, cmap='tab20', vmin=0, vmax=opt.classes-1)
                plt.title('Prediction')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"result_{i}.png"))
                plt.close()
    
    # Print overall results
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        print("\nTest completed!")
        print(f"Overall Mean Dice: {mean_dice:.4f}")
        
        # List top 5 best and worst cases
        if len(dice_scores) > 5:
            indices = np.argsort(dice_scores)
            print("\nTop 5 best cases:")
            for i in range(5):
                idx = indices[-(i+1)]
                print(f"  Image {idx+1}: Dice = {dice_scores[idx]:.4f}")
                
            print("\nTop 5 worst cases:")
            for i in range(5):
                idx = indices[i]
                print(f"  Image {idx+1}: Dice = {dice_scores[idx]:.4f}")
    else:
        print("\nNo valid dice scores calculated. Check if the test data contains valid mask files.")

if __name__ == "__main__":
    test_model_on_data()