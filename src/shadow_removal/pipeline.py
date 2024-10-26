from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from shadow_removal.config import Config
from shadow_removal.models.unext import UNextGCNet, UNextDRNet
from shadow_removal.utils.image_processing import preprocess_image, postprocess_image


class ShadowRemovalPipeline:
    """Main pipeline for shadow removal.
    
    This class provides methods for removing shadows from images using
    a two-stage deep learning approach.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model1 = UNextGCNet(num_classes=3, input_channels=3).to(device)
        self.model2 = UNextDRNet(num_classes=3, input_channels=6).to(device)
        
        self.model1.load_weights(Config.GCNET_WEIGHTS)
        self.model2.load_weights(Config.DRNET_WEIGHTS)
    
    @torch.no_grad()
    def process_image(self, image_path, output_path):
        """Process a single image to remove shadows.
        
        Args:
            image_path: Path to input image
            output_path: Path to save the processed image
            
        Returns:
            numpy.ndarray: Processed image array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Preprocess
        tensor, pad_h, pad_w = preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # First model inference
        shadow = self.model1(tensor)
        model1_output = torch.clamp(tensor / shadow, 0, 1)
        
        # Combine inputs for second model
        combined_input = torch.cat((tensor, model1_output), 1)
        final_output, _, _, _ = self.model2(combined_input)
        
        # Post-process
        result = postprocess_image(final_output, pad_h, pad_w)
        
        # Ensure output directory exists if output_path is a file
        output_path = Path(output_path)
        if output_path.suffix:  # If path has an extension, it's a file path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:  # If no extension, treat as directory path
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / Path(image_path).name
            
        cv2.imwrite(str(output_path), result)
        return result
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
        """        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        image_paths = list(input_dir.glob('*'))
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                output_path = output_dir / image_path.name
                self.process_image(image_path, output_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")