"""
Model Export Toolkit for IMX500 and ONNX conversion.

This module provides utilities for exporting PyTorch models to ONNX format
and quantizing them for the Sony IMX500 target platform.
"""

import glob
import os
import torch
import onnx
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import model_compression_toolkit as mct
from edgemdt_tpc import get_target_platform_capabilities


class ImageFolderDataset(Dataset):
    """Loads JPG/PNG images from a folder and preprocesses them for calibration."""
    
    def __init__(self, folder_path, transform=None, limit=None):
        """
        Args:
            folder_path: Path to directory containing images
            transform: torchvision transforms to apply to images
            limit: Maximum number of images to load (None for all)
        """
        self.image_paths = list(glob.glob(os.path.join(folder_path, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))
        print(f"Found {len(self.image_paths)} images in {folder_path} for calibration.")
        
        if limit:
            self.image_paths = self.image_paths[:limit]
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


class RepresentativeDataGenerator:
    """Generator for representative calibration data used in quantization."""
    
    def __init__(self, image_folder_path, input_shape=(3, 256, 256), 
                 batch_size=1, num_images=50, device='cuda'):
        """
        Args:
            image_folder_path: Path to directory containing calibration images
            input_shape: Model input shape as (C, H, W)
            batch_size: Batch size for data loading
            num_images: Number of images to use for calibration
            device: Device to load data to ('cuda' or 'cpu')
        """
        self.image_folder_path = image_folder_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_images = num_images
        self.device = device
        self.callCounter = 0
        
        # Create preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((input_shape[1], input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset and dataloader
        self.dataset = ImageFolderDataset(
            image_folder_path, 
            transform=self.transform, 
            limit=num_images
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    def __call__(self):
        """
        Generator function that yields batches of calibration data.
        
        Yields:
            torch.Tensor: Batch of preprocessed images on the specified device
        """
        for x, _ in self.dataloader:
            self.callCounter += 1
            if self.callCounter - 2 <= self.num_images // self.batch_size:
                x = x.unsqueeze(0)
            yield x.to(device=self.device)
    
    def set_transform(self, transform):
        """
        Update the preprocessing transform.
        
        Args:
            transform: New torchvision transform to apply
        """
        self.transform = transform
        self.dataset = ImageFolderDataset(
            self.image_folder_path, 
            transform=transform, 
            limit=self.num_images
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )


class OnnxExporter:
    """Utility for exporting PyTorch models to ONNX format."""
    
    def __init__(self, input_shape=(3, 256, 256), opset_version=15, 
                 input_names=None, output_names=None, dynamic_axes=None):
        """
        Args:
            input_shape: Model input shape as (C, H, W)
            opset_version: ONNX opset version to use
            input_names: List of input tensor names
            output_names: List of output tensor names
            dynamic_axes: Dictionary defining dynamic axes (e.g., batch dimension)
        """
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {"input": {0: "batch_size"}}
    
    def export(self, model, output_path, batch_size=1, device='cpu', verbose=False):
        """
        Export a PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Path where the ONNX model will be saved
            batch_size: Batch size for the dummy input
            device: Device to run the export on ('cuda' or 'cpu')
            verbose: Whether to print verbose ONNX export logs
        
        Returns:
            onnx.ModelProto: Loaded ONNX model
        """
        model.eval()
        model.to(device=device)
        
        # Create dummy input
        if len(self.input_shape) == 3:
            dummy_input = torch.randn(
                batch_size, 
                self.input_shape[0], 
                self.input_shape[1], 
                self.input_shape[2]
            ).to(device=device)
        else:
            dummy_input = torch.randn(
                batch_size, 
                self.input_shape[0], 
                self.input_shape[1], 
            ).to(device=device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            opset_version=self.opset_version,
            dynamic_axes=self.dynamic_axes,
            verbose=verbose,
        )
        
        # Validate the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ Exported ONNX model to {output_path}")
        
        return onnx_model


class Imx500Exporter:
    """Utility for quantizing PyTorch models for the Sony IMX500 target platform."""
    
    def __init__(self, tpc_version='1.0', device='cuda'):
        """
        Args:
            tpc_version: Target platform capabilities version
            device: Device to run quantization on ('cuda' or 'cpu')
        """
        self.tpc_version = tpc_version
        self.device = device
        self.tpc = get_target_platform_capabilities(
            tpc_version=tpc_version, 
            device_type="imx500"
        )
    
    def quantize(self, model, representative_data_gen, output_path):
        """
        Quantize a PyTorch model for the IMX500 and export to ONNX.
        
        Args:
            model: PyTorch model to quantize
            representative_data_gen: Generator function yielding calibration data
            output_path: Path where the quantized ONNX model will be saved
        
        Returns:
            tuple: (quantized_model, quantization_info)
        """
        model.to(device=self.device)
        model.eval()
        
        # Perform post-training quantization
        quantized_model, quant_info = mct.ptq.pytorch_post_training_quantization(
            in_module=model,
            representative_data_gen=representative_data_gen,
            target_platform_capabilities=self.tpc
        )
        
        # Export quantized model to ONNX
        mct.exporter.pytorch_export_model(
            model=quantized_model,
            save_model_path=output_path,
            repr_dataset=representative_data_gen,
            serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX
        )
        
        print(f"✅ Quantized ONNX model saved to {output_path}")
        return quantized_model, quant_info
    
    def set_tpc_version(self, version):
        """
        Update the target platform capabilities version.
        
        Args:
            version: New TPC version string
        """
        self.tpc_version = version
        self.tpc = get_target_platform_capabilities(
            tpc_version=version, 
            device_type="imx500"
        )
