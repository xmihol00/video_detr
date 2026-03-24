
import glob
import inspect
import os
import torch
import onnx
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import model_compression_toolkit as mct

try:
    from edgemdt_tpc import get_target_platform_capabilities
    EDGEMDT_AVAILABLE = True
except ImportError:
    EDGEMDT_AVAILABLE = False
    print("WARNING: edgemdt_tpc not found. Quantization for IMX500 will fail if requested.")

class ImageFolderDataset(Dataset):
    """Loads JPG/PNG images from a folder and preprocesses them for calibration."""
    
    def __init__(self, folder_path, transform=None, limit=None):
        self.image_paths = list(glob.glob(os.path.join(folder_path, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))
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
        return img, 0

class RepresentativeDataGenerator:
    """Generator for representative calibration data used in quantization."""
    
    def __init__(self, image_folder_path, input_shape=(3, 256, 256), 
                 batch_size=1, num_images=50, device='cpu'):
        self.image_folder_path = image_folder_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_images = num_images
        self.device = device
        self.callCounter = 0
        
        self.transform = transforms.Compose([
            transforms.Resize((input_shape[1], input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
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
        for x, _ in self.dataloader:
            self.callCounter += 1
            yield [x.to(device=self.device)]

class OnnxExporter:
    def __init__(self, input_shape=(3, 256, 256), opset_version=15, 
                 input_names=None, output_names=None, dynamic_axes=None):
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {"input": {0: "batch_size"}}
    
    def export(self, model, output_path, batch_size=1, device='cpu', verbose=False):
        model.eval()
        model.to(device=device)
        
        dummy_input = torch.randn(
            batch_size, *self.input_shape
        ).to(device=device)
        
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
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        return onnx_model

class Imx500Exporter:
    def __init__(self, tpc_version='1.0', device='cpu'):
        if not EDGEMDT_AVAILABLE:
            raise ImportError("edgemdt_tpc module is required for Imx500Exporter but was not found.")
            
        self.tpc_version = tpc_version
        self.device = device
        self.tpc = get_target_platform_capabilities(
            tpc_version=tpc_version, 
            device_type="imx500"
        )

    def _withPatchedTorchOnnxExport(self):
        originalExport = torch.onnx.export
        exportSignature = inspect.signature(originalExport)
        supportsDynamo = "dynamo" in exportSignature.parameters

        def _patchedExport(*args, **kwargs):
            if supportsDynamo and "dynamo" not in kwargs:
                kwargs["dynamo"] = False
            return originalExport(*args, **kwargs)

        return originalExport, _patchedExport

    def _exportQuantizedModelStatic(self, quantizedModel, representative_data_gen, output_path):
        sampleInput = next(representative_data_gen())
        if isinstance(sampleInput, list):
            sampleArgs = tuple(sampleInput)
        elif isinstance(sampleInput, tuple):
            sampleArgs = sampleInput
        else:
            sampleArgs = (sampleInput,)

        # Export with fixed input shape and no dynamic axes to avoid dynamo dynamic-shape validation issues.
        torch.onnx.export(
            quantizedModel,
            sampleArgs,
            output_path,
            input_names=[f"input_{i}" for i in range(len(sampleArgs))] if len(sampleArgs) > 1 else ["input"],
            output_names=["output"],
            dynamic_axes=None,
            opset_version=20,
            verbose=False,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    
    def quantize(self, model, representative_data_gen, output_path):
        model.to(device=self.device)
        model.eval()
        
        quantized_model, quant_info = mct.ptq.pytorch_post_training_quantization(
            in_module=model,
            representative_data_gen=representative_data_gen,
            target_platform_capabilities=self.tpc
        )
        quantized_model.eval()

        originalExport, patchedExport = self._withPatchedTorchOnnxExport()
        torch.onnx.export = patchedExport
        try:
            mct.exporter.pytorch_export_model(
                model=quantized_model,
                save_model_path=output_path,
                repr_dataset=representative_data_gen,
                serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX
            )
        except Exception:
            # Fallback to fully static ONNX export path if MCT exporter hits dynamo/dynamic-shape issues.
            self._exportQuantizedModelStatic(quantized_model, representative_data_gen, output_path)
        finally:
            torch.onnx.export = originalExport
        
        return quantized_model, quant_info
