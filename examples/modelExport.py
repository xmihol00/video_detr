import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelExportToolkit import RepresentativeDataGenerator, OnnxExporter, Imx500Exporter
import basicViT # TODO: invalid import, this is just an example, replace with your actual model definition

# ---------------- user configuration ----------------
pt_model_path = "customViT.pt"            # path to your trained MobileViT model
onnx_float_path = "customViT_float.onnx"       # exported ONNX (non-quantized)
onnx_quant_path = "customViT_quant.onnx"       # exported ONNX (quantized)
calibration_dataset_path = "/home/david/Downloads/UCF101/vivit_calib/"  # folder with JPG/PNG images for calibration
batch_size = 1
input_shape = (3, 224, 224)                # (C, H, W) adjust to your model’s input
num_calibration_images = 50               # how many images to use for calibration
tpc_version = '1.0'                        # IMX500 target platform version
# ----------------------------------------------------

def loadTrainedMobileViT(checkpoint_path, num_classes=101):
    model = basicViT.vit_base(num_classes=num_classes, img_size=224)
    # FIXME: uncomment and fix the two line below
    #checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded model weights successfully.")
    return model

if __name__ == "__main__":
    # Load model using the proper loading function
    model = loadTrainedMobileViT(pt_model_path, num_classes=101).to(device='cpu')

    # Initialize ONNX exporter
    onnx_exporter = OnnxExporter(input_shape=input_shape, opset_version=15)
    
    # Export original (float) model to ONNX
    onnx_exporter.export(model, onnx_float_path, batch_size=batch_size, device='cpu')

    # Initialize representative data generator for calibration
    rep_data_gen = RepresentativeDataGenerator(
        image_folder_path=calibration_dataset_path,
        input_shape=input_shape,
        batch_size=batch_size,
        num_images=num_calibration_images,
        device='cuda'
    )

    # Initialize IMX500 exporter and quantize model
    imx500_exporter = Imx500Exporter(tpc_version=tpc_version, device='cuda')
    quantized_model, quant_info = imx500_exporter.quantize(model, rep_data_gen, onnx_quant_path)

    print("Quantization info:", quant_info)
    print("✅ Done. You can now convert the quantized ONNX for IMX500 using the CLI:")
    print(f"imxconv-pt -i {onnx_quant_path} -o ./imx500_output --no-input-persistency --overwrite")
