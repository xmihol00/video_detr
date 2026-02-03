# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

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


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='number of warmup epochs')
    parser.add_argument('--warmup_start_lr', default=1e-6, type=float,
                        help='initial learning rate for warmup')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help="Path to the pretrained model checkpoint. If set, loads full model weights for training. Overrides --pretrained_backbone.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    #parser.add_argument('--pretrained_backbone', default="resnet50-19c8e357.pth", type=str,
    parser.add_argument('--pretrained_backbone', default="yolov8n.pt", type=str,
                        help="Pass a backbone pretrained on ImageNet")
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help="If true, we freeze the backbone weights during training")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default="/mnt/matylda5/xmihol00/yolov8/datasets/coco/", type=str)
    parser.add_argument('--num_classes', default=91, type=int,
                        help='number of classes for the dataset')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--square_images', action='store_true',
                        help="If true, images are resized to square during training")

    parser.add_argument('--output_dir', default='weights/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def register_hooks(model):
    hooks = []

    def hook(module, inp, out):
        cls = module.__class__.__name__

        in_shapes = [
            tuple(x.shape) for x in inp if torch.is_tensor(x)
        ]

        if torch.is_tensor(out):
            out_shape = tuple(out.shape)
        elif isinstance(out, (list, tuple)):
            out_shape = [tuple(x.shape) for x in out if torch.is_tensor(x)]
        else:
            out_shape = type(out)

        param_shapes = {
            name: tuple(p.shape)
            for name, p in module.named_parameters(recurse=False)
        }

        buffer_shapes = {
            name: tuple(b.shape)
            for name, b in module.named_buffers(recurse=False)
        }

        print(f"{cls}")
        print(f"  input:  {in_shapes}")
        print(f"  output: {out_shape}")
        if param_shapes:
            print(f"  params: {param_shapes}")
        if buffer_shapes:
            print(f"  buffers:{buffer_shapes}")

    for m in model.modules():
        if len(list(m.children())) == 0:  # leaf modules only
            hooks.append(m.register_forward_hook(hook))

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    out = model(dummy_input)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # Create warmup + step decay scheduler
    # Warmup: linearly increase LR from warmup_start_lr to lr over warmup_epochs
    # Then: step decay at lr_drop epoch
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=args.warmup_start_lr / args.lr,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        # No warmup, use only step decay scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.pretrained_path is not None:
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Pretrained model loaded successfully")

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
