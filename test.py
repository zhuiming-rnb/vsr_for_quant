import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LFSRNet
from dataset import LightFieldTestDataset, get_folder_list
from utils import (load_config, compute_psnr, compute_ssim,
                   save_light_field, AverageMeter, patch_inference,
                   remap_legacy_state_dict)


MODEL_INPUT_NAME = "lr_input"
MODEL_OUTPUT_NAME = "sr_output"
TRT_BATCH_SIZE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Light Field SR Testing')
    parser.add_argument('--lr_path', type=str, default='vsr_for_quant/0403_8x_8x_test_small',
                        help='Path to LR test data folders')
    parser.add_argument('--config_path', type=str, default='vsr_for_quant/config.json',
                        help='Path to config.json')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to GT data folders (optional, for metric computation)')
    parser.add_argument('--model_path', type=str, default='vsr_for_quant/checkpoint_epoch_1000.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_path', type=str, default='vsr_for_quant/test_output',
                        help='Path to save SR output images')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device id to use (default: 0)')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Override inference patch size from config')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Override inference overlap from config')
    parser.add_argument('--trt_fp16', action='store_true',
                        help='Use TensorRT FP16 inference (requires ONNX export, may fail with Transformer models)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster inference (recommended for Transformer models)')
    return parser.parse_args()


def load_model(config, model_path, device):
    """Load the model from checkpoint (compatible with old and new model keys)"""
    model = LFSRNet(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    # Handle both direct state_dict and checkpoint dict
    if 'model_state_dict' in checkpoint:
        state_dict = remap_legacy_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        epoch_info = checkpoint.get('epoch', 'unknown')
        print(f'Loaded checkpoint from epoch {epoch_info}')
    else:
        state_dict = remap_legacy_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        print('Loaded model state dict directly')
    model = model.to(device)
    model.eval()
    return model

import torch.backends.mha
torch.backends.mha.set_fastpath_enabled(False)

def _export_model_to_onnx(model, args, onnx_path, scale_x, scale_y):
    """Export model to ONNX format for TRT"""
    import onnx

    config = load_config(args.config_path)
    num_views = config['data']['num_views']
    patch_size = args.patch_size if args.patch_size else 64
    H, W = patch_size, patch_size

    sample_input = torch.randn(
        TRT_BATCH_SIZE,
        num_views,
        1,
        H,
        W,
        device="cpu",
        dtype=torch.float32,
    )

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            input_names=[MODEL_INPUT_NAME],
            output_names=[MODEL_OUTPUT_NAME],
            opset_version=17,
            dynamo=False,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def _load_or_build_trt_engine(onnx_path, engine_path, compiled_model_path):
    """Build or load TRT FP16 engine"""
    from modelopt.torch._deploy._runtime import RuntimeRegistry
    from modelopt.torch._deploy.utils.torch_onnx import OnnxBytes

    if os.path.exists(compiled_model_path):
        with open(compiled_model_path, "rb") as f:
            compiled_model = f.read()
        if not os.path.exists(engine_path):
            with open(engine_path, "wb") as f:
                f.write(compiled_model[8:])
        return compiled_model

    client = RuntimeRegistry.get({"runtime": "TRT", "accelerator": "GPU", "precision": "fp16"})
    onnx_bytes = OnnxBytes(onnx_path).to_bytes()
    temp_engine_path = engine_path + ".tmp"
    compiled_model = client.ir_to_compiled(
        onnx_bytes,
        {"engine_path": temp_engine_path},
    )

    with open(compiled_model_path, "wb") as f:
        f.write(compiled_model)
    with open(engine_path, "wb") as f:
        f.write(compiled_model[8:])

    return compiled_model


class TRTBackend:
    """TensorRT backend wrapper"""
    def __init__(self, client, compiled_model, io_shapes):
        from modelopt.torch._deploy._runtime import TRTBackend as TRTBackendClass
        self.backend = TRTBackendClass(client, compiled_model, io_shapes)

    def run(self, input_tensor):
        return self.backend.run(input_tensor)


def build_trt_fp16_model(args, model, scale_x, scale_y):
    """Build TRT FP16 engine and return backend"""
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT FP16 inference requires CUDA.")

    os.makedirs(args.output_path, exist_ok=True)

    onnx_path = os.path.join(args.output_path, "vsr_model.onnx")
    compiled_model_path = os.path.join(args.output_path, "vsr_model.modelopt.engine")
    engine_path = os.path.join(args.output_path, "vsr_model.engine")

    checkpoint_mtime = os.path.getmtime(args.model_path)
    onnx_is_stale = (not os.path.exists(onnx_path)) or os.path.getmtime(onnx_path) < checkpoint_mtime

    if onnx_is_stale:
        print(f"Exporting ONNX model to: {onnx_path}")
        _export_model_to_onnx(model, args, onnx_path, scale_x, scale_y)

    if not os.path.exists(compiled_model_path) or os.path.getmtime(compiled_model_path) < os.path.getmtime(onnx_path):
        print(f"Building TensorRT FP16 engine from: {onnx_path}")
        _load_or_build_trt_engine(onnx_path, engine_path, compiled_model_path)

    from modelopt.torch._deploy._runtime import RuntimeRegistry
    client = RuntimeRegistry.get({"runtime": "TRT", "accelerator": "GPU", "precision": "fp16"})

    config = load_config(args.config_path)
    num_views = config['data']['num_views']
    patch_size = args.patch_size if args.patch_size else 64
    H, W = patch_size, patch_size

    io_shapes = {
        MODEL_INPUT_NAME: [TRT_BATCH_SIZE, num_views, 1, H, W],
        MODEL_OUTPUT_NAME: [TRT_BATCH_SIZE, num_views, 1, H * scale_y, W * scale_x],
    }

    with open(compiled_model_path, "rb") as f:
        compiled_model = f.read()

    backend = TRTBackend(client, compiled_model, io_shapes)

    sample_input = torch.randn(
        TRT_BATCH_SIZE,
        num_views,
        1,
        H,
        W,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.inference_mode():
        torch_output = model(sample_input)
        trt_output = backend.run(sample_input)

    max_abs_diff = (torch_output.float() - trt_output.float()).abs().max().item()
    mean_abs_diff = (torch_output.float() - trt_output.float()).abs().mean().item()
    print(f"TensorRT sanity check: mean abs diff={mean_abs_diff:.6e}, max abs diff={max_abs_diff:.6e}")

    return backend


def test(args):
    # ---- Load config ----
    config = load_config(args.config_path)

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)
        print(f'Using GPU: {args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # ---- Inference patch settings ----
    inference_cfg = config.get('inference', {})
    patch_size = args.patch_size if args.patch_size is not None else inference_cfg.get('patch_size', None)
    overlap = args.overlap if args.overlap is not None else inference_cfg.get('overlap', None)
    use_patch_inference = patch_size is not None and overlap is not None
    if use_patch_inference:
        print(f'Using patch inference: patch_size={patch_size}, overlap={overlap}')
    else:
        print('Using full-image inference')

    scale_x = config['upsampling']['scale_x']
    scale_y = config['upsampling']['scale_y']
    num_views = config['data']['num_views']

    # ---- Load model ----
    model = load_model(config, args.model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    # ---- TRT FP16 / torch.compile setup ----
    trt_backend = None
    if args.trt_fp16:
        print("Building TensorRT FP16 model...")
        model = model.cpu().float().eval()
        trt_backend = build_trt_fp16_model(args, model, scale_x, scale_y)
        model = None  # TRT backend will be used instead
    elif args.compile:
        print("Compiling model with torch.compile (FP16)...")
        model = model.to(device).half().eval()
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully")

    # ---- Build dataset & dataloader ----
    has_gt = args.gt_path is not None and os.path.isdir(args.gt_path)
    folder_list = get_folder_list(args.lr_path)
    print(f'Found {len(folder_list)} test samples')

    if has_gt:
        dataset = LightFieldTestDataset(
            lr_path=args.lr_path,
            gt_path=args.gt_path,
            num_views=num_views,
            folder_list=folder_list
        )
    else:
        # When no GT is available, still use LightFieldTestDataset but with
        # lr_path as gt_path (gt won't be used for saving, only structurally needed)
        dataset = LightFieldTestDataset(
            lr_path=args.lr_path,
            gt_path=args.lr_path,  # placeholder, gt won't be evaluated
            num_views=num_views,
            folder_list=folder_list
        )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # ---- Output directory ----
    os.makedirs(args.output_path, exist_ok=True)

    # ---- Run inference ----
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Testing'):
            lr = batch['lr'].to(device)   # (1, V, 1, H, W)
            gt = batch['gt'].to(device)   # (1, V, 1, H_gt, W_gt)
            name = batch['name'][0]

            # Forward pass
            if trt_backend is not None:
                lr_fp16 = lr.half()
                pred = trt_backend.run(lr_fp16)
                pred = pred.float()
            elif args.compile:
                lr_fp16 = lr.half()
                pred = model(lr_fp16)
                pred = pred.float()
            elif use_patch_inference:
                pred = patch_inference(model, lr, patch_size, overlap,
                                       scale_x, scale_y, device)
            else:
                pred = model(lr)

            # Compute metrics if GT is available
            if has_gt:
                B, V, C, H, W = pred.shape
                pred_flat = pred.reshape(B * V, C, H, W)
                gt_flat = gt.reshape(B * V, C, H, W)
                psnr_val = compute_psnr(pred_flat, gt_flat)
                ssim_val = compute_ssim(pred_flat, gt_flat)
                psnr_meter.update(psnr_val.item())
                ssim_meter.update(ssim_val.item())
                tqdm.write(f'{name}  PSNR: {psnr_val.item():.4f} dB  '
                           f'SSIM: {ssim_val.item():.6f}')

            # Save SR output
            save_dir = os.path.join(args.output_path, name)
            save_light_field(pred[0], save_dir, num_views)

    # ---- Print summary ----
    print('\n' + '=' * 60)
    print(f'Results saved to: {args.output_path}')
    print(f'Total samples: {len(folder_list)}')
    if has_gt:
        print(f'Average PSNR: {psnr_meter.avg:.4f} dB')
        print(f'Average SSIM: {ssim_meter.avg:.6f}')

        # Save metrics to txt
        metrics_path = os.path.join(args.output_path, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f'Model: {args.model_path}\n')
            f.write(f'Test LR: {args.lr_path}\n')
            f.write(f'Test GT: {args.gt_path}\n')
            f.write(f'Patch size: {patch_size}, Overlap: {overlap}\n')
            f.write(f'Scale: {scale_x}x{scale_y}\n')
            f.write(f'Num samples: {len(folder_list)}\n')
            f.write(f'Average PSNR: {psnr_meter.avg:.4f} dB\n')
            f.write(f'Average SSIM: {ssim_meter.avg:.6f}\n')
        print(f'Metrics saved to: {metrics_path}')
    print('=' * 60)


if __name__ == '__main__':
    args = parse_args()
    test(args)