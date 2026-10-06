"""
NAFNet Standalone Deblur Runner
Run image deblurring using pretrained NAFNet models on GPU or CPU.

Usage:
    python run_deblur.py --input demo/blurry.jpg --output demo/deblur_result.png --model gopro
    python run_deblur.py --input path/to/images/ --output path/to/outputs/ --model gopro
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path

# Ensure NAFNet root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import torch
import torch.nn.functional as F
from basicsr.models import create_model
from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


MODEL_CONFIGS = {
    'gopro': {
        'config': os.path.join(ROOT_DIR, 'options/test/GoPro/NAFNet-width64.yml'),
        'weights': os.path.join(ROOT_DIR, 'experiments/pretrained_models/NAFNet-GoPro-width64.pth'),
        'name': 'NAFNet-GoPro-width64',
        'desc': 'GoPro Deblur Model (Camera shake & fast motion)'
    },
    'gopro-small': {
        'config': os.path.join(ROOT_DIR, 'options/test/GoPro/NAFNet-width32.yml'),
        'weights': os.path.join(ROOT_DIR, 'experiments/pretrained_models/NAFNet-GoPro-width32.pth'),
        'name': 'NAFNet-GoPro-width32',
        'desc': 'GoPro Light Deblur Model (Width 32, faster)'
    },
    'reds': {
        'config': os.path.join(ROOT_DIR, 'options/test/REDS/NAFNet-width64.yml'),
        'weights': os.path.join(ROOT_DIR, 'experiments/pretrained_models/NAFNet-REDS-width64.pth'),
        'name': 'NAFNet-REDS-width64',
        'desc': 'REDS Deblur Model (Video motion blur with JPEG compression)'
    }
}


def load_deblur_model(model_key='gopro', device=None):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_CONFIGS.keys())}")

    cfg_info = MODEL_CONFIGS[model_key]
    config_path = cfg_info['config']
    weights_path = cfg_info['weights']

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at: {weights_path}\n"
            f"Please download the weights into 'experiments/pretrained_models/'."
        )

    opt = parse(config_path, is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1 if (torch.cuda.is_available() and device != 'cpu') else 0
    opt['path']['pretrain_network_g'] = weights_path

    model = create_model(opt)
    return model, cfg_info


def deblur_image(model, img_bgr, pad_factor=64):
    """
    Deblur an image using the NAFNet model.
    Handles padding to multiples of pad_factor (e.g. 64) and unpads result.
    """
    # Convert BGR (cv2) to RGB float tensor [0, 1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = img2tensor(img_rgb / 255.0, bgr2rgb=False, float32=True).unsqueeze(0)

    _, _, h, w = img_t.shape
    pad_h = (pad_factor - h % pad_factor) % pad_factor
    pad_w = (pad_factor - w % pad_factor) % pad_factor

    if pad_h > 0 or pad_w > 0:
        img_padded = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        img_padded = img_t

    # Feed data and run inference
    model.feed_data(data={'lq': img_padded})
    if model.opt.get('val', {}).get('grids', False):
        model.grids()

    model.test()

    if model.opt.get('val', {}).get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    output_tensor = visuals['result']

    # Unpad back to original dimensions
    output_tensor = output_tensor[:, :, :h, :w]

    # Convert back to uint8 BGR numpy array
    output_img = tensor2img([output_tensor], rgb2bgr=True)
    return output_img


def process_file(model, input_path, output_path):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"[-] Error: Failed to read image '{input_path}'")
        return False

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    restored = deblur_image(model, img)
    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imwrite(restored, str(output_path))
    print(f"[+] Restored: {input_path} ({w}x{h}) -> {output_path} [{elapsed:.3f}s]")
    return True


def main():
    parser = argparse.ArgumentParser(description="NAFNet Deblur Inference Runner")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path to an input image or a folder of images.")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Path to save output image or directory. Defaults to 'output_<filename>'")
    parser.add_argument('--model', '-m', type=str, default='gopro',
                        choices=['gopro', 'gopro-small', 'reds'],
                        help="Deblur model to use (default: 'gopro').")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Device to use (default: 'cuda').")
    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] Running on GPU: {gpu_name}")
    else:
        print("[*] Running on CPU")

    print(f"[*] Loading model: {args.model} ({MODEL_CONFIGS[args.model]['desc']})...")
    model, cfg_info = load_deblur_model(args.model, device=args.device)
    print(f"[+] Model loaded successfully!")

    input_path = Path(args.input)

    # Process directory
    if input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_extensions])

        if not image_files:
            print(f"[-] No valid images found in directory: {input_path}")
            return

        out_dir = Path(args.output) if args.output else input_path / 'deblur_results'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Found {len(image_files)} image(s) to process. Outputs will be saved to: {out_dir}")
        for img_file in image_files:
            out_file = out_dir / f"{img_file.stem}_deblurred.png"
            process_file(model, img_file, out_file)

    # Process single image
    elif input_path.is_file():
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = input_path.parent / f"{input_path.stem}_deblurred.png"
        process_file(model, input_path, out_path)
    else:
        print(f"[-] Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
