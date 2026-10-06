# NAFNet Deblur Setup & Execution Walkthrough

The **NAFNet** repository has been configured, dependencies and official pretrained weights have been downloaded, compatibility fixes for Python 3.13 on Windows have been applied, and image deblurring has been executed on your **NVIDIA GeForce RTX 4090** GPU.

---

## Changes Made

### 1. Environment & Git Trust Configuration
- Configured Git's `safe.directory` for `d:/research/NAFNet` and `//192.168.15.10/d/research/NAFNet` to eliminate repository ownership warnings.
- Verified that all required dependencies (`torch 2.10.0+cu130`, `torchvision`, `opencv-python`, `numpy`, `scipy`, `pyyaml`, `pillow`, `tqdm`, `lmdb`, `addict`, etc.) in `C:\ComfyUI_portable\python_embeded\python.exe` are operational with CUDA enabled.

### 2. Python 3.13 & Setuptools Compatibility Fixes
- **[setup.py](file:///d:/research/NAFNet/setup.py)**:
  - Fixed `KeyError: '__version__'` caused by Python 3.13's scoping changes with `exec()` and `locals()`.
  - Passed `USERPROFILE` to Git subprocess calls so `.gitconfig` is recognized on Windows.
  - Added environment variable checks (`NO_CUDA_EXT` / `BASICSR_EXT=none`) to avoid requiring MSVC / CUDA Toolkit compiler for pure-PyTorch NAFNet models.
- **[basicsr/\_\_init\_\_.py](file:///d:/research/NAFNet/basicsr/__init__.py)**:
  - Created the missing package `__init__.py` so setuptools and Python recognize `basicsr` as an importable module.
  - Successfully installed NAFNet in editable mode: `pip install -e . --no-deps --no-build-isolation`.
- **[basicsr/demo.py](file:///d:/research/NAFNet/basicsr/demo.py)**:
  - Added `sys.path` resolution so the script can run from any working directory without `ModuleNotFoundError`.

### 3. Pretrained Models Downloaded
Downloaded the official weights into [experiments/pretrained_models](file:///d:/research/NAFNet/experiments/pretrained_models):
- `NAFNet-GoPro-width64.pth` (271.8 MB) — SOTA model for camera shake and fast motion blur (GoPro benchmark).
- `NAFNet-REDS-width64.pth` (271.8 MB) — SOTA model for video motion blur with JPEG compression artifacts (REDS benchmark).
- `NAFNet-GoPro-width32.pth` (68.7 MB) — Lightweight and faster GoPro model.

### 4. Streamlined Deblur CLI Runner
- Created **[run_deblur.py](file:///d:/research/NAFNet/run_deblur.py)**:
  - Works with single images or entire folders of images.
  - Automatically handles image padding (reflection padding to multiples of 64) and unpads afterwards, supporting any arbitrary input resolution.
  - Simplifies model selection (`--model gopro`, `--model reds`, `--model gopro-small`).

---

## Verification & Test Results

### 1. Official Demo Execution
Tested both GoPro and REDS models with the official runner on `demo/blurry.jpg`:
```powershell
# GoPro Model
python basicsr/demo.py -opt options/test/GoPro/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_gopro_official.png

# REDS Model
python basicsr/demo.py -opt options/test/REDS/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_reds_official.png
```
**Result**: Both exited with code 0 and generated valid, restored images.

### 2. Streamlined Runner Execution (`run_deblur.py`)
Tested across all 3 models using the NVIDIA GeForce RTX 4090:
```powershell
python run_deblur.py -i demo/blurry.jpg -o demo/deblur_gopro_64.png -m gopro
python run_deblur.py -i demo/blurry.jpg -o demo/deblur_gopro_32.png -m gopro-small
python run_deblur.py -i demo/blurry.jpg -o demo/deblur_reds_64.png -m reds
```

**Benchmark Results on RTX 4090 (1280x720 HD image)**:
| Model | Type | File Size | Inference Time | Output File |
| :--- | :--- | :--- | :--- | :--- |
| `NAFNet-GoPro-width64` | Full SOTA | 1.22 MB | **0.362s** | [deblur_gopro_64.png](file:///d:/research/NAFNet/demo/deblur_gopro_64.png) |
| `NAFNet-GoPro-width32` | Fast / Compact | 1.11 MB | **0.264s** | [deblur_gopro_32.png](file:///d:/research/NAFNet/demo/deblur_gopro_32.png) |
| `NAFNet-REDS-width64` | JPEG / Video | 1.33 MB | **0.356s** | [deblur_reds_64.png](file:///d:/research/NAFNet/demo/deblur_reds_64.png) |

---

## Quick Reference: How to Run Deblurring

### Single Image Deblur
```powershell
# Standard motion / camera shake deblur (GoPro):
python run_deblur.py -i path/to/blurry_image.jpg -o path/to/sharp_image.png

# Video motion blur with compression artifacts (REDS):
python run_deblur.py -i path/to/blurry_image.jpg -o path/to/sharp_image.png -m reds
```

### Batch Deblur an Entire Folder
```powershell
python run_deblur.py -i path/to/blurry_folder/ -o path/to/output_folder/ -m gopro
```
