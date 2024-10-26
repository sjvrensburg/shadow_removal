# Shadow Removal

A deep learning based shadow removal pipeline that can be used both as a command-line tool and as a Python library.

The library uses the excellent [GCDRNet model](https://github.com/ZZZHANG-jx/GCDRNet/tree/main) and [weights](https://1drv.ms/f/s!Ak15mSdV3Wy4iYkeUK0TYUAajBPaBQ?e=BzXbk3), provided by Zhang et al. Please see and cite their paper if you use this code:

```
@article{zhang2023appearance,
title={Appearance Enhancement for Camera-captured Document Images in the Wild},
author={Zhang, Jiaxin and Liang, Lingyu and Ding, Kai and Guo, Fengjun and Jin, Lianwen},
journal={IEEE Transactions on Artificial Intelligence},
year={2023}}
```

Unfortunately, the authors of the [GCDRNet model](https://github.com/ZZZHANG-jx/GCDRNet/tree/main) and [weights](https://1drv.ms/f/s!Ak15mSdV3Wy4iYkeUK0TYUAajBPaBQ?e=BzXbk3) did not specify a licence.

## Installation

```bash
pip install shadow-removal
```

## Usage

### As a Command-Line Tool

```bash
# Process a single image
shadow-removal --input image.jpg --output output.jpg

# Process a directory of images
shadow-removal --input input_dir --output output_dir
```

### As a Python Library

```python
from shadow_removal import ShadowRemovalPipeline

# Initialize the pipeline
pipeline = ShadowRemovalPipeline(device="cuda")

# Process a single image
pipeline.process_image("input.jpg", "output.jpg")

# Process a directory
pipeline.process_directory("input_dir", "output_dir")
```

## Development

1. Clone the repository
2. Install poetry: `pip install poetry`
3. Install dependencies: `poetry install`
4. Run tests: `poetry run pytest`
