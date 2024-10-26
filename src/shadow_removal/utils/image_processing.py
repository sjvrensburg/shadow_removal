import cv2
import numpy as np
import torch


def preprocess_image(image, target_size=None, stride=32):
    """Preprocess image for model input."""
    h, w = image.shape[:2]

    # Pad image to be divisible by stride
    if h % stride != 0:
        pad_h = stride - (h % stride)
        image = cv2.copyMakeBorder(image, pad_h, 0, 0, 0, cv2.BORDER_REPLICATE)

    if w % stride != 0:
        pad_w = stride - (w % stride)
        image = cv2.copyMakeBorder(image, 0, 0, pad_w, 0, cv2.BORDER_REPLICATE)

    if target_size:
        image = cv2.resize(image, (target_size, target_size))

    # Convert to tensor
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
    tensor = tensor.unsqueeze(0)

    return tensor, pad_h if 'pad_h' in locals() else 0, pad_w if 'pad_w' in locals() else 0


def postprocess_image(tensor, padding_h=0, padding_w=0):
    """Convert tensor back to image."""
    image = tensor[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    if padding_h > 0 or padding_w > 0:
        image = image[padding_h:, padding_w:]

    return image
