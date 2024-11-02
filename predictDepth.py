import torch
import cv2

import cv2
import torch

def prepare_image(rgb_file, input_size=(616, 1064), padding_values=[123.675, 116.28, 103.53]):
    """
    Prepares an RGB image for model input.

    Parameters:
        rgb_file (str): Path to the RGB image file.
        input_size (tuple): Desired input size (height, width).
        padding_values (list): Values for padding the image.

    Returns:
        torch.Tensor: Preprocessed RGB image ready for model input.
    """
    # Load RGB image
    rgb_origin = cv2.imread(rgb_file)
    if rgb_origin is None:
        raise ValueError(f"Image not found or could not be loaded: {rgb_file}")

    rgb_origin = rgb_origin[:, :, ::-1]  # Convert BGR to RGB

    # Resize while keeping the aspect ratio
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Pad to the desired input size
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding_values)

    # Normalize the image
    mean = torch.tensor(padding_values).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb_tensor = torch.div((rgb_tensor - mean), std)
    rgb_tensor = rgb_tensor[None, :, :, :]  # Add batch dimension and move to GPU

    return rgb_tensor

rgb = cv2.imread('park.png')
rgb = prepare_image('park.png')
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
