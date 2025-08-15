import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import pathlib


def calculate_cell_brightness(image, mask, filepath, channel, subtract_background=True, logger=None):
    """
    Calculates cell brightness and saves results to `*image-folder*/*image-name*/brightness/`

    :param image: image numpy array
    :param mask: current mask numpy array
    :param filepath: path to source image file
    :param channel: how to calculate brightness \n
        - `0` - red channel only
        - `1` - green channel only
        - `2` - blue channel only
        - `3` - average brightness of all channels
    :param subtract_background: if True, subtracts background brightness from each cell
    """
    if not image.any():
        if logger:
            logger.info("No image data provided")
        return "No image data provided"

    image = image[0]
    mask = mask[0]
    if 0 <= channel <= 2:
        brightness = image[:, :, channel]
    elif channel == 3:
        brightness = np.mean(image, axis=2)
    else:
        if logger:
            logger.info("Invalid channel specified")
        raise ValueError("Channel must be 0 (red), 1 (green), 2 (blue), or 3 (gray)")

    filename = "cell_brightness"
    channel_filename = ["red", "green", "blue", "gray"]
    filename += f"_{channel_filename[channel]}"

    # Calculating brightness of every cell
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0]
    if len(object_ids) == 0:
        if logger:
            logger.info("No cells found in the mask")
        return "No cells found in the mask"

    # Estimate background
    background_brightness = 0
    if subtract_background:
        background_mask = (mask == 0)
        if np.any(background_mask):
            background_brightness = brightness[background_mask].mean()
        else:
            background_brightness = 0

    data = []
    center_coords = []
    for obj_id in object_ids:
        obj_mask = (mask == obj_id)
        mean_brightness = brightness[obj_mask].mean() - background_brightness
        data.append({'id': obj_id, 'mean_brightness': mean_brightness})

        coords = np.column_stack(np.where(obj_mask))
        center_x = coords[:, 1].mean()
        center_y = coords[:, 0].mean()
        center_coords.append((center_x, center_y))

    data = pd.DataFrame(data)

    results_dir = os.path.splitext(filepath)[0]
    brightness_dir = os.path.join(results_dir, "brightness")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(brightness_dir):
        os.makedirs(brightness_dir)
    
    data.to_csv(os.path.join(brightness_dir, f"{filename}.csv"), index=False)

    if logger:
        logger.info("Brightness of all cells is calculated")

    # Colormap (index cells)
    colormap_mask = Image.fromarray(create_colormap_mask(mask))
    im_masks = label_image(colormap_mask, object_ids, center_coords)
    im_masks.save(os.path.join(brightness_dir, "mask_colormap.png"))

    if logger:
        logger.info("Colormap is created")

    # Brightness visualization map
    brightness_map = np.zeros_like(brightness)

    for obj_id in object_ids:
        obj_mask = (mask == obj_id)
        mean_brightness = data.loc[data['id'] == obj_id, 'mean_brightness'].values[0]
        brightness_map[obj_mask] = mean_brightness

    # Make brightness map more contrastive
    vmin = brightness_map[brightness_map > 0].min()
    vmax = brightness_map.max()
    brightness_norm = (brightness_map - vmin) / (vmax - vmin + 1e-8)
    brightness_norm = np.clip(brightness_norm, 0, 1)
    gamma = 0.5
    brightness_gamma = np.power(brightness_norm, gamma)
    brightness_map_scaled = (brightness_gamma * 255).astype(np.uint8)

    brightness_image = Image.fromarray(brightness_map_scaled)
    brightness_image = brightness_image.convert("L")
    brightness_image = label_image(brightness_image, data['mean_brightness'].map(lambda x: f"{x:.2f}"), center_coords, color=255)
    brightness_image.save(os.path.join(brightness_dir, f"{filename}_visualization_map.png"))

    if logger:
        logger.info("Brightness visualization map is created")

    return "Done! Results saved to folder"


def create_colormap_mask( mask):
    colormap = ((np.random.rand(1000000,3)*0.8+0.1)*255).astype(np.uint8)
    tmp_mask = np.copy(mask).astype(np.uint8)

    colors = colormap[:tmp_mask.max(), :3]
    cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)

    layerz = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)

    new_tmp_mask = tmp_mask[np.newaxis,...]

    layerz[...,:3] = cellcolors[new_tmp_mask[0],:]
    layerz[...,3] = 128 * (new_tmp_mask[0]>0).astype(np.uint8)

    return layerz


def label_image(image, values, coords, color=(255, 255, 255)):
    image_labeled = image.copy()

    font_path = pathlib.Path.home().joinpath(".cellpose_plus", "DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), size=20)
    
    I1 = ImageDraw.Draw(image_labeled)
    
    for value, coord in zip(values, coords):
        I1.text((coord[0], coord[1]), str(value), 
                anchor="mb",
                fill=color,
                font=font)

    return image_labeled
