"""Creates mask using extracted ML data."""

import matplotlib
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .helper import SpaceName, DoorWindow
from .floorplan_data import FloorplanData

MAX_SIZE = 1024


def create_overlay_image(
    floorplan_data: FloorplanData,
    base_image: Image,
):
    base_image = resize_image(base_image, MAX_SIZE)

    # create space mask
    space_mask = get_space_mask(
        floorplan_data,
    )
    space_mask = space_mask.resize(base_image.size)
    base_image = Image.blend(base_image, space_mask, alpha=0.5)

    dw_mask = get_dw_mask(
        floorplan_data,
    )
    dw_mask = dw_mask.resize(base_image.size)
    base_image = Image.blend(base_image, dw_mask, alpha=0.5)

    # create perimeter mask
    perimeter_mask = get_perimeter_mask(
        floorplan_data,
    )
    perimeter_mask = perimeter_mask.resize(base_image.size)
    perimeter_mask = perimeter_mask.convert(
        "L",
    )
    base_image.paste(perimeter_mask, (0, 0), mask=perimeter_mask)

    return base_image


def resize_image(img, max_size):
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img_resized


def draw_polygon(draw, polygon, height, color):
    """Draw the polygon."""
    x, y = polygon[:, 0], polygon[:, 1]
    y = [height - yi for yi in y]
    draw.polygon(list(zip(x, y)), fill=color, width=0)


def get_perimeter_mask(
    data: FloorplanData,
):
    """Plot the DXF data."""
    boundary = (data.perimeter[:, :2]).astype(int)
    bxy = list(boundary.max(axis=0))
    img = Image.new("RGBA", bxy, "white")
    draw = ImageDraw.Draw(img)

    draw_polygon(draw, boundary, bxy[1], "black")
    return img


def get_space_mask(
    data: FloorplanData,
):
    """Plot the DXF data."""
    boundary = (data.perimeter[:, :2]).astype(int)
    bxy = list(boundary.max(axis=0))
    img = Image.new("RGBA", bxy, "black")
    draw = ImageDraw.Draw(img)

    for s, p in data.spaces.items():
        color = SpaceName[s.split("-")[0]].value["color"]
        draw_polygon(draw, (p[:, :2]).astype(int), bxy[1], color)
    return img


def get_dw_mask(
    data: FloorplanData,
):
    """Plot the DXF data."""
    boundary = (data.perimeter[:, :2]).astype(int)
    bxy = list(boundary.max(axis=0))
    img = Image.new("RGBA", bxy, "black")
    draw = ImageDraw.Draw(img)

    for s, p in data.door_windows.items():
        color = DoorWindow[s.split("-")[0]].value["color"]
        draw_polygon(draw, (p[:, :2]).astype(int), bxy[1], color)
    return img
