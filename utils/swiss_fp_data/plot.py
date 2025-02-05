"""Plot utilities for vectorized RPLAN dataset."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .floorplan_data import Boundary, DoorWindow, SpaceName, FloorplanData


def convert_to_2D_ring(polygon):
    lr = polygon[:, :2]
    return lr


def plot_space(
    space_ring,
    name,
    color,
    marker="o",
    fill=True,
    rotate_text=False,
    text_size=6,
    **kwargs,
):
    """Plot the space on existing or new plot."""
    lr = space_ring
    ax = plt.gca()
    plt.plot(
        *lr.T, markersize=4, marker=marker, mfc="white", color=color, zorder=1, **kwargs
    )
    for p in lr:
        plt.text(
            *p,
            f"{(p[0]/ 1000):.02f},{(p[1]/ 1000):.02f}",
            horizontalalignment="center",
            size=3,
        )
    if fill:
        ax.fill(*lr.T, color=color, alpha=0.2)
    plt.text(
        *np.mean(lr[1:], axis=0),
        name,
        rotation=0 if not rotate_text else 90,
        horizontalalignment="center",
        verticalalignment="center",
        size=text_size,
    )
    return ax


def plot_dxf(data: FloorplanData, fig_name):
    """Plot the DXF data."""
    try:
        boundary = data.perimeter
        bx, by, _ = boundary.max(axis=0) + 0.1
        fig, ax = plt.subplots(figsize=(bx / 1000, by / 1000))
        ring = convert_to_2D_ring(boundary)
        plot_space(
            ring,
            None,
            "#000000",
            fill=False,
            lw=2,
            ls="-",
        )
    except (TypeError, AttributeError):
        bx, by = 20, 20
        fig, ax = plt.subplots(figsize=(10, 10))

    for s, p in data.spaces.items():
        color = SpaceName[s.split("-")[0]].value["color"]
        max_x, max_y, _ = p.max(axis=0)
        min_x, min_y, _ = p.min(axis=0)
        rotate = (max_x - min_x) < (max_y - min_y)
        ring = convert_to_2D_ring(p)
        plot_space(ring, s, color, fill=True, text_size=9, rotate_text=rotate)

    for d, p in data.door_windows.items():
        color = DoorWindow[d.split("-")[0]].value["color"]
        max_x, max_y, _ = p.max(axis=0)
        min_x, min_y, _ = p.min(axis=0)
        rotate = (max_x - min_x) < (max_y - min_y)
        ring = convert_to_2D_ring(p)
        plot_space(ring, d, color, rotate_text=rotate)

    for bo, p in data.boundaries.items():
        color = Boundary[bo.split("-")[0]].value["color"]
        ring = convert_to_2D_ring(p)
        plot_space(ring, None, color)

    if len(data.errors) > 0:
        message = "DXF file has the following errors:\n"
        for i, error in enumerate(data.errors):
            message += f"{i+1:02d}: {error}\n"
        plt.text(
            0.1,
            0.1,
            message,
            fontsize=12,
            color="black",
            va="bottom",
            alpha=1.0,
            transform=plt.gcf().transFigure,
            bbox=dict(
                facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.5"
            ),
        )

    plt.xlim(-0.1, bx)
    plt.ylim(-0.1, by)
    plt.axis("off")
    fig.patch.set_visible(False)
    plt.savefig(fig_name, dpi=10, bbox_inches="tight")
    plt.close()
