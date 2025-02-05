import triangle
from shapely.geometry import Polygon, LinearRing
from tripy import earclip
import numpy as np


def triangulate_polygon_with_holes(shapely_polygon: Polygon):
    """
    Triangulates a Shapely polygon with optional holes using the Triangle library.
    Returns a list of triangles, each represented as a triplet of (x, y) points.

    Args:
        shapely_polygon (Polygon): A Shapely Polygon object with or without holes.

    Returns:
        List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        List of triangles as vertex triplets.
    """
    if not isinstance(shapely_polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon.")

    # Ensure the polygon is valid
    if not shapely_polygon.is_valid:
        shapely_polygon = shapely_polygon.buffer(0)
        if not shapely_polygon.is_valid:
            raise ValueError("Polygon is invalid and could not be fixed.")

    # Prepare outer boundary
    points = list(LinearRing(shapely_polygon.exterior).coords)[
        :-1
    ]  # Remove duplicate closing point
    segments = [(i, i + 1) for i in range(len(points) - 1)] + [
        (len(points) - 1, 0)
    ]  # Closing segment

    # Prepare holes and inner boundaries
    holes = []
    for i, interior in enumerate(shapely_polygon.interiors):
        hole_polygon = Polygon(interior)
        if not hole_polygon.is_valid:
            hole_polygon = hole_polygon.buffer(0)
        if not hole_polygon.is_valid or hole_polygon.is_empty:
            continue

        hole_points = list(hole_polygon.exterior.coords)[:-1]
        holes.append(list(hole_polygon.centroid.coords)[0])  # Centroid of the hole

        # Add hole vertices and segments
        start_idx = len(points)
        points.extend(hole_points)
        segments.extend(
            [(start_idx + i, start_idx + i + 1) for i in range(len(hole_points) - 1)]
        )
        segments.append(
            (start_idx + len(hole_points) - 1, start_idx)
        )  # Closing segment

    # Prepare Triangle input
    data = {
        "vertices": points,
        "segments": segments,
        "holes": holes,
    }
    # Perform triangulation
    try:
        tri = triangle.triangulate(data, "p")  # "p" option respects holes
    except Exception as e:
        # return np.array(earclip(shapely_polygon.exterior.coords))  # Fallback to tripy if Triangle fails
        raise ValueError(f"Triangulation failed: {e}")

    # Extract triangles as triplets of (x, y) vertices
    triangles = [
        tuple(tuple(tri["vertices"][i]) for i in triangle)
        for triangle in tri.get("triangles", [])
    ]
    return triangles
