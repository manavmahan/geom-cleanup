"""Custom utility functions."""

import numpy as np
from skspatial.objects import Line
import math
import shapely


def get_projected_line(first: shapely.LineString, second: shapely.LineString):
    """Return the projected line from first to second line if parallel."""
    p1 = projection_line(first.coords[0], second)
    p2 = projection_line(first.coords[1], second)
    if p1.length != p2.length:
        return None
    return p1, p2


def get_line_projection(line: shapely.LineString, other: shapely.LineString):
    """Return the projection of line on other line."""
    p1 = project_point(line.coords[0], other)
    p2 = project_point(line.coords[1], other)
    return shapely.LineString([p1, p2])


def angle_between_lines(line1, line2):
    # Extract the direction vectors of the lines
    x1, y1 = (
        line1.coords[-1][0] - line1.coords[0][0],
        line1.coords[-1][1] - line1.coords[0][1],
    )
    x2, y2 = (
        line2.coords[-1][0] - line2.coords[0][0],
        line2.coords[-1][1] - line2.coords[0][1],
    )

    # Calculate the dot product of the vectors
    dot_product = x1 * x2 + y1 * y2

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = math.sqrt(x1**2 + y1**2)
    magnitude2 = math.sqrt(x2**2 + y2**2)

    # Calculate the cosine of the angle between the vectors
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # Get the angle in radians and then convert to degrees
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def are_parallel(first: shapely.LineString, second: shapely.LineString):
    p1, p2 = first.coords
    d1 = projection_distance(p1, second)
    d2 = projection_distance(p2, second)
    if d1 == d2:
        return True, d1
    return False, angle_between_lines(first, second)


def line_intersection(l1, l2):
    """Calculate line intersection by projection."""
    x = np.array(l1.coords[-1])
    u = np.array(l2.coords[0])
    v = np.array(l2.coords[-1])
    n = v - u

    n /= np.linalg.norm(n, 2)
    p = u + n * np.dot(x - u, n)
    return shapely.Point(p)


def is_vertical_or_horizontal(line):
    if line.coords[0][0] == line.coords[1][0]:
        return True
    return line.coords[0][1] == line.coords[1][1]


def project_point(point, line: shapely.LineString):
    if isinstance(point, tuple):
        x = np.array(point)
    elif isinstance(point, shapely.Point):
        x = np.array(point.coords[-1])
    else:
        x = point

    try:
        line = Line.from_points(point_a=line.coords[0], point_b=line.coords[1])
        projected_point = line.project_point(x)
        return np.array(projected_point)
    except ValueError as e:
        return None


def projection_line(p: shapely.Point, line: shapely.LineString):
    projected_point = project_point(p, line)
    if projected_point is None:
        return None
    return shapely.LineString((p, projected_point))


def projection_distance(x, line):
    projected_line = projection_line(x, line)
    return projected_line.length if projected_line is not None else None


def simplify_small_lines(polygon, tolerance=1):
    """Simplify line string by removing small lines."""
    if len(polygon) < 3:
        return polygon
    lines = []
    polygon = shapely.Polygon(polygon)
    points = np.array(polygon.buffer(0).exterior.coords)

    for ps in zip(points[:-1], points[1:]):
        ls = shapely.LineString(ps)
        if ls.length > 0:
            lines.append(ls)

    num_lines = len(lines)
    points = []
    i = 0
    while i < num_lines:
        if lines[i].length > tolerance:
            points.append(lines[i].coords[0])
        else:
            pline = lines[(i - 1) % num_lines]
            nline = lines[(i + 1) % num_lines]
            intersection = line_intersection(pline, nline)
            points.append(intersection.coords[0])
            i += 1
        i += 1

    lr = shapely.LinearRing(points).simplify(tolerance)
    lr = np.array(lr.coords)
    lr = simplify_small_angles(lr[:-1], tolerance)
    lr = simplify_small_angles(lr[:-1], tolerance)
    return lr


def aline_with_xy(line, alongX=True):
    x1, x2 = line.xy[0]
    y1, y2 = line.xy[1]
    if alongX:
        p = (x1, y2)
    else:
        p = (x2, y1)
    return p


def start_with_xy_line(points: np.ndarray):
    i = 0
    while i < len(points):
        cpoint = points[i]
        npoint = points[(i + 1) % len(points)]
        ppoint = points[i - 1]
        diff_prev = abs(ppoint - cpoint)
        diff_next = abs(npoint - cpoint)
        if any(diff_next == 0) and any(diff_prev == 0):
            break
        i += 1
    return np.vstack([points[i:], points[:i]])


def remove_straight_lines(points: np.ndarray):
    """Remove straight lines from the polygon."""
    removed = None
    if len(points) < 3:
        return None

    if all(points[0] == points[-1]):
        points = points[:-1]

    n = len(points)
    for i, p in enumerate(points):
        p_point = points[(i - 1) % n]
        n_point = points[(i + 1) % n]
        line = shapely.LineString([p_point, n_point])
        p_dist = projection_distance(p, line)
        if p_dist is None or p_dist < 0.001:
            removed = i
            break
    if removed is not None:
        return remove_straight_lines(np.delete(points, removed, axis=0))
    return np.array(shapely.Polygon(points).exterior.coords)


def simplify_small_angles(points: np.ndarray, tolerance):
    """Simplify line string by removing small lines."""
    if len(points) < 3:
        return None

    points = remove_straight_lines(points)
    if points is None or len(points) < 3:
        return None

    points = start_with_xy_line(points)
    try:
        lr = shapely.LinearRing(points).simplify(tolerance)
    except:
        return points
    points = np.array(lr.coords)

    ps = []
    i = 0
    while i < len(points):
        cpoint = points[i]
        npoint = points[(i + 1) % len(points)]
        cline = shapely.LineString([cpoint, npoint])

        ps.append(cpoint)
        diff = abs(npoint - cpoint)
        if not any(diff == 0):
            p = aline_with_xy(cline, diff[0] < diff[1])
            ps.append(p)
            points[(i + 1) % len(points)] = p
        i += 1

    r = shapely.LinearRing(ps)
    r = r.simplify(tolerance)
    arr = np.array(r.coords, dtype=np.int32)
    return arr


def get_overlap(p1, p2, overlap=0.0):
    intersection = p1.intersection(p2)
    if not intersection.is_empty:
        if intersection.geom_type == "Point":
            return None
        if (
            intersection.geom_type == "LineString"
            or intersection.geom_type == "MultiLineString"
        ):
            if intersection.length > overlap:
                return intersection
            else:
                raise RuntimeError(f"Intersecting length: {p1}, {p2}, {intersection}")
        else:
            raise RuntimeError(f"Intersecting shape: {p1}, {p2}, {intersection}")
    return None


def offset_polygon(coords, line_to_offset, offset):
    if isinstance(coords, shapely.Polygon):
        coords = np.array(coords.exterior.coords)[:, :2]
    line = shapely.LineString(
        (coords[line_to_offset], coords[(line_to_offset + 1) % len(coords)])
    )
    previous_line = shapely.LineString(
        (coords[line_to_offset - 1], coords[line_to_offset])
    )
    next_line = shapely.LineString(
        (
            coords[(line_to_offset + 1) % len(coords)],
            coords[(line_to_offset + 2) % len(coords)],
        )
    )

    offset_line = line.parallel_offset(offset)
    p1 = line_intersection(previous_line, offset_line)
    p2 = line_intersection(next_line, offset_line)

    coords[line_to_offset] = p1.coords[0]
    coords[(line_to_offset + 1) % len(coords)] = p2.coords[0]
    if line_to_offset == 0:
        coords[-1] = p1.coords[0]
    if line_to_offset == len(coords) - 2:
        coords[0] = p2.coords[0]
    return coords


def is_collinear(p1, p2, p3, tol=1e-7):
    """
    Check if three points are collinear.
    :param p1, p2, p3: Points as (x, y) tuples.
    :param tol: Tolerance for floating-point comparisons.
    :return: True if the points are collinear, otherwise False.
    """
    return (
        abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < tol
    )


def remove_collinear_points(polygon: shapely.Polygon) -> shapely.Polygon:
    """
    Remove collinear points from the polygon.
    :param polygon: A Shapely Polygon.
    :return: A new Polygon with collinear points removed.
    """
    coords = list(polygon.exterior.coords[:-1])  # Exclude the repeated last point
    if len(coords) <= 3:
        return polygon  # No collinear points to remove

    new_coords = []
    n = len(coords)

    for i in range(n):
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[(i + 1) % n]

        if not is_collinear(p1, p2, p3):
            new_coords.append(p2)

    # Ensure the polygon is closed
    if new_coords[0] != new_coords[-1]:
        new_coords.append(new_coords[0])

    return shapely.Polygon(new_coords)
