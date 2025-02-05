import math
import shapely


def round_value(value, round_to):
    """Rounds a value to a given decimal precision."""
    return round(value / round_to) * round_to


def round_coordinates(coords, round_to=0.1):
    """Rounds coordinates to a given decimal precision."""
    return [(round_value(x, round_to), round_value(y, round_to)) for x, y in coords]


class RPoint(shapely.Point):
    def __new__(cls, x, y=None, round_to=0.1):
        """Create a Point with rounded coordinates."""
        if y is not None:
            x, y = round_value(x, round_to), round_value(y, round_to)
        else:
            x, y = round_value(x[0], round_to), round_value(x[1], round_to)
        return super().__new__(cls, (x, y))


class RLineString(shapely.LineString):
    def __new__(cls, coords):
        """Create a LineString with rounded coordinates."""
        coords = [x.coords if isinstance(x, RPoint) else x for x in coords ]
        return super().__new__(cls, round_coordinates(coords,))


class RPolygon(shapely.Polygon):
    def __new__(cls, shell, holes=None):
        """Create a Polygon with rounded coordinates."""
        shell = round_coordinates(shell,)
        holes = [round_coordinates(h,) for h in holes] if holes else None
        return super().__new__(cls, shell, holes)


def point_on_line(point, line):
    p1 = RPoint(line.coords[0])
    p2 = RPoint(line.coords[1])
    return (
        shapely.distance(p1, point) < 1e-6
        or shapely.distance(p2, point) < 1e-6
        or shapely.contains(line, point)
    )


def line_intersection(p1, p2, p3, p4):
    """
    Computes the intersection point of two lines given by (p1, p2) and (p3, p4).
    Lines are extended if necessary.

    Args:
        p1, p2: (x, y) tuple representing the first line.
        p3, p4: (x, y) tuple representing the second line.

    Returns:
        (x, y): Intersection point if lines are not parallel.
        None: If the lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Compute determinant
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(D) < 1e-10:  # Lines are parallel
        return None

    # Compute intersection
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D

    return RPoint(Px, Py)


def project_point_on_line(point, line):
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    x3, y3 = point
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = (dx * (x3 - x1) + dy * (y3 - y1)) / det
    return RPoint(x1 + a * dx, y1 + a * dy)


def project_line_on_line(line: RLineString, project_on: RLineString):
    first = project_point_on_line(line.coords[0], project_on)
    second = project_point_on_line(line.coords[1], project_on)
    return shapely.geometry.LineString([first, second])


def get_overlap(
    line1: RLineString,
    line2: RLineString,
    max_distance_threshold,
    min_distance_threshold,
    angle_threshold=5,
):
    a1 = get_angle(line1)
    a2 = get_angle(line2)
    if abs(a1 - a2) > angle_threshold:
        return None
    first = project_point_on_line(line2.coords[0], line1)
    second = project_point_on_line(line2.coords[1], line1)

    if first.distance(RPoint(line2.coords[0])) > max_distance_threshold:
        return None

    if first.distance(RPoint(line2.coords[0])) < min_distance_threshold:
        return None

    projected_line = shapely.geometry.LineString([first, second])
    intersection = projected_line.intersection(line1)
    if intersection.is_empty:
        return None
    if intersection.geom_type == "Point":
        return None
    if intersection.geom_type == "LineString":
        return intersection


def get_angle(line: RLineString):
    slope = (line.coords[1][1] - line.coords[0][1]), (
        line.coords[1][0] - line.coords[0][0]
    )
    angle = (math.atan2(*slope) + 2 * math.pi) % math.pi
    return math.degrees(angle)


def shorter_longer_lines(rectangle: RPolygon):
    lines = []
    for i in range(len(list(rectangle.exterior.coords)) - 1):
        lines.append(
            RLineString([rectangle.exterior.coords[i], rectangle.exterior.coords[i + 1]])
        )
    return sorted(lines, key=lambda x: x.length)


def mitre_join(rect1: RPolygon, rect2: RPolygon) -> RPolygon:
    intersection = rect1.intersection(rect2)
    assert (
        intersection.geom_type == "Point"
    ), f"Rectangles do not intersect or intersect at multiple points. {intersection}"

    lines_p1 = shorter_longer_lines(rect1)
    lines_p2 = shorter_longer_lines(rect2)
    middle_p1 = RLineString(
        [
            lines_p1[0].interpolate(lines_p1[0].length / 2),
            lines_p1[1].interpolate(lines_p1[1].length / 2),
        ]
    )

    middle_p2 = RLineString(
        [
            lines_p2[0].interpolate(lines_p2[0].length / 2),
            lines_p2[1].interpolate(lines_p2[1].length / 2),
        ]
    )
    middle_intersection = line_intersection(*middle_p1.coords, *middle_p2.coords)
    line = RLineString([intersection, middle_intersection])
    extended_line = shapely.affinity.scale(line, xfact=2, yfact=2, origin=intersection)

    l1 = next(l for l in lines_p1[:2] if point_on_line(intersection, l))
    l2 = next(l for l in lines_p2[:2] if point_on_line(intersection, l))

    p2 = next(p for p in l1.coords if RPoint(p).distance(intersection) > 1e-5)
    p4 = next(p for p in l2.coords if RPoint(p).distance(intersection) > 1e-5)

    return rect1.union(
        RPolygon([intersection, p2, extended_line.coords[1]])
    ), rect2.union(RPolygon([intersection, extended_line.coords[1], p4]))


def get_center_line_thickness(rect1: RPolygon, rect2: RPolygon) -> RPolygon:
    intersection = rect1.intersection(rect2)
    assert (
        intersection.geom_type == "Point"
    ), f"Rectangles do not intersect or intersect at multiple points. {intersection}"

    lines_p1 = shorter_longer_lines(rect1)
    lines_p2 = shorter_longer_lines(rect2)
    middle_p1 = RLineString(
        [
            lines_p1[0].interpolate(lines_p1[0].length / 2).coords[0],
            lines_p1[1].interpolate(lines_p1[1].length / 2).coords[0],
        ]
    )

    middle_p2 = RLineString(
        [
            lines_p2[0].interpolate(lines_p2[0].length / 2).coords[0],
            lines_p2[1].interpolate(lines_p2[1].length / 2).coords[0],
        ]
    )
    middle_intersection = line_intersection(*middle_p1.coords, *middle_p2.coords)
    def get_farthest_point(line, point):
        return max(line.coords, key=lambda x: shapely.distance(RPoint(x), point))
    
    p1 = tuple(get_farthest_point(middle_p1, middle_intersection))
    p2 = tuple(get_farthest_point(middle_p2, middle_intersection))
    
    l1 = RLineString(
        [p1, tuple(middle_intersection.coords[0])]
    )
    l2 = RLineString(
        [p2, tuple(middle_intersection.coords[0])]
    )
    return (l1, lines_p1[0].length), (l2, lines_p2[0].length)


def extend_lines_for_polygonize(wall_lines):
    for i1, (l1, t1) in enumerate(wall_lines):
        for i2, (l2, t2) in enumerate(wall_lines):
            if i1 == i2:
                continue
            p1 = l1.buffer(t1)
            p2 = l2.buffer(t2)
            if (p1.intersects(p2) or p1.touches(p2)) and (not l1.intersects(l2)):
                point = line_intersection(*l1.coords, *l2.coords)
                if point is None:
                    continue
                l1 = RLineString(list(l1.coords) + list(point.coords)).simplify(0.01)
                l2 = RLineString(list(l2.coords) + list(point.coords)).simplify(0.01)
                
                l1 = RLineString([(l1.bounds[0], l1.bounds[1]), (l1.bounds[2], l1.bounds[3])])
                l2 = RLineString([(l2.bounds[0], l2.bounds[1]), (l2.bounds[2], l2.bounds[3])])
                wall_lines[i1] = [l1, t1]
                wall_lines[i2] = [l2, t2]
