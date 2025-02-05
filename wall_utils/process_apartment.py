import numpy as np
from shapely import MultiLineString, unary_union, union_all
from shapely.ops import polygonize
import shapely
from tripy import earclip

from .wall_utils import (
    collect_all_lines,
    create_wall_polygons,
    get_adjoining_polygons,
    get_center_lines,
    get_horizontal_vertical_walls,
    get_nearest_parrallel_lines,
)

from .utility_functions import extend_lines_for_polygonize, get_angle, project_point_on_line

from .utility_functions import (RPoint, RLineString, RPolygon)


def process_apartment(example: dict, min_area=2, scaling=0.001):
    perimeter = RPolygon(scaling * np.array(example["perimeter"])[:, :2]).buffer(0)
    if perimeter.area < min_area:
        raise ValueError("Perimeter polygon is too small.")

    complete_walls = perimeter.buffer(0)
    spaces = {}
    for s in example["spaces"]:
        polygon = RPolygon(scaling * np.array(example["spaces"][s])[:, :2]).buffer(0)
        if polygon.geom_type != "Polygon" or not polygon.is_valid:
            print(f"Space {s} is not a valid polygon.")
            continue

        if polygon.area < min_area:
            print(f"Space {s} is too small.")
            continue
        
        spaces[s] = polygon
        complete_walls = complete_walls.difference(
            spaces[s]
        ).buffer(0)

    all_lines = collect_all_lines(perimeter, spaces)
    wall_lines = get_nearest_parrallel_lines(all_lines, spaces, perimeter, overlap_threshold=0.1, max_distance_threshold=0.5, min_distance_threshold=0.05)
    wall_polygons = list(create_wall_polygons(wall_lines))
    
    horizontal_walls, vertical_walls = get_horizontal_vertical_walls(wall_polygons)
    wall_fillet = complete_walls
    for p in wall_polygons:
        wall_fillet = wall_fillet.difference(p)
    
    adjoining_polygons = list(get_adjoining_polygons(wall_polygons))
    cl_wall = list(get_center_lines(adjoining_polygons))
    
    cl_spaces = get_centerline_spaces(cl_wall, spaces)
    cl_dws = dict()
    for d in example["door_windows"]:
        dw = RPolygon(scaling * np.array(example["door_windows"][d])[:, :2]).buffer(0)
        cl_dw = place_dw_on_centerline(cl_wall, dw)
        if cl_dw is not None:
            cl_dws[d] = cl_dw
            
    ibs = []
    for internal in example["boundaries"]:
        if internal.startswith("INTERNAL"):
            internal = RPolygon(scaling * np.array(example["boundaries"][internal])[:, :2]).buffer(0)
            ibs.append(internal)

    # wall_fillet = list(get_fillet_polygons(adjoining_polygons))
    return dict(
        walls=list(merge_with_horizontal_walls(horizontal_walls, wall_fillet) + vertical_walls,),
        wall_polygons=list(wall_polygons),
        spaces = spaces,
        perimeter = perimeter,
        centerlines = dict(
            spaces=cl_spaces,
            door_windows=cl_dws,
            walls=cl_wall,
            triangles=dict(list(triangulate_spaces(cl_spaces))),
            internal=list(internal_centerline(ibs, cl_wall))
        )
    )


def merge_with_horizontal_walls(walls, fillet):
    h_walls = union_all([unary_union(walls), unary_union(fillet)])
    if h_walls.geom_type == "MultiPolygon":
        return list(h_walls.geoms)
    elif h_walls.geom_type == "Polygon":
        return [h_walls]


def align_terrace(terrace, wall_lines):
    terrace = terrace.simplify(0.1)
    coords = list(terrace.exterior.coords)[: -1]
    for i1 in range(len(coords)):
        i2 = (i1 + 1) % len(coords)
        l1 = RLineString([coords[i1], coords[i2]])
        
        filtered_wall_lines = [x for x in wall_lines if abs(get_angle(x[0]) - get_angle(l1)) < 5]
        distances = [(x, project_point_on_line(coords[i1], x[0]).distance(RPoint(coords[i1]))) for x in filtered_wall_lines]
        filtered_wall_lines = filter(lambda c: 2 * c[0][1] > c[1], distances)
        sorted_wall_lines = sorted(filtered_wall_lines, key=lambda x: x[1])
        if len(sorted_wall_lines) == 0:
            continue
        l2 = sorted_wall_lines[0][0][0]
        pr1 = project_point_on_line(coords[i1], l2)
        
        coords[i1] = pr1.coords[0]
        coords[i2] = project_point_on_line(coords[i2], l2).coords[0]
    return RPolygon(coords).simplify(0.1)


def get_centerline_spaces(wall_center_lines, spaces):
    wall_center_lines = list(wall_center_lines)
    extend_lines_for_polygonize(wall_center_lines,)
    
    lines = [x[0] for x in wall_center_lines]
    lines = MultiLineString(lines)
    lines = unary_union(lines)
    polygons = list(polygonize(lines))

    cl_spaces = dict()
    for s in spaces:
        s_point = spaces[s].representative_point()
        try:
            cl_spaces[s] = next(p for p in polygons if p.contains(s_point))
        except StopIteration:
            cl_spaces[s] = align_terrace(spaces[s], wall_center_lines)
    return cl_spaces


def orient_dw(dw: RPolygon):
    d1 = RPoint(dw.exterior.coords[0]).distance(RPoint(dw.exterior.coords[1]))
    if len(dw.exterior.coords) < 3:
        return d1, RLineString([dw.exterior.coords[0], dw.exterior.coords[1]])
    d2 = RPoint(dw.exterior.coords[1]).distance(RPoint(dw.exterior.coords[2]))
    if d1 < d2:
        dw = RPolygon([dw.exterior.coords[1], dw.exterior.coords[2], dw.exterior.coords[3], dw.exterior.coords[4]])
        return d2, RLineString([dw.exterior.coords[1], dw.exterior.coords[2]])
    return d1, RLineString([dw.exterior.coords[0], dw.exterior.coords[1]])

def place_dw_on_centerline(wall_centerlines, dw):
    try:
        l, dw_line = orient_dw(dw)
    except IndexError:
        return None
    filtered_centerlines = [x for x in wall_centerlines if abs(get_angle(x[0]) - get_angle(dw_line)) < 5]
    dist_centerlines = [(x, project_point_on_line(dw.exterior.coords[0], x[0]).distance(RPoint(dw.exterior.coords[0]))) for x in filtered_centerlines]
    # filtered_centerlines = filter(lambda c: 2 * c[0][1] > c[1], dist_centerlines)
    sorted_centerlines = sorted(dist_centerlines, key=lambda x: x[1])
    if len(sorted_centerlines) == 0:
        return None
    line = sorted_centerlines[0][0][0]
    p1 = project_point_on_line(dw_line.coords[0], line)
    p2 = project_point_on_line(dw_line.coords[1], line)
    return RLineString([p1.coords[0], p2.coords[0]])


def triangulate_spaces(spaces):
    for s in spaces:
        space = spaces[s]
        yield(s, list(triangulate(np.array(space.exterior.coords))))


def triangulate(points):
    """
    Triangulates a polygon.
    """
    if all(points[0] == points[-1]):
        points = points[:-1]
    if points.shape[1] == 3:
        points = points[:, :2]
    points = RPolygon(points)
    points = points.buffer(0)
    triangles = earclip(points.exterior.coords[:-1])
    triangles = [RPolygon(t) for t in triangles]
    return triangles


def internal_centerline(ibs, wall_centerlines):
    for l, _ in wall_centerlines:
        for x in ibs:
            intersection = l.intersection(x)
            if intersection and intersection.geom_type == "LineString" and intersection.length > 0.5:
                yield l
                break
