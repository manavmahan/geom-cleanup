import numpy as np
from .utility_functions import (
    get_angle, get_center_line_thickness, get_overlap, mitre_join, shorter_longer_lines,
    RLineString, RPoint, RPolygon
)


def collect_all_lines_with_angle(lines: list[RLineString], threshold=5):
    lines_with_angle = {}
    for line in lines:
        if line.length < 1e-5:
            continue
        angle = get_angle(line)
        for a in lines_with_angle:
            if abs(a - angle) < threshold:
                angle = a
                break
        if angle not in lines_with_angle:
            lines_with_angle[angle] = []
        lines_with_angle[angle].append(line)
    return lines_with_angle


def collect_all_lines(perimeter, spaces):
    lines = []
    for i in range(len(perimeter) - 1):
        lines.append(RLineString([perimeter[i], perimeter[i + 1]]))
    for s in spaces:
        space = spaces[s]
        for i in range(len(space) - 1):
            lines.append(RLineString([space[i], space[i + 1]]))
    return collect_all_lines_with_angle(lines)


def get_nearest_parrallel_lines(
    all_lines, spaces, perimeter, overlap_threshold, max_distance_threshold, min_distance_threshold,
):
    parrallel_lines = []
    temp_spaces = [RPolygon(spaces[s]) for s in spaces]
    for angle in all_lines:
        lines = all_lines[angle]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x = get_overlap(
                    lines[i], lines[j], max_distance_threshold=max_distance_threshold, min_distance_threshold=min_distance_threshold
                )
                if x is not None and x.length > overlap_threshold:
                    y = get_overlap(
                        lines[j], lines[i], max_distance_threshold=max_distance_threshold, min_distance_threshold=min_distance_threshold
                    )
                    if y is not None:
                        mid_point = RPoint(np.average(np.vstack([x.coords, y.coords]), axis=0))
                        if (not any(
                            [s.contains(mid_point) for s in temp_spaces]
                        )) and (perimeter.contains(mid_point)):
                            parrallel_lines.append((lines[i], lines[j], x, y))
    return parrallel_lines


def create_wall_polygons(wall_lines):
    for _, _, l1, l2 in wall_lines:
        x = list(l1.xy)[0] + list(l2.xy)[0][::-1]
        y = list(l1.xy)[1] + list(l2.xy)[1][::-1]
        p = RPolygon([l1.coords[0], l1.coords[1], l2.coords[1], l2.coords[0]])
        if not p.is_valid:
            p = RPolygon(
                [l1.coords[0], l1.coords[1], l2.coords[0], l2.coords[1]]
            )
        yield p


def get_adjoining_polygons(wall_polygons):
    for i, p1 in enumerate(wall_polygons):
        for j in range(len(wall_polygons)):
            if i == j: continue
            intersection = p1.intersection(wall_polygons[j])
            if intersection and intersection.geom_type == "Point":
                yield (p1, wall_polygons[j])


def get_fillet_polygons(adjoining_polygons: list[(RPolygon, RPolygon)]):
    for p1, p2 in adjoining_polygons:
        for p in mitre_join(p1, p2):
            yield p


def get_horizontal_vertical_walls(walls):
    horizontal_walls = []
    vertical_walls = []
    for wall in walls:
        lines = shorter_longer_lines(wall)
        if get_angle(lines[2]) < 5:
            horizontal_walls.append(wall)
        else:
            vertical_walls.append(wall)
    return horizontal_walls, vertical_walls


def get_center_lines(adjoining_wall_polygons):
    all_center_lines = []
    for p1, p2 in adjoining_wall_polygons:
        value = get_center_line_thickness(p1, p2)
        all_center_lines.extend(value)
    
    all_center_lines_with_angle = dict()
    for line, thickness in all_center_lines:
        angle = np.round(get_angle(line))
        thickness = np.round(thickness, 2)
        if (angle, thickness) not in all_center_lines_with_angle:
            all_center_lines_with_angle[(angle, thickness)] = []
        all_center_lines_with_angle[(angle, thickness)].append([line, thickness])

    for angle, thickness in all_center_lines_with_angle:
        lines_to_merge = [x[0] for x in all_center_lines_with_angle[(angle, thickness)]]
        while merge_lines(lines_to_merge):
            pass
        for line in lines_to_merge:
            yield line, thickness


def merge_lines(lines):
    if len(lines) < 2:
        return False
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = lines[i].intersection(lines[j])
            if intersection or intersection.geom_type == "Point":
                lines[i] = RLineString(np.array(lines[i].union(lines[j]).bounds).reshape(2, 2))
                del lines[j]
                return True
    return False


def get_wall_triangles(example, min_area=2):
    perimeter_polygon = RPolygon(np.array(example["perimeter"])[:, :2]).buffer(0)
    if perimeter_polygon.area < min_area:
        raise ValueError("Perimeter polygon is too small.")

    perimeter = np.array(
        perimeter_polygon.exterior.coords
    )
    complete_walls = RPolygon(perimeter).buffer(0)
    spaces = {}
    spaces_polygons = {}
    for s in example["spaces"]:
        polygon = RPolygon(np.array(example["spaces"][s])[:, :2]).buffer(0)
        if polygon.geom_type != "Polygon" or not polygon.is_valid:
            print(f"Space {s} is not a valid polygon.")
            continue

        if polygon.area < min_area:
            print(f"Space {s} is too small.")
            continue
        
        spaces_polygons[s] = polygon
        spaces[s] = np.array(polygon.exterior.coords)
        complete_walls = complete_walls.difference(
            RPolygon(spaces[s])
        ).buffer(0)
    all_lines = collect_all_lines(perimeter, spaces)
    wall_lines = get_nearest_parrallel_lines(all_lines, spaces, perimeter_polygon, overlap_threshold=0.1, max_distance_threshold=1.0, min_distance_threshold=0.05)
    return list(create_wall_triangles(wall_lines))


def create_wall_triangles(wall_lines):
    for line in wall_lines:
        l1, l2, _, _ = line
        p1 = RPolygon([l1.coords[0], l1.coords[1], l2.coords[0], l2.coords[1]])
        if not p1.is_valid:
            p1 = RPolygon(
                [l1.coords[0], l1.coords[1], l2.coords[1], l2.coords[0]]
            )
        yield RPolygon([p1.exterior.coords[0], p1.exterior.coords[1], p1.exterior.coords[2]])
        yield RPolygon([p1.exterior.coords[0], p1.exterior.coords[2], p1.exterior.coords[3]])
