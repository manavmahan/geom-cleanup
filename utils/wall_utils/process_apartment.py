import numpy as np
from shapely import unary_union, union_all

from .wall_utils import (
    collect_all_lines,
    create_wall_polygons,
    get_adjoining_polygons,
    get_center_lines,
    get_horizontal_vertical_walls,
    get_nearest_parrallel_lines,
)

from .utility_functions import (RPolygon)


def process_apartment(example: dict, min_area=2):
    perimeter_polygon = RPolygon(np.array(example["perimeter"])[:, :2]).buffer(0)
    if perimeter_polygon.area < min_area:
        raise ValueError("Perimeter polygon is too small.")

    perimeter = np.array(
        RPolygon(np.array(example["perimeter"])[:, :2]).buffer(0).exterior.coords
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
        spaces[s] = np.array(
            RPolygon(np.array(example["spaces"][s])[:, :2])
            .buffer(0)
            .exterior.coords
        )
        complete_walls = complete_walls.difference(
            RPolygon(spaces[s])
        ).buffer(0)

    all_lines = collect_all_lines(perimeter, spaces)
    wall_lines = get_nearest_parrallel_lines(all_lines, spaces, perimeter_polygon, overlap_threshold=0.1, max_distance_threshold=0.5, min_distance_threshold=0.05)
    wall_polygons = list(create_wall_polygons(wall_lines))
    
    horizontal_walls, vertical_walls = get_horizontal_vertical_walls(wall_polygons)
    wall_fillet = complete_walls
    for p in wall_polygons:
        wall_fillet = wall_fillet.difference(p)
    
    adjoining_polygons = list(get_adjoining_polygons(wall_polygons))
    wall_center_lines = get_center_lines(adjoining_polygons)

    # wall_fillet = list(get_fillet_polygons(adjoining_polygons))
    return dict(
        walls=list(merge_with_horizontal_walls(horizontal_walls, wall_fillet) + vertical_walls,),
        wall_lines=list(wall_center_lines),
        wall_polygons=list(wall_polygons),
        spaces = spaces_polygons,
        perimeter = perimeter_polygon,
    )

def merge_with_horizontal_walls(walls, fillet):
    h_walls = union_all([unary_union(walls), unary_union(fillet)])
    if h_walls.geom_type == "MultiPolygon":
        return list(h_walls.geoms)
    elif h_walls.geom_type == "Polygon":
        return [h_walls]
