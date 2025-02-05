from enum import Enum
import ezdxf
import math
import numpy as np
import shapely


class Boundary(Enum):
    """Boundary layers in DXF file."""

    INTERNAL = dict(layer="BO-INTERNAL", color="#e377c2")
    SERVICE = dict(layer="BO-SERVICE", color="#e377c2")
    EXTENDED_BOUNDARY = dict(layer="BO-EXTENDED_BOUNDARY", color="#aaffaa")
    NEGATIVE_SPACE = dict(layer="BO-NEGATIVE_SPACE", color="#ff7f0e")


class DoorWindow(Enum):
    """Door and window layers in DXF file."""

    DOOR = dict(layer="DW-DOOR", color="#65000F")
    OPENING = dict(layer="DW-OPENING", color="#65101F")
    WINDOW = dict(layer="DW-WINDOW", color="#1f77b4")
    ENTRANCE = dict(layer="DW-ENTRANCE", color="#ff7f0e")


class SpaceName(Enum):
    """Space layers in DXF file."""

    LIVING_DINING = dict(layer="SP-LIVING_DINING", color="#bcbd22")
    LIVING = dict(layer="SP-LIVING", color="#bcbd22")
    KITCHEN_DINING = dict(layer="SP-KITCHEN_DINING", color="#9467bd")
    KITCHEN = dict(layer="SP-KITCHEN", color="#9467bd")
    DINING = dict(layer="SP-DINING", color="#2ca02c")
    MASTER_BEDROOM = dict(layer="SP-MASTER_BEDROOM", color="#030D38")
    WALKIN_CLOSET = dict(layer="SP-WALKIN_CLOSET", color="#8c564b")
    BEDROOM = dict(layer="SP-BEDROOM", color="#ff7f0e")
    OFFICE = dict(layer="SP-OFFICE", color="#ff7f0e")
    TOILET = dict(layer="SP-TOILET", color="#17becf")
    CORRIDOR = dict(layer="SP-CORRIDOR", color="#aaaaaa")
    SERVICE = dict(layer="SP-SERVICE", color="#d62728")
    UTILITY = dict(layer="SP-UTILITY", color="#8c564b")
    GARDEN = dict(layer="SP-GARDEN", color="#9467bd")
    TERRACE = dict(layer="SP-TERRACE", color="#2ca02c")


class Errors(Enum):
    MISSING_PERIMETER = "No polyline in PERIMETER layer."
    MORE_THAN_ONE_PERIMETER_POLYLINES = "More than one polyline in PERIMETER layer."
    NO_SERVICE_WALLS = "No polyline in BO-SERVICE layer."
    NO_INTERNAL_WALLS = "No polyline in BO-INTERNAL layer."
    INVALID_SPACE = "Invalid space {0}."
    INVALID_DOOR_WINDOW = "Invalid door/window {0}."
    INVALID_LAYER = "Invalid layer {0}."
    NO_NORTH = "No NORTH found."
    OVERLAPPING_OBJECTS = "Overlapping objects {0} and {1}."
    OUTSIDE_PERIMETER = "Object {0} is outside the perimeter."
    NO_ACCESS_SPACE = "No access for {0}."


def is_overlap(p1, p2):
    intersection = p1.intersection(p2)
    if isinstance(intersection, shapely.Polygon) and intersection.area > 0:
        return True
    return False


def round_off(a_list, z_value=0.0):
    """Rounds off an array to the nearest 0.1 and append z-value."""
    arr = np.array(a_list)
    if arr.shape[-1] == 2:
        arr_z = np.ones((arr.shape[0], 1)) * z_value
        arr = np.hstack([arr, arr_z])
    arr = 50 * np.round(arr / 50)
    return arr


def append_z(a_list, z_value=0.0):
    """Append z-value to an array."""
    if isinstance(a_list, shapely.MultiPolygon):
        return append_z(a_list.geoms[0], z_value)

    if isinstance(a_list, shapely.Polygon):
        return append_z(a_list.exterior.coords, z_value)

    if isinstance(a_list, (shapely.LinearRing, shapely.LineString)):
        return append_z(a_list.coords, z_value)

    if isinstance(a_list, dict):
        for key in a_list:
            a_list[key] = append_z(a_list[key], z_value)
        return a_list

    arr = np.array(a_list)
    if arr.shape[-1] == 2:
        arr_z = np.ones((arr.shape[0], 1)) * z_value
        arr = np.hstack([arr, arr_z])
    return arr


def read_dxf_file(file):
    """Reads a DXF file and return the contents of model space."""
    doc = ezdxf.readfile(file)
    return doc.modelspace()


def direction_vector(line):
    if isinstance(line, shapely.LineString):
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[1]
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return (x2 - x1, y2 - y1)


# Function to check if two lines are perpendicular
def are_perpendicular(line1, line2):
    # Get direction vectors of the lines
    v1 = direction_vector(line1)
    v2 = direction_vector(line2)

    # Calculate the dot product of the vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Lines are perpendicular if the dot product is close to 0
    return math.isclose(dot_product, 0, abs_tol=1e-3)
