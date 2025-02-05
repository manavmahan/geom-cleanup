import pickle
import numpy as np
import shapely


from .helper import (
    Boundary,
    DoorWindow,
    SpaceName,
    Errors,
    append_z,
    is_overlap,
    round_off,
    read_dxf_file,
    are_perpendicular,
)

from . import utility_functions as u


class FloorplanData:
    def __init__(
        self,
        id,
    ) -> None:
        self.id = id
        self.north = None
        self.perimeter = None
        self.boundaries = {}
        self.spaces = {}
        self.door_windows = {}
        self.door_connections = {}
        self.errors = []

    def get_origin(self):
        return np.min(self.perimeter, axis=0)

    def get_angle(self):
        l1 = None
        l2 = None
        for i in range(len(self.perimeter)):
            next_line = [
                self.perimeter[i],
                self.perimeter[(i + 1) % len(self.perimeter)],
            ]
            prev_line = [self.perimeter[i], self.perimeter[i - 1]]
            if are_perpendicular(next_line, prev_line):
                l1, l2 = next_line, prev_line
                break

        if l1 is None:
            for s in self.spaces:
                for i in range(len(self.spaces[s])):
                    next_line = [
                        self.spaces[s][i],
                        self.spaces[s][(i + 1) % len(self.spaces[s])],
                    ]
                    prev_line = [self.spaces[s][i], self.spaces[s][i - 1]]
                    if are_perpendicular(next_line, prev_line):
                        l1, l2 = next_line, prev_line
                        break

        if l1 is None:
            return 0.0
        return np.arctan2(l1[1][1] - l1[0][1], l1[1][0] - l1[0][0])

    def reposition(self, origin):
        if all(origin == 0):
            return
        self.perimeter -= origin
        for k in self.boundaries:
            self.boundaries[k] -= origin
        for k in self.spaces:
            self.spaces[k] -= origin
        for k in self.door_windows:
            self.door_windows[k] -= origin

    def rotate(self, angle):
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        self.perimeter = np.dot(self.perimeter, rotation_matrix)
        for k in self.boundaries:
            self.boundaries[k] = np.dot(self.boundaries[k], rotation_matrix)
        for k in self.spaces:
            self.spaces[k] = np.dot(self.spaces[k], rotation_matrix)
        for k in self.door_windows:
            self.door_windows[k] = np.dot(self.door_windows[k], rotation_matrix)

    def round_off(self, multiplier=1):
        self.perimeter = round_off(self.perimeter * multiplier)
        for k in self.boundaries:
            self.boundaries[k] = round_off(self.boundaries[k] * multiplier)
        for k in self.spaces:
            self.spaces[k] = round_off(self.spaces[k] * multiplier)
        for k in self.door_windows:
            self.door_windows[k] = round_off(self.door_windows[k] * multiplier)

    def save(self, save_to=None):
        save_dict = dict(
            id=self.id,
            north=self.north,
            perimeter=self.perimeter,
            boundaries=self.boundaries,
            spaces=self.spaces,
            door_windows=self.door_windows,
            door_connections=self.door_connections,
            errors=self.errors,
        )
        if save_to is None:
            return save_dict
        with open(save_to, "wb") as f:
            pickle.dump(save_dict, f)

    def add_boundary(self, name, boundary):
        enum_name = Boundary[name].name
        name = f"{enum_name}-{len([x for x in self.boundaries if enum_name in x]) + 1}"
        self.boundaries[name] = boundary

    def add_door_window(self, name, door_window):
        enum_name = DoorWindow[name].name
        name = (
            f"{enum_name}-{len([x for x in self.door_windows if enum_name in x]) + 1}"
        )
        if len(door_window) != 4:
            self.errors.append(
                Errors.INVALID_DOOR_WINDOW.value.format(
                    f"{name} - {len(door_window)} points."
                )
            )
        else:
            if (
                shapely.LineString(door_window[0:2]).length
                < shapely.LineString(door_window[1:3]).length
            ):
                door_window = np.roll(door_window, 1, axis=0)
        self.door_windows[name] = door_window

    def add_space(self, name, space):
        enum_name = SpaceName[name].name
        name = f"{enum_name}-{len([x for x in self.spaces if enum_name == x.split('-')[0]]) + 1}"
        self.spaces[name] = space
        if len(space) < 4:
            self.errors.append(Errors.INVALID_SPACE.value.format(name))

    def add_line(self, name, lines):
        if name.startswith("SP-"):
            self.add_space(name[3:], lines)
        elif name.startswith("BO-"):
            self.add_boundary(name[3:], lines)
        elif name.startswith("DW-"):
            self.add_door_window(name[3:], lines)
        elif name == "PERIMETER":
            if self.perimeter is not None:
                self.perimeter = np.vstack([self.perimeter, lines])
                self.errors.append(Errors.MORE_THAN_ONE_PERIMETER_POLYLINES.value)
            else:
                self.perimeter = lines
        else:
            self.errors.append(Errors.INVALID_LAYER.format(name))

    @property
    def doors(self):
        return dict(
            (x, self.door_windows[x])
            for x in self.door_windows
            if x.startswith(DoorWindow.DOOR.name)
            or x.startswith(DoorWindow.ENTRANCE.name)
            or x.startswith(DoorWindow.OPENING.name)
        )

    @property
    def windows(self):
        return dict(
            (x, self.door_windows[x])
            for x in self.door_windows
            if x.startswith(DoorWindow.WINDOW.name)
        )

    @property
    def service_walls(self):
        return dict(
            (x, self.boundaries[x])
            for x in self.boundaries
            if x.startswith(Boundary.SERVICE.name)
        )

    @property
    def internal_walls(self):
        return dict(
            (x, self.boundaries[x])
            for x in self.boundaries
            if x.startswith(Boundary.INTERNAL.name)
        )

    def init_door_connections(self):
        self.door_connections = {}
        for d_name, door in self.doors.items():
            if d_name not in self.door_connections:
                self.door_connections[d_name] = [None, None]
            l1, l2 = shapely.LineString(door[0:2]), shapely.LineString(door[2:4])
            found = [False, False]
            for s_name, s in self.spaces.items():
                for ps in zip(s, np.roll(s, 1, axis=0)):
                    l3 = shapely.LineString(ps)
                    if l1.within(l3):
                        self.door_connections[d_name][0] = s_name
                        found[0] = True
                    elif l2.within(l3):
                        self.door_connections[d_name][1] = s_name
                        found[1] = True

            for b_name, b in self.internal_walls.items():
                for ps in zip(b, np.roll(b, 1, axis=0)):
                    l3 = shapely.LineString(b)
                    if l1.intersects(l3) or l2.intersects(l3):
                        self.errors.append(
                            Errors.INVALID_DOOR_WINDOW.value.format(
                                f"{d_name} - {b_name}"
                            )
                        )

            for b_name, b in self.service_walls.items():
                for ps in zip(b, np.roll(b, 1, axis=0)):
                    l3 = shapely.LineString(ps)
                    if not found[0] and l1.within(l3):
                        self.door_connections[d_name][0] = b_name
                    elif not found[1] and l2.within(l3):
                        self.door_connections[d_name][1] = b_name

        invalid_doors = []
        for d_name, door in self.door_connections.items():
            if door[0] is None or door[1] is None:
                self.errors.append(
                    Errors.INVALID_DOOR_WINDOW.value.format(f"{d_name} - {door}")
                )
                invalid_doors.append(d_name)
        for d_name in invalid_doors:
            del self.door_connections[d_name]

    def merge_connected_spaces(self, connected_spaces):
        i, j = None, None
        for i1, s1 in enumerate(connected_spaces):
            for i2, s2 in enumerate(connected_spaces):
                if i1 >= i2:
                    continue
                for x in s1:
                    for y in s2:
                        if x == y:
                            i, j = i1, i2
                            break
        if i is None:
            return connected_spaces
        connected_spaces[i].extend(connected_spaces[j])
        del connected_spaces[j]
        for i in range(len(connected_spaces)):
            connected_spaces[i] = list(set(connected_spaces[i]))
        return self.merge_connected_spaces(connected_spaces)

    def get_connected_spaces(self):
        connected_spaces = []
        for i1, (s1_name, s1) in enumerate(self.spaces.items()):
            for i2, (s2_name, s2) in enumerate(self.spaces.items()):
                if i1 >= i2:
                    continue
                p1 = shapely.Polygon(s1)
                p2 = shapely.Polygon(s2)
                intersection = p1.intersection(p2)
                if (
                    isinstance(
                        intersection, (shapely.LineString, shapely.MultiLineString)
                    )
                    and intersection.length > 0.6
                ):
                    connected_spaces.append([s1_name, s2_name])
        return connected_spaces

    def check_space_access(self):
        connected_spaces = self.get_connected_spaces()
        connected_spaces.extend(self.door_connections.values())
        merged_spaces = self.merge_connected_spaces(connected_spaces)
        for i in range(len(merged_spaces)):
            if not any(x.startswith("SERVICE-") for x in merged_spaces[i]):
                merged_spaces[i] = []

        for s_name, s in self.spaces.items():
            if not any(s_name in x for x in merged_spaces):
                self.errors.append(Errors.NO_ACCESS_SPACE.value.format(s_name))

    def check_windows(self):
        invalid_boundaries = self.internal_walls
        if invalid_boundaries is not None:
            invalid_boundaries.update(self.service_walls)
        else:
            invalid_boundaries = self.service_walls
        if invalid_boundaries is None:
            return
        for w_name, window in self.windows.items():
            l1 = shapely.LineString(window[0:2])
            l2 = shapely.LineString(window[2:4])
            for b_name, b in invalid_boundaries.items():
                for ps in zip(b[:-1], b[1:]):
                    l3 = shapely.LineString(ps)
                    i1 = l1.intersection(l3)
                    i2 = l2.intersection(l3)
                    if (i1 and not isinstance(i1, shapely.Point)) or (
                        i2 and not isinstance(i2, shapely.Point)
                    ):
                        self.errors.append(
                            Errors.INVALID_DOOR_WINDOW.value.format(
                                f"{w_name}-{b_name}"
                            )
                        )

            for s_name, s in self.spaces.items():
                p = shapely.Polygon(s)
                if l1.within(p) and l2.within(p):
                    self.errors.append(
                        Errors.INVALID_DOOR_WINDOW.value.format(f"{w_name}-{s_name}")
                    )

    def check_overlap_inside(self):
        all_polygons = {}
        all_polygons.update({x: shapely.Polygon(self.spaces[x]) for x in self.spaces})
        # all_polygons.update({
        #     x: shapely.Polygon(self.door_windows[x]) for x in self.door_windows
        # })
        # all_polygons.update(
        #     {x: shapely.Polygon(self.boundaries[x]) for x in self.boundaries}
        # )
        perimeter = shapely.Polygon(self.perimeter)
        for i, (name1, s1) in enumerate(all_polygons.items()):
            p1 = shapely.Polygon(s1)
            if not perimeter.contains(p1):
                self.errors.append(Errors.OUTSIDE_PERIMETER.value.format(name1))
            for j, (name2, s2) in enumerate(all_polygons.items()):
                if i >= j:
                    continue
                p2 = shapely.Polygon(s2)
                if is_overlap(p1, p2):
                    self.errors.append(
                        Errors.OVERLAPPING_OBJECTS.value.format(name1, name2)
                    )

    def check_for_errors(self):
        self.errors = []
        if self.perimeter is None:
            self.errors.append(Errors.MISSING_PERIMETER.value)

        if len(self.service_walls) == 0:
            self.errors.append(Errors.NO_SERVICE_WALLS.value)

        if len(self.internal_walls) == 0:
            self.errors.append(Errors.NO_INTERNAL_WALLS.value)

        self.check_overlap_inside()
        self.init_door_connections()
        self.check_space_access()
        self.check_windows()

        self.errors = sorted(list(set(self.errors)))
        return len(self.errors) > 0

    def sort_names(self):
        self.boundaries = dict(sorted(self.boundaries.items()))
        # self.spaces = dict(sorted(self.spaces.items()))
        self.door_windows = dict(sorted(self.door_windows.items()))

    def clean_polygons(self):
        for s in self.spaces.keys():
            r = u.simplify_small_angles(
                self.spaces[s],
                0,
            )
            if r is None:
                self.spaces.pop(s)
            else:
                self.spaces[s] = r
        for b in self.boundaries.keys():
            r = u.simplify_small_angles(
                self.boundaries[b],
                0,
            )
            if r is None:
                self.boundaries.pop(b)
            else:
                self.boundaries[b] = r
        self.perimeter = u.simplify_small_angles(self.perimeter, 0)

    def add_entrance(self):
        found = False
        for d_name, door in self.doors.items():
            for p in self.boundaries:
                l = shapely.Polygon(self.boundaries[p])
                ins = l.intersection(shapely.Polygon(door))
                if ins and isinstance(ins, shapely.Polygon):
                    self.add_door_window(DoorWindow.ENTRANCE.name, door)
                    found = True
                    break
            if found:
                self.door_windows.pop(d_name)
                break

    def add_extended_boundary(self):
        p1 = shapely.Polygon(self.perimeter)
        # ps = []
        # for s in self.spaces:
        #     if s.startswith(SpaceName.TERRACE.name):
        #         ps.append(shapely.Polygon(self.spaces[s]))
        # ps = shapely.union_all(ps)
        # p2 = p1.difference(ps)
        # ps = append_z(shapely.Polygon(p1.exterior).envelope.exterior)
        ps = append_z(p1.envelope.exterior)
        self.add_boundary(Boundary.EXTENDED_BOUNDARY.name, ps)


def load_floorplan(file):
    if isinstance(file, dict):
        load_dict = file
    else:
        with open(file, "rb") as f:
            load_dict = pickle.load(
                f,
            )

    obj = FloorplanData(
        id=load_dict["id"],
    )
    obj.north = load_dict["north"]
    obj.boundaries = load_dict["boundaries"]
    obj.spaces = load_dict["spaces"]
    obj.door_windows = load_dict["door_windows"]
    obj.door_connections = load_dict["door_connections"]
    obj.errors = load_dict["errors"]
    obj.perimeter = load_dict["perimeter"]
    for x in list(obj.boundaries.keys()):
        if x.startswith(Boundary.EXTENDED_BOUNDARY.name):
            obj.boundaries.pop(x)
    obj.add_extended_boundary()
    return obj


def get_floorplan_data(id, file):
    """Reads DXF file and converts it to FloorplanData."""
    all_names = ["PERIMETER"]
    all_names += ["SP-" + x for x in SpaceName.__members__]
    all_names += ["BO-" + x for x in Boundary.__members__]
    all_names += ["DW-" + x for x in DoorWindow.__members__]

    dxf_content = read_dxf_file(file)
    data = FloorplanData(
        id=id,
    )
    try:
        north = dxf_content.query('INSERT[name=="north"]').first
        north = np.round(north.get_dxf_attrib("rotation"))
        data.north = north
    except AttributeError:
        data.errors.append(Errors.NO_NORTH.value)

    for ele in dxf_content:
        if ele.dxftype() != "LWPOLYLINE":
            continue
        if not hasattr(ele, "dxf"):
            continue
        if not hasattr(ele.dxf, "layer"):
            continue
        if ele.dxf.layer not in all_names:
            data.errors.append(Errors.INVALID_LAYER.value.format(ele.dxf.layer))
            continue
        points = []
        for line in ele:
            points.append(line[:2])
        arr_points = round_off(points, ele.dxf.elevation)
        data.add_line(ele.dxf.layer, arr_points)
    data.sort_names()
    return data
