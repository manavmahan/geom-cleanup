"""Class to store shape information."""

import numpy as np
import shapely

from . import utility_functions as u
from .align_utils import Alignment, align_both
from .align_utils import (
    get_closest_alignment,
    merge_alignments,
    align_line,
    merge_point,
)


def is_all_points_unique(polygon):
    n_polygon = np.unique(polygon, axis=0)
    if all(polygon[0] == polygon[-1]):
        return len(n_polygon) == len(polygon) - 1
    return len(n_polygon) == len(polygon)


def clean_polygon(polygon, name):
    if is_all_points_unique(polygon):
        return polygon
    polygon = shapely.Polygon(polygon)
    polygon = polygon.simplify(
        2,
    )
    polygon = np.array(polygon.exterior.coords)
    if is_all_points_unique(polygon):
        return polygon
    raise RuntimeError(f"Not a simple polygon {name}: {shapely.Polygon(polygon)}")


def clean_triangles(triangles):
    def unique_filter(triangle):
        triangle = np.unique(triangle, axis=0)
        return len(triangle) == 3

    return np.array(list(filter(unique_filter, triangles)))


class Spaces:
    """Class to store shape information"""

    def __init__(self, ext_ring, rooms, doors) -> None:
        self.ext_ring = np.array(ext_ring)
        if self.ext_ring.shape[1] > 2:
            self.ext_ring = self.ext_ring[:, :2]
        self.ext_ring = u.simplify_small_lines(self.ext_ring, 1)
        self.ext_alignment = []
        for p1, p2 in zip(self.ext_ring[:-1], self.ext_ring[1:]):
            line = shapely.LineString([p1, p2])
            self.ext_alignment.append(line)

        self.spaces = dict()
        for room in rooms:
            coords = np.array(rooms[room])
            if coords.shape[1] > 2:
                coords = coords[:, :2]
            coords = u.simplify_small_lines(coords, 1)
            for p1, p2 in zip(coords[:-1], coords[1:]):
                line = shapely.LineString([p1, p2])
            self.spaces[room] = np.array(coords)

        self.doors = dict()
        for d in doors:
            if doors[d].shape[1] > 2:
                doors[d] = doors[d][:, :2]
            self.doors[d] = np.array(doors[d])
        self.__alignments = list()

    def update_spaces(self):
        for s in list(self.spaces.keys()):
            r = shapely.LinearRing(self.spaces[s])
            r = u.simplify_small_angles(
                np.array(r.coords),
                0,
            )
            if r is None:
                self.spaces.pop(s)
            else:
                self.spaces[s] = r

    def append_to_alignment(self, s1, i1, l1):
        appended = False
        for a in self.__alignments:
            appended = a.append_line(s1, i1, l1)
            if appended:
                return
        b = Alignment()
        b.init_first(s1, i1, l1)
        self.__alignments.append(b)

    def __find_alignments(self):
        self.__alignments = []
        for s1 in self.spaces:
            r1 = self.spaces[s1]
            for i1, (p1, p2) in enumerate(zip(r1[:-1], r1[1:])):
                l1 = shapely.LineString([p1, p2])
                self.append_to_alignment(s1, i1, l1)

    def align_lines(self, tolerance):
        self.__find_alignments()
        # print('\n'.join([str(x) for x in self.__alignments]))
        self.merge_alignments_internal(tolerance)
        for a in self.__alignments:
            a.move_lines(self.spaces)
        self.merge_alignments_external(tolerance)
        self.update_spaces()

        # merge after creating spaces
        self.__find_alignments()
        self.merge_alignments_internal(tolerance)
        for a in self.__alignments:
            a.move_lines(self.spaces)
        self.update_spaces()
        # self.get_door_lines()

    def merge_alignments_internal(self, distance):
        rerun = False
        for i in range(len(self.__alignments)):
            j, _ = get_closest_alignment(
                self.__alignments[i], self.__alignments, distance
            )

            if j is not None:
                a1 = self.__alignments[i]
                a2 = self.__alignments[j]
                alignment = merge_alignments(a1, a2)
                del self.__alignments[i]
                del self.__alignments[j - 1]
                self.__alignments.append(alignment)
                rerun = True
                break

        if rerun:
            self.merge_alignments_internal(distance)

    def merge_alignments_external(self, distance):
        for i in range(len(self.__alignments)):
            x1, l1 = get_closest_alignment(
                self.__alignments[i], self.ext_alignment, distance
            )
            if x1 is not None:
                a1 = self.__alignments[i]
                a1.move_alignment(l1)
                a1.move_lines(self.spaces)

    def get_door_lines(self, tolerance):
        """Move door lines to alignment."""
        n_doors = []
        for door in self.doors:
            for p1, p2 in zip(door[:-1], door[1:]):
                l = shapely.LineString([p1, p2])
                if l.length < tolerance / 2:
                    continue
                _, a = get_closest_alignment(
                    l, self.__alignments, max_dist=tolerance / 2
                )
                if a is not None:
                    p1, p2 = align_line(l, a)
                    n_doors.append(shapely.LineString([p1, p2]))
                else:
                    n_doors.append(l)
        self.doors = n_doors
        self.merge_doors(tolerance)
        self.break_door_lines(tolerance)
        self.rotate_door_lines()

    def merge_doors(self, tolerance):
        i1 = 0
        while i1 < len(self.doors):
            l1 = self.doors[i1]
            for i2 in range(i1, len(self.doors)):
                l2 = self.doors[i2]
                _, l2 = get_closest_alignment(l1, [l2], max_dist=tolerance)
                if l2 is not None:
                    ps = align_both(l1, l2)
                    l1 = shapely.LineString(ps[0])
                    _, l1 = merge_point(l1, shapely.Point(ps[1][0]))
                    _, l1 = merge_point(l1, shapely.Point(ps[1][1]))
                    self.doors[i1] = l1.coords
                    del self.doors[i2]
                    break
                merged, l1 = merge_point(l1, shapely.Point(self.doors[i2].coords[0]))
                if merged:
                    merged, l1 = merge_point(
                        l1, shapely.Point(self.doors[i2].coords[1])
                    )
                    if merged:
                        self.doors[i1] = l1.coords
                        del self.doors[i2]
                        break
            i1 += 1

    def break_door_lines(self, tolerance):
        n_doors = []
        space_lines = []
        for s in self.spaces:
            ring = self.spaces[s]
            for l in zip(ring, np.roll(ring, 1, axis=0)):
                space_lines.append(shapely.LineString(l))
        for door in self.doors:
            split = False
            for line in space_lines:
                split_doors = intersect_and_split(door, line, tolerance)
                split = len(split_doors) > 1
                if split:
                    n_doors.extend(split_doors)
                    break
            if not split:
                n_doors.append(door)
        self.doors = n_doors

    def rotate_door_lines(self):
        for i in range(len(self.doors)):
            line = self.doors[i]
            mid_point = line.interpolate(0.5, normalized=True)
            rotated_line = shapely.affinity.rotate(
                line, 90, origin=mid_point, use_radians=False
            )
            self.doors[i] = rotated_line.coords


def intersect_and_split(line1, line2, tolerance):
    point = line1.intersection(line2)
    if isinstance(point, shapely.Point):
        intersection_point = point.coords[0]
        coords = list(line1.coords)

        l1 = shapely.LineString([coords[0], intersection_point])
        l2 = shapely.LineString([intersection_point, coords[1]])
        return list(filter(lambda x: x.length > tolerance / 2, [l1, l2]))
    return [line1]
