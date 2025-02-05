"""Class to align lines to form spaces."""

import numpy as np
import shapely

from . import utility_functions as u


def get_alignment(spaces, max_dist):
    """Return the first combination of lines for an alignment."""
    for s1 in spaces:
        ring = spaces[s1]
        for i1 in range(len(ring) - 1):
            p1, p2 = ring[i1], ring[i1 + 1]
            line = shapely.LineString([p1, p2])
            dist, (
                s2,
                i2,
            ) = get_closest_line(line, spaces, max_dist=max_dist)
            if dist == 0:
                continue
            if s2 is not None and i2 is not None:
                return (s1, i1), (s2, i2)
    return (None, None), (None, None)


def get_closest_line(l1, spaces, max_dist):
    """ "Find the closest and overlapping line."""
    dist = max_dist
    (space, ind) = None, None

    for s in spaces:
        ring = spaces[s]
        for i in range(len(ring) - 1):
            l2 = shapely.LineString([ring[i], ring[i + 1]])
            if l1.coords == l2.coords:  # must be a different line
                continue

            p1 = u.project_point(l1.coords[0], l2)
            p2 = u.project_point(l1.coords[1], l2)

            pl1 = shapely.LineString([l1.coords[0], p1])
            pl2 = shapely.LineString([l1.coords[1], p2])
            if pl1.length != pl2.length:  # must be parallel
                continue

            if pl1.length == 0:  # must have some gap
                continue

            overlap = l2.intersection(shapely.LineString([p1, p2]))
            if not isinstance(overlap, shapely.Point):
                # must have some overlap after projection.
                if overlap.length == 0:
                    continue

            if pl1.length < dist:
                dist = pl1.length
                space, ind = s, i
    return dist, (
        space,
        ind,
    )


def get_closest_alignment(a1, alignments, max_dist):
    """Find the closest alignment."""
    dist = max_dist
    ind = None
    line = None

    l1 = a1.line if isinstance(a1, Alignment) else a1
    for i, a2 in enumerate(alignments):
        if isinstance(a2, Alignment):
            l2 = a2.line
        else:
            l2 = a2
        if l1.coords == l2.coords:  # must be a different line
            continue

        p1 = u.project_point(l1.coords[0], l2)
        p2 = u.project_point(l1.coords[1], l2)

        if p1 is None or p2 is None:
            continue
        pl1 = shapely.LineString([l1.coords[0], p1])
        pl2 = shapely.LineString([l1.coords[1], p2])
        if pl1.length != pl2.length:  # must be parallel
            continue

        if pl1.length == 0:  # must have some gap
            continue

        projected_line = shapely.LineString([p1, p2])
        overlap = l2.intersection(projected_line)
        if not isinstance(overlap, shapely.Point):
            if overlap.length == 0:  # must have some overlap after projection.
                continue

        if pl1.length < dist:
            dist = pl1.length
            ind = i
            line = l2
    return ind, line


def merge_point(line, point: shapely.Point):
    """Merges a point to the line."""
    projection = u.projection_distance(point, line)
    if projection != 0:  # cannot merge if the point is not on the line.
        return False, line
    d = line.line_locate_point(point, normalized=True)
    if d == 0:
        line1 = shapely.LineString([point, *line.coords])
    elif d < 1:
        line1 = shapely.LineString([line.coords[0], point, line.coords[1]])
    else:
        line1 = shapely.LineString(
            [
                *line.coords,
                point,
            ]
        )
    line1 = line1.simplify(0)
    return True, line1


def align_line(line, align_with):
    """Align line with another line."""
    d1 = u.projection_distance(line.coords[0], align_with)
    d2 = u.projection_distance(line.coords[1], align_with)
    # if np.round(d1-d2, 3) != 0:
    #     raise RuntimeWarning("Lines to align are not parallel: ",
    #                          line, align_with, d1, d2)
    pp1 = u.project_point(line.coords[0], align_with)
    pp2 = u.project_point(line.coords[1], align_with)
    return pp1, pp2


def align_both(line1, line2):
    """Move both lines to their middle."""
    d1 = u.projection_distance(line1.coords[0], line2)
    d2 = u.projection_distance(line1.coords[1], line2)
    if np.round(d1 - d2, 3) != 0:
        raise RuntimeWarning("Lines to align are not parallel: ", line1, line2, d1, d2)

    pp1 = u.project_point(line1.coords[0], line2)
    pp2 = u.project_point(line1.coords[1], line2)
    pp3 = u.project_point(line2.coords[0], line1)
    pp4 = u.project_point(line2.coords[1], line1)

    pp1 = np.mean([pp1, line1.coords[0]], axis=0)
    pp2 = np.mean([pp2, line1.coords[1]], axis=0)
    pp3 = np.mean([pp3, line2.coords[0]], axis=0)
    pp4 = np.mean([pp4, line2.coords[1]], axis=0)

    return (pp1, pp2), (pp3, pp4)


def merge_alignments(a1, a2):
    """Merge two overlapping alignments."""
    alignment = Alignment()
    alignment.info = a1.info
    for i in a2.info:
        if i not in alignment.info:
            alignment.info[i] = []
        alignment.info[i].extend(a2.info[i])

    (pp1, pp2), (pp3, pp4) = align_both(a1.line, a2.line)
    line = shapely.LineString([pp1, pp2])
    _, line = merge_point(line, shapely.Point(pp3))
    _, line = merge_point(line, shapely.Point(pp4))
    alignment.line = line
    return alignment


class Alignment:
    """Class for alignments."""

    def __init__(
        self,
    ):
        """Initialises blank alignment."""
        self.info = dict()
        self.line = None

    def init_first(self, space, ind, line):
        """Append the first line to alignment."""
        self.info = dict()
        self.info[space] = [ind]
        self.line = line

    def append_line(self, space, ind, line):
        """Append more lines to alignment if line is one the alignment."""
        p1 = shapely.Point(line.coords[0])
        m1, n_line = merge_point(self.line, p1)
        if not m1:
            return False
        p2 = shapely.Point(line.coords[1])
        m2, n_line = merge_point(n_line, p2)
        if not m2:
            return False
        self.line = n_line
        if space not in self.info:
            self.info[space] = []
        self.info[space].append(ind)
        return True

    def move_lines(
        self,
        spaces,
    ):
        """Move lines to alignment."""
        for space in self.info:
            for i in self.info[space]:
                line = shapely.LineString((spaces[space][i], spaces[space][i + 1]))
                p = align_line(
                    line,
                    self.line,
                )
                spaces[space][i] = p[0]
                spaces[space][i + 1] = p[1]
                if i == 0:
                    spaces[space][-1] = p[0]
                if i == len(spaces[space]) - 2:
                    spaces[space][0] = p[1]

    def move_alignment(self, new_line):
        """Move alignment to a new line."""
        p1, p2 = align_line(self.line, new_line)
        self.line = shapely.LineString([p1, p2])

    def __str__(self) -> str:
        """Return string for alignment."""
        return f"Line:\t{self.line}\t{self.info}"

    def __repr__(self) -> str:
        """Return representation for alignment."""
        return self.__str__()
