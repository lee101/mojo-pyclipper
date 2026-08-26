"""pyclipper-compatible API for convex integer polygon geometry."""

from __future__ import annotations

import math
import itertools
import operator
from collections import namedtuple
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np

from ._lib import addr, lib

CT_INTERSECTION = 0
CT_UNION = 1
CT_DIFFERENCE = 2
CT_XOR = 3

PT_SUBJECT = 0
PT_CLIP = 1

PFT_EVENODD = 0
PFT_NONZERO = 1
PFT_POSITIVE = 2
PFT_NEGATIVE = 3

JT_SQUARE = 0
JT_ROUND = 1
JT_MITER = 2

ET_CLOSEDPOLYGON = 0
ET_CLOSEDLINE = 1
ET_OPENBUTT = 2
ET_OPENSQUARE = 3
ET_OPENROUND = 4

__all__ = [
    "CT_INTERSECTION",
    "CT_UNION",
    "CT_DIFFERENCE",
    "CT_XOR",
    "PT_SUBJECT",
    "PT_CLIP",
    "PFT_EVENODD",
    "PFT_NONZERO",
    "PFT_POSITIVE",
    "PFT_NEGATIVE",
    "JT_SQUARE",
    "JT_ROUND",
    "JT_MITER",
    "ET_CLOSEDPOLYGON",
    "ET_CLOSEDLINE",
    "ET_OPENBUTT",
    "ET_OPENSQUARE",
    "ET_OPENROUND",
    "PyIntRect",
    "ClipperException",
    "Area",
    "Orientation",
    "PointInPolygon",
    "PointsInPolygon",
    "ReversePath",
    "ReversePaths",
    "CleanPolygon",
    "CleanPolygons",
    "scale_to_clipper",
    "scale_from_clipper",
    "PyPolyNode",
    "PolyTreeToPaths",
    "ClosedPathsFromPolyTree",
    "OpenPathsFromPolyTree",
    "Pyclipper",
    "PyclipperOffset",
    "SimplifyPolygon",
    "SimplifyPolygons",
]

PyIntRect = namedtuple("PyIntRect", "left top right bottom")


class ClipperException(Exception):
    pass


def _path_array(path: Sequence[Sequence[Integral]], minimum: int = 0) -> np.ndarray:
    if isinstance(path, (list, tuple)):
        if not path:
            if minimum:
                raise ClipperException(f"path must contain at least {minimum} vertices")
            return np.empty((0, 2), dtype=np.int64)
        try:
            flat = np.fromiter(
                map(operator.index, itertools.chain.from_iterable(path)),
                dtype=np.int64,
            )
        except TypeError as error:
            raise TypeError("coordinates must be integer pairs") from error
        except (ValueError, OverflowError) as error:
            raise OverflowError("coordinates must lie within Clipper's 62-bit range") from error
        if flat.size != 2 * len(path):
            raise TypeError("path must be an iterable of (x, y) integer points")
        points = flat.reshape(-1, 2)
    else:
        try:
            points = np.asarray(path)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("path must be an iterable of (x, y) integer points") from error
    if points.size == 0:
        if minimum:
            raise ClipperException(f"path must contain at least {minimum} vertices")
        return np.empty((0, 2), dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise TypeError("path must be an iterable of (x, y) integer points")
    if len(points) < minimum:
        raise ClipperException(f"path must contain at least {minimum} vertices")
    if not np.issubdtype(points.dtype, np.integer):
        raise TypeError("coordinates must be integer pairs")
    if points.size and (np.any(points <= -(1 << 62)) or np.any(points >= (1 << 62))):
        raise OverflowError("coordinates must lie within Clipper's 62-bit range")
    return np.ascontiguousarray(points, dtype=np.int64)


def _signed_area_float(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    if points.dtype == np.float64 and points.flags.c_contiguous:
        return float(lib().mpc_area_f64(addr(points, np.float64), len(points)))
    if points.dtype == np.int64 and points.flags.c_contiguous:
        return float(lib().mpc_area(addr(points, np.int64), len(points)))
    converted = np.ascontiguousarray(points, dtype=np.float64)
    return float(lib().mpc_area_f64(addr(converted, np.float64), len(converted)))


def Area(poly) -> float:
    points = _path_array(poly)
    if len(points) < 3:
        return 0.0
    return float(lib().mpc_area(addr(points, np.int64), len(points)))


def Orientation(poly) -> bool:
    return Area(poly) >= 0.0


def PointInPolygon(point, poly) -> int:
    points = _path_array(poly)
    try:
        valid_point = len(point) == 2 and all(
            isinstance(value, Integral) for value in point
        )
    except TypeError:
        valid_point = False
    if not valid_point:
        raise TypeError("point must be an integer pair")
    x, y = map(operator.index, point)
    if x <= -(1 << 62) or x >= (1 << 62) or y <= -(1 << 62) or y >= (1 << 62):
        raise OverflowError("point coordinates must lie within Clipper's 62-bit range")
    if len(points) < 3:
        return 0
    return int(
        lib().mpc_point_in_polygon(addr(points, np.int64), len(points), x, y)
    )


def PointsInPolygon(points, poly) -> np.ndarray:
    """Vectorized extension: classify many points as -1 boundary, 0 outside, or 1 inside."""
    polygon = _path_array(poly)
    queries = _path_array(points)
    result = np.empty(len(queries), dtype=np.int64)
    if not len(queries):
        return result
    if len(polygon) < 3:
        result.fill(0)
    else:
        lib().mpc_points_in_polygon(
            addr(polygon, np.int64),
            len(polygon),
            addr(queries, np.int64),
            len(queries),
            addr(result, np.int64, writable=True),
        )
    return result


def ReversePath(path):
    return [list(point) for point in reversed(path)]


def ReversePaths(paths):
    return [ReversePath(path) for path in paths]


def _distance_from_line2(point, start, end) -> float:
    a = float(start[1] - end[1])
    b = float(end[0] - start[0])
    c = a * start[0] + b * start[1]
    c = a * point[0] + b * point[1] - c
    return c * c / (a * a + b * b)


def _slopes_near_collinear(first, second, third, threshold) -> bool:
    if abs(first[0] - second[0]) > abs(first[1] - second[1]):
        if (first[0] > second[0]) == (first[0] < third[0]):
            return _distance_from_line2(first, second, third) < threshold
        if (second[0] > first[0]) == (second[0] < third[0]):
            return _distance_from_line2(second, first, third) < threshold
        return _distance_from_line2(third, first, second) < threshold
    if (first[1] > second[1]) == (first[1] < third[1]):
        return _distance_from_line2(first, second, third) < threshold
    if (second[1] > first[1]) == (second[1] < third[1]):
        return _distance_from_line2(second, first, third) < threshold
    return _distance_from_line2(third, first, second) < threshold


def CleanPolygon(poly, distance=1.415):
    points = [tuple(map(int, point)) for point in _path_array(poly)]
    threshold = float(distance) ** 2
    size = len(points)
    if not size:
        return []
    previous = [(index - 1) % size for index in range(size)]
    following = [(index + 1) % size for index in range(size)]
    processed = [False] * size

    def close(first, second):
        return (
            (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 <= threshold
        )

    def exclude(index):
        before, after = previous[index], following[index]
        following[before] = after
        previous[after] = before
        processed[before] = False
        return before

    current = 0
    while not processed[current] and following[current] != previous[current]:
        before, after = previous[current], following[current]
        if close(points[current], points[before]):
            current = exclude(current)
            size -= 1
        elif close(points[before], points[after]):
            exclude(after)
            current = exclude(current)
            size -= 2
        elif _slopes_near_collinear(
            points[before], points[current], points[after], threshold
        ):
            current = exclude(current)
            size -= 1
        else:
            processed[current] = True
            current = following[current]
    if size < 3:
        return []
    result = []
    for _ in range(size):
        result.append(points[current])
        current = following[current]
    return [list(point) for point in result]


def CleanPolygons(polys, distance=1.415):
    return [CleanPolygon(poly, distance) for poly in polys]


def _scale_recursive(value, scale: Real, inverse: bool):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(item, Real) for item in value):
            if inverse:
                return [float(value[0]) / scale, float(value[1]) / scale]
            return [int(float(value[0]) * scale), int(float(value[1]) * scale)]
        return [_scale_recursive(item, scale, inverse) for item in value]
    raise TypeError("expected a path or list of paths")


def scale_to_clipper(path_or_paths, scale=1 << 31):
    return _scale_recursive(path_or_paths, scale, False)


def scale_from_clipper(path_or_paths, scale=1 << 31):
    return _scale_recursive(path_or_paths, scale, True)


def _is_convex(points: np.ndarray) -> bool:
    return len(points) >= 3 and bool(
        lib().mpc_is_convex(addr(points, np.int64), len(points))
    )


def _float_path(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float64)


def _clip_plane(poly: np.ndarray, edge_start, edge_end, orientation: float, inside: bool):
    if len(poly) < 3:
        return np.empty((0, 2), dtype=np.float64)
    target = np.empty((len(poly) + 2, 2), dtype=np.float64)
    count = lib().mpc_clip_halfplane(
        addr(poly, np.float64),
        len(poly),
        float(edge_start[0]),
        float(edge_start[1]),
        float(edge_end[0]),
        float(edge_end[1]),
        orientation,
        int(inside),
        addr(target, np.float64, writable=True),
        len(target),
    )
    if count < 0:
        raise ClipperException("native clipping output exceeded its buffer capacity")
    return np.ascontiguousarray(target[:count])


def _intersection(
    poly: np.ndarray,
    cutter: np.ndarray,
    target: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    subject = _float_path(poly)
    cutter_float = _float_path(cutter)
    orientation = 1.0 if _signed_area_float(cutter_float) > 0 else -1.0
    capacity = len(subject) + len(cutter_float) + 2
    if target is None or len(target) < capacity:
        target = np.empty((capacity, 2), dtype=np.float64)
    if scratch is None or len(scratch) < capacity:
        scratch = np.empty((capacity, 2), dtype=np.float64)
    count = lib().mpc_intersection(
        addr(subject, np.float64),
        len(subject),
        addr(cutter_float, np.float64),
        len(cutter_float),
        orientation,
        addr(target, np.float64, writable=True),
        addr(scratch, np.float64, writable=True),
        capacity,
    )
    if count < 0:
        raise ClipperException("native intersection output exceeded its buffer capacity")
    return target[:count]


def _difference(poly: np.ndarray, cutter: np.ndarray) -> list[np.ndarray]:
    remainder = _float_path(poly)
    pieces = []
    orientation = 1.0 if _signed_area_float(cutter) > 0 else -1.0
    for index in range(len(cutter)):
        start, end = cutter[index], cutter[(index + 1) % len(cutter)]
        outside = _clip_plane(remainder, start, end, orientation, False)
        if len(outside) >= 3 and abs(_signed_area_float(outside)) > 1e-8:
            pieces.append(outside)
        remainder = _clip_plane(remainder, start, end, orientation, True)
        if len(remainder) < 3:
            break
    return pieces


def _subtract_many(poly: np.ndarray, cutters: Iterable[np.ndarray]) -> list[np.ndarray]:
    pieces = [_float_path(poly)]
    for cutter in cutters:
        pieces = [part for piece in pieces for part in _difference(piece, cutter)]
        if not pieces:
            break
    return pieces


def _union(polys: Iterable[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for poly in polys:
        result.extend(_subtract_many(poly, result))
    return result


def _filled(paths: list[np.ndarray], fill_type: int) -> list[np.ndarray]:
    if fill_type == PFT_POSITIVE:
        return [path for path in paths if _signed_area_float(path) > 0]
    if fill_type == PFT_NEGATIVE:
        return [path for path in paths if _signed_area_float(path) < 0]
    return list(paths)


def _require_disjoint(paths: list[np.ndarray]):
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            overlap = _intersection(paths[left], paths[right])
            if len(overlap) >= 3 and abs(_signed_area_float(overlap)) > 1e-8:
                raise NotImplementedError(
                    "paths within one operand must have interior-disjoint regions"
                )


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(
        np.int64
    )


def _valid_inset(candidate: np.ndarray, original: np.ndarray, delta: float) -> bool:
    orientation = 1.0 if _signed_area_float(original) > 0 else -1.0
    required = abs(delta)
    for index in range(len(original)):
        start = original[index]
        end = original[(index + 1) % len(original)]
        dx, dy = end - start
        edge_length = math.sqrt(float(dx * dx + dy * dy))
        values = (
            (dx * (candidate[:, 1] - start[1]) - dy * (candidate[:, 0] - start[0]))
            * orientation
        )
        if np.any(values < required * edge_length - 1e-8):
            return False
    return True


def _clean_result(poly: np.ndarray) -> list[list[int]]:
    rounded = _round_away(poly)
    if len(rounded) > 1:
        keep = np.ones(len(rounded), dtype=bool)
        keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
        rounded = rounded[keep]
    if len(rounded) > 1 and np.array_equal(rounded[0], rounded[-1]):
        rounded = rounded[:-1]
    while len(rounded) >= 3:
        current = rounded.astype(np.float64)
        previous = np.roll(current, 1, axis=0)
        following = np.roll(current, -1, axis=0)
        cross = (current[:, 0] - previous[:, 0]) * (
            following[:, 1] - current[:, 1]
        ) - (current[:, 1] - previous[:, 1]) * (following[:, 0] - current[:, 0])
        keep = cross != 0.0
        if np.all(keep):
            break
        rounded = rounded[keep]
    area = _signed_area_float(rounded)
    if len(rounded) < 3 or area == 0.0:
        return []
    if area < 0.0:
        rounded = rounded[::-1]
    return rounded.tolist()


def _finish(polys: Iterable[np.ndarray], reverse: bool = False):
    result = [clean for poly in polys if (clean := _clean_result(poly))]
    if reverse:
        result = [ReversePath(poly) for poly in result]
    return result


class PyPolyNode:
    def __init__(self, contour=None, parent=None, is_hole=False, is_open=False):
        self.Contour = contour or []
        self.Parent = parent
        self.Childs = []
        self.IsHole = bool(is_hole)
        self.IsOpen = bool(is_open)

    def GetNext(self):
        if self.Childs:
            return self.Childs[0]
        node = self
        while node.Parent is not None:
            siblings = node.Parent.Childs
            index = siblings.index(node)
            if index + 1 < len(siblings):
                return siblings[index + 1]
            node = node.Parent
        return None


def _tree(paths):
    root = PyPolyNode()
    for path in paths:
        root.Childs.append(PyPolyNode(path, root))
    return root


def PolyTreeToPaths(polytree):
    paths = []

    def visit(node):
        if node.Contour:
            paths.append(node.Contour)
        for child in node.Childs:
            visit(child)

    visit(polytree)
    return paths


def ClosedPathsFromPolyTree(polytree):
    return [path for node, path in _walk_tree(polytree) if not node.IsOpen]


def OpenPathsFromPolyTree(polytree):
    return [path for node, path in _walk_tree(polytree) if node.IsOpen]


def _walk_tree(root):
    for child in root.Childs:
        if child.Contour:
            yield child, child.Contour
        yield from _walk_tree(child)


class Pyclipper:
    def __init__(self, init_options=0):
        if init_options != 0:
            raise NotImplementedError("nonzero Pyclipper init_options are not supported")
        self._subject = []
        self._clip = []
        self._preserve_collinear = False
        self.ReverseSolution = False
        self._strictly_simple = False
        self._intersection_target = None
        self._intersection_scratch = None

    @property
    def PreserveCollinear(self):
        return self._preserve_collinear

    @PreserveCollinear.setter
    def PreserveCollinear(self, value):
        if value:
            raise NotImplementedError("PreserveCollinear=True is not supported")
        self._preserve_collinear = False

    @property
    def StrictlySimple(self):
        return self._strictly_simple

    @StrictlySimple.setter
    def StrictlySimple(self, value):
        if value:
            raise NotImplementedError("StrictlySimple=True is not supported")
        self._strictly_simple = False

    def AddPath(self, path, poly_type, closed=True):
        if not closed:
            raise NotImplementedError("open paths are not supported")
        points = _path_array(path, 3)
        if np.abs(points).max(initial=0) > (1 << 52):
            raise NotImplementedError("clipping coordinates must not exceed 2**52")
        if not _is_convex(points):
            raise NotImplementedError("only simple convex paths are supported")
        native_points = _float_path(points)
        if poly_type == PT_SUBJECT:
            self._subject.append(native_points)
        elif poly_type == PT_CLIP:
            self._clip.append(native_points)
        else:
            raise ClipperException("invalid polygon type")
        return True

    def AddPaths(self, paths, poly_type, closed=True):
        added = False
        for path in paths:
            added = self.AddPath(path, poly_type, closed) or added
        return added

    def Clear(self):
        self._subject.clear()
        self._clip.clear()

    def GetBounds(self):
        paths = self._subject + self._clip
        if not paths:
            return PyIntRect(0, 0, 0, 0)
        points = np.concatenate(paths)
        return PyIntRect(
            int(points[:, 0].min()),
            int(points[:, 1].min()),
            int(points[:, 0].max()),
            int(points[:, 1].max()),
        )

    def Execute(self, clip_type, subj_fill_type=PFT_EVENODD, clip_fill_type=PFT_EVENODD):
        if clip_type not in (CT_INTERSECTION, CT_UNION, CT_DIFFERENCE, CT_XOR):
            raise ClipperException("invalid clip type")
        if subj_fill_type not in (PFT_EVENODD, PFT_NONZERO, PFT_POSITIVE, PFT_NEGATIVE):
            raise ClipperException("invalid subject fill type")
        if clip_fill_type not in (PFT_EVENODD, PFT_NONZERO, PFT_POSITIVE, PFT_NEGATIVE):
            raise ClipperException("invalid clip fill type")
        subject_paths = _filled(self._subject, subj_fill_type)
        clip_paths = _filled(self._clip, clip_fill_type)
        _require_disjoint(subject_paths)
        _require_disjoint(clip_paths)
        if clip_type == CT_INTERSECTION:
            if len(subject_paths) == 1 and len(clip_paths) == 1:
                capacity = len(subject_paths[0]) + len(clip_paths[0]) + 2
                if self._intersection_target is None or len(self._intersection_target) < capacity:
                    self._intersection_target = np.empty((capacity, 2), dtype=np.float64)
                    self._intersection_scratch = np.empty_like(self._intersection_target)
                raw = [
                    _intersection(
                        subject_paths[0],
                        clip_paths[0],
                        self._intersection_target,
                        self._intersection_scratch,
                    )
                ]
            else:
                raw = _union(
                    _intersection(subject, clip)
                    for subject in subject_paths
                    for clip in clip_paths
                )
        elif clip_type == CT_DIFFERENCE:
            raw = [
                piece
                for subject in subject_paths
                for piece in _subtract_many(subject, clip_paths)
            ]
        elif clip_type == CT_UNION:
            raw = _union(subject_paths + clip_paths)
        else:
            raw = [
                piece
                for subject in subject_paths
                for piece in _subtract_many(subject, clip_paths)
            ]
            raw.extend(
                piece
                for clip in clip_paths
                for piece in _subtract_many(clip, subject_paths)
            )
        return _finish(raw, self.ReverseSolution)

    def Execute2(self, clip_type, subj_fill_type=PFT_EVENODD, clip_fill_type=PFT_EVENODD):
        return _tree(self.Execute(clip_type, subj_fill_type, clip_fill_type))


class PyclipperOffset:
    def __init__(self, miter_limit=2.0, arc_tolerance=0.25):
        self.MiterLimit = float(miter_limit)
        self.ArcTolerance = float(arc_tolerance)
        self._paths = []

    def AddPath(self, path, join_type, end_type):
        if join_type not in (JT_MITER, JT_SQUARE, JT_ROUND):
            raise ClipperException("invalid join type")
        if end_type != ET_CLOSEDPOLYGON:
            raise NotImplementedError("only ET_CLOSEDPOLYGON is supported")
        points = _path_array(path, 3)
        if np.abs(points).max(initial=0) > (1 << 52):
            raise NotImplementedError("offset coordinates must not exceed 2**52")
        if not _is_convex(points):
            raise NotImplementedError("only simple convex paths are supported")
        self._paths.append((_float_path(points), join_type))

    def AddPaths(self, paths, join_type, end_type):
        for path in paths:
            self.AddPath(path, join_type, end_type)

    def Clear(self):
        self._paths.clear()

    def Execute(self, delta):
        if not isinstance(delta, Real) or not math.isfinite(float(delta)):
            raise TypeError("delta must be a finite real number")
        raw = []
        for integer_points, join_type in self._paths:
            points = _float_path(integer_points)
            normals = np.empty_like(points)
            if delta == 0:
                raw.append(points)
                continue
            tolerance = min(abs(float(delta)) * 0.25, max(0.01, self.ArcTolerance))
            step = 2 * np.arccos(
                max(-1.0, min(1.0, 1.0 - tolerance / abs(float(delta))))
            )
            circle_steps = max(4, int(np.ceil(2 * np.pi / max(step, 1e-6))))
            capacity = len(points) * (circle_steps + 4)
            target = np.empty((capacity, 2), dtype=np.float64)
            count = lib().mpc_offset_polygon(
                addr(points, np.float64),
                addr(normals, np.float64, writable=True),
                len(points),
                float(delta),
                join_type,
                self.MiterLimit,
                self.ArcTolerance,
                addr(target, np.float64, writable=True),
                capacity,
            )
            if count < 0:
                raise ClipperException("native offset output exceeded its buffer capacity")
            candidate = np.ascontiguousarray(target[:count])
            feasible = float(delta) >= 0 or _valid_inset(candidate, points, float(delta))
            if (
                len(candidate) >= 3
                and feasible
                and _signed_area_float(candidate) * _signed_area_float(points) > 0
            ):
                raw.append(candidate)
        return _finish(_union(raw))

    def Execute2(self, delta):
        return _tree(self.Execute(delta))


def SimplifyPolygon(poly, fill_type=PFT_EVENODD):
    points = _path_array(poly, 3)
    if not _is_convex(points):
        raise NotImplementedError("only already-simple convex polygons are supported")
    return [_clean_result(points.astype(np.float64))]


def SimplifyPolygons(polys, fill_type=PFT_EVENODD):
    return [path for poly in polys for path in SimplifyPolygon(poly, fill_type)]
