import math

import numpy as np
import pyclipper as reference
import pytest

import mojo_pyclipper as mpc
from mojo_pyclipper._lib import addr


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]
CLIP = [(50, -10), (80, -10), (80, 110), (50, 110)]


def canonical(path):
    points = [tuple(point) for point in path]
    if mpc.Area(points) < 0:
        points.reverse()
    rotations = [tuple(points[index:] + points[:index]) for index in range(len(points))]
    return min(rotations)


def regular_polygon(vertices, radius=1000, center=(0, 0), phase=0.0):
    cx, cy = center
    return [
        (
            round(cx + radius * math.cos(phase + 2 * math.pi * index / vertices)),
            round(cy + radius * math.sin(phase + 2 * math.pi * index / vertices)),
        )
        for index in range(vertices)
    ]


def execute(module, clip_type, subject=SQUARE, clip=CLIP):
    pc = module.Pyclipper()
    pc.AddPath(subject, module.PT_SUBJECT, True)
    pc.AddPath(clip, module.PT_CLIP, True)
    return pc.Execute(clip_type, module.PFT_NONZERO, module.PFT_NONZERO)


def covered(solution, point, module):
    return any(module.PointInPolygon(point, path) != 0 for path in solution)


def total_area(paths, module):
    return sum(abs(module.Area(path)) for path in paths)


def test_constants_match_upstream():
    names = [
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
    ]
    assert {name: getattr(mpc, name) for name in names} == {
        name: getattr(reference, name) for name in names
    }


@pytest.mark.parametrize(
    "path",
    [
        SQUARE,
        list(reversed(SQUARE)),
        [(0, 0), (9, 0), (4, 7)],
        regular_polygon(127, 10_000, (300, -800), 0.13),
    ],
)
def test_area_matches_upstream(path):
    assert mpc.Area(path) == reference.Area(path)


@pytest.mark.parametrize("vertices", [3, 4, 5, 6, 7, 9, 127])
def test_area_simd_tail_matches_upstream(vertices):
    path = regular_polygon(vertices, 100_000, (91, -37), 0.017)
    assert mpc.Area(path) == reference.Area(path)
    assert mpc.Area(np.asarray(path, dtype=np.int64)) == reference.Area(path)


def test_orientation_matches_upstream():
    for path in (SQUARE, list(reversed(SQUARE)), regular_polygon(17)):
        assert mpc.Orientation(path) == reference.Orientation(path)


def test_point_in_polygon_boundary_and_interior_match():
    for point in [(0, 0), (50, 0), (100, 30), (50, 50), (-1, 50), (101, 101)]:
        assert mpc.PointInPolygon(point, SQUARE) == reference.PointInPolygon(point, SQUARE)


def test_point_in_polygon_random_parity():
    polygon = regular_polygon(37, 10_000, (127, -331), 0.071)
    rng = np.random.default_rng(7)
    queries = rng.integers(-12_000, 12_000, size=(500, 2))
    assert [mpc.PointInPolygon(point, polygon) for point in queries] == [
        reference.PointInPolygon(point, polygon) for point in queries
    ]


def test_vectorized_points_in_polygon_matches_upstream_scalar_api():
    polygon = regular_polygon(101, 50_000)
    rng = np.random.default_rng(11)
    queries = rng.integers(-60_000, 60_000, size=(2_000, 2), dtype=np.int64)
    got = mpc.PointsInPolygon(queries, polygon)
    expected = np.array(
        [reference.PointInPolygon(point, polygon) for point in queries], dtype=np.int64
    )
    assert np.array_equal(got, expected)


def test_empty_vectorized_query_does_not_cross_ffi():
    result = mpc.PointsInPolygon(np.empty((0, 2), dtype=np.int64), SQUARE)
    assert result.shape == (0,) and result.dtype == np.int64


def test_ffi_address_rejects_wrong_layout_dtype_and_empty_buffers():
    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones((2, 2), dtype=np.int32), np.int64)
    with pytest.raises(ValueError, match="contiguous"):
        addr(np.ones((2, 4), dtype=np.int64)[:, ::2], np.int64)
    with pytest.raises(ValueError, match="empty"):
        addr(np.empty((0, 2), dtype=np.int64), np.int64)
    array = np.ones((2, 2), dtype=np.int64)
    array.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        addr(array, np.int64, writable=True)


def test_reverse_helpers_match_upstream():
    paths = [SQUARE, CLIP]
    assert mpc.ReversePath(SQUARE) == reference.ReversePath(SQUARE)
    assert mpc.ReversePaths(paths) == reference.ReversePaths(paths)


def test_scale_helpers_match_upstream():
    path = [[0.0, 0.25], [-0.5, 3.125], [7.0, -9.75]]
    for scale in (10, 1 << 20):
        clipped = mpc.scale_to_clipper(path, scale)
        assert clipped == reference.scale_to_clipper(path, scale)
        assert mpc.scale_from_clipper(clipped, scale) == reference.scale_from_clipper(
            clipped, scale
        )


@pytest.mark.parametrize(
    "path",
    [
        [(0, 0), (1, 0), (10, 0), (10, 10), (0, 10)],
        [(0, 0), (2, 0), (10, 0), (10, 10), (0, 10)],
        [(0, 0), (10, 0), (10, 1), (10, 10), (0, 10)],
        [(0, 0), (10, 0), (9, 1), (10, 10), (0, 10)],
        [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)],
    ],
)
def test_clean_polygon_matches_upstream(path):
    assert [list(point) for point in mpc.CleanPolygon(path)] == reference.CleanPolygon(path)


def test_clean_polygons_matches_upstream():
    paths = [SQUARE + [(0, 0)], [(0, 0), (5, 0), (10, 0), (0, 10)]]
    got = [[list(point) for point in path] for path in mpc.CleanPolygons(paths)]
    assert got == reference.CleanPolygons(paths)


def test_get_bounds_matches_upstream():
    ours, theirs = mpc.Pyclipper(), reference.Pyclipper()
    for module, instance in ((mpc, ours), (reference, theirs)):
        instance.AddPath(SQUARE, module.PT_SUBJECT)
        instance.AddPath(CLIP, module.PT_CLIP)
    assert ours.GetBounds() == theirs.GetBounds()
    ours.Clear()
    assert ours.GetBounds() == mpc.PyIntRect(0, 0, 0, 0)


def test_rectangle_intersection_vertices_match_upstream():
    assert canonical(execute(mpc, mpc.CT_INTERSECTION)[0]) == canonical(
        execute(reference, reference.CT_INTERSECTION)[0]
    )


@pytest.mark.parametrize(
    "clip_type",
    [mpc.CT_INTERSECTION, mpc.CT_UNION, mpc.CT_DIFFERENCE, mpc.CT_XOR],
)
def test_boolean_operations_match_upstream_coverage(clip_type):
    got = execute(mpc, clip_type)
    expected = execute(reference, clip_type)
    for x in range(-20, 121, 7):
        for y in range(-20, 121, 7):
            point = (x, y)
            if any(reference.PointInPolygon(point, path) == -1 for path in expected):
                continue
            assert covered(got, point, mpc) == covered(expected, point, reference)


@pytest.mark.parametrize(
    "clip_type",
    [mpc.CT_INTERSECTION, mpc.CT_UNION, mpc.CT_DIFFERENCE, mpc.CT_XOR],
)
def test_boolean_operation_areas_obey_set_identities(clip_type):
    got = execute(mpc, clip_type)
    area_a = abs(mpc.Area(SQUARE))
    area_b = abs(mpc.Area(CLIP))
    intersection = total_area(execute(mpc, mpc.CT_INTERSECTION), mpc)
    expected = {
        mpc.CT_INTERSECTION: intersection,
        mpc.CT_UNION: area_a + area_b - intersection,
        mpc.CT_DIFFERENCE: area_a - intersection,
        mpc.CT_XOR: area_a + area_b - 2 * intersection,
    }[clip_type]
    assert total_area(got, mpc) == expected


@pytest.mark.parametrize("seed", range(8))
def test_random_convex_boolean_coverage_matches_upstream(seed):
    subject = regular_polygon(3 + seed % 7, 900 + 17 * seed, (0, 0), 0.13 * seed)
    clip = regular_polygon(4 + seed % 6, 750, (220 + 9 * seed, -130), 0.19 * seed)
    rng = np.random.default_rng(seed)
    queries = rng.integers(-1_200, 1_300, size=(200, 2))
    for clip_type in (
        mpc.CT_INTERSECTION,
        mpc.CT_UNION,
        mpc.CT_DIFFERENCE,
        mpc.CT_XOR,
    ):
        got = execute(mpc, clip_type, subject, clip)
        for point in queries:
            in_subject = reference.PointInPolygon(point, subject)
            in_clip = reference.PointInPolygon(point, clip)
            if in_subject == -1 or in_clip == -1:
                continue
            expected = {
                mpc.CT_INTERSECTION: bool(in_subject and in_clip),
                mpc.CT_UNION: bool(in_subject or in_clip),
                mpc.CT_DIFFERENCE: bool(in_subject and not in_clip),
                mpc.CT_XOR: bool(in_subject) != bool(in_clip),
            }[clip_type]
            assert covered(got, point, mpc) == expected


def test_multiple_disjoint_subject_paths_match_upstream_area():
    subjects = [SQUARE, [(200, 0), (260, 0), (260, 100), (200, 100)]]
    ours, theirs = mpc.Pyclipper(), reference.Pyclipper()
    ours.AddPaths(subjects, mpc.PT_SUBJECT)
    theirs.AddPaths(subjects, reference.PT_SUBJECT)
    got = ours.Execute(mpc.CT_UNION, mpc.PFT_NONZERO, mpc.PFT_NONZERO)
    expected = theirs.Execute(
        reference.CT_UNION, reference.PFT_NONZERO, reference.PFT_NONZERO
    )
    assert total_area(got, mpc) == total_area(expected, reference)


def test_multiple_disjoint_clip_paths_difference_matches_upstream_area():
    clips = [
        [(10, -10), (30, -10), (30, 110), (10, 110)],
        [(70, -10), (90, -10), (90, 110), (70, 110)],
    ]
    ours, theirs = mpc.Pyclipper(), reference.Pyclipper()
    ours.AddPath(SQUARE, mpc.PT_SUBJECT)
    theirs.AddPath(SQUARE, reference.PT_SUBJECT)
    ours.AddPaths(clips, mpc.PT_CLIP)
    theirs.AddPaths(clips, reference.PT_CLIP)
    got = ours.Execute(mpc.CT_DIFFERENCE, mpc.PFT_NONZERO, mpc.PFT_NONZERO)
    expected = theirs.Execute(
        reference.CT_DIFFERENCE, reference.PFT_NONZERO, reference.PFT_NONZERO
    )
    assert total_area(got, mpc) == total_area(expected, reference)


def test_positive_and_negative_fill_types_match_upstream():
    clockwise = list(reversed(SQUARE))
    for fill_type in (mpc.PFT_POSITIVE, mpc.PFT_NEGATIVE):
        ours, theirs = mpc.Pyclipper(), reference.Pyclipper()
        ours.AddPath(clockwise, mpc.PT_SUBJECT)
        theirs.AddPath(clockwise, reference.PT_SUBJECT)
        assert bool(ours.Execute(mpc.CT_UNION, fill_type)) == bool(
            theirs.Execute(reference.CT_UNION, fill_type)
        )


def test_reverse_solution_matches_upstream_orientation():
    ours = mpc.Pyclipper()
    ours.ReverseSolution = True
    ours.AddPath(SQUARE, mpc.PT_SUBJECT)
    result = ours.Execute(mpc.CT_UNION)
    assert len(result) == 1 and not mpc.Orientation(result[0])


def test_execute2_and_tree_helpers():
    pc = mpc.Pyclipper()
    pc.AddPath(SQUARE, mpc.PT_SUBJECT)
    tree = pc.Execute2(mpc.CT_UNION)
    expected = [[list(point) for point in SQUARE]]
    assert mpc.PolyTreeToPaths(tree) == expected
    assert mpc.ClosedPathsFromPolyTree(tree) == expected
    assert mpc.OpenPathsFromPolyTree(tree) == []
    assert tree.Childs[0].Parent is tree
    assert tree.GetNext() is tree.Childs[0]
    assert tree.Childs[0].GetNext() is None


def test_nonzero_init_options_fail_explicitly():
    with pytest.raises(NotImplementedError, match="init_options"):
        mpc.Pyclipper(1)
    pc = mpc.Pyclipper()
    with pytest.raises(NotImplementedError, match="PreserveCollinear"):
        pc.PreserveCollinear = True
    with pytest.raises(NotImplementedError, match="StrictlySimple"):
        pc.StrictlySimple = True


def test_nonconvex_and_open_paths_fail_explicitly():
    concave = [(0, 0), (10, 0), (4, 4), (10, 10), (0, 10)]
    with pytest.raises(NotImplementedError, match="convex"):
        mpc.Pyclipper().AddPath(concave, mpc.PT_SUBJECT)
    with pytest.raises(NotImplementedError, match="open"):
        mpc.Pyclipper().AddPath(SQUARE, mpc.PT_SUBJECT, False)


def test_overlapping_paths_within_one_operand_fail_explicitly():
    pc = mpc.Pyclipper()
    pc.AddPath(SQUARE, mpc.PT_SUBJECT)
    pc.AddPath([(10, 10), (110, 10), (110, 110), (10, 110)], mpc.PT_SUBJECT)
    with pytest.raises(NotImplementedError, match="interior-disjoint"):
        pc.Execute(mpc.CT_UNION)


def offset(module, path, join_type, delta, arc_tolerance=0.25):
    instance = module.PyclipperOffset(miter_limit=2.0, arc_tolerance=arc_tolerance)
    instance.AddPath(path, join_type, module.ET_CLOSEDPOLYGON)
    return instance.Execute(delta)


@pytest.mark.parametrize("delta", [10, -10])
def test_miter_rectangle_offset_matches_upstream_exactly(delta):
    assert canonical(offset(mpc, SQUARE, mpc.JT_MITER, delta)[0]) == canonical(
        offset(reference, SQUARE, reference.JT_MITER, delta)[0]
    )


@pytest.mark.parametrize("vertices", [3, 4, 5, 8, 20])
def test_square_offset_matches_upstream_exactly(vertices):
    polygon = regular_polygon(vertices)
    got = offset(mpc, polygon, mpc.JT_SQUARE, 25)
    expected = offset(reference, polygon, reference.JT_SQUARE, 25)
    assert canonical(got[0]) == canonical(expected[0])


@pytest.mark.parametrize("vertices", [3, 4, 8, 20])
def test_round_offset_respects_upstream_arc_tolerance(vertices):
    polygon = regular_polygon(vertices)
    got = offset(mpc, polygon, mpc.JT_ROUND, 25)
    expected = offset(reference, polygon, reference.JT_ROUND, 25)
    assert total_area(got, mpc) == pytest.approx(
        total_area(expected, reference), rel=2e-5
    )


def test_zero_and_overlarge_negative_offsets_match_upstream():
    assert canonical(offset(mpc, SQUARE, mpc.JT_MITER, 0)[0]) == canonical(SQUARE)
    assert offset(mpc, SQUARE, mpc.JT_MITER, -60) == offset(
        reference, SQUARE, reference.JT_MITER, -60
    )


def test_offset_execute2_returns_closed_paths():
    instance = mpc.PyclipperOffset()
    instance.AddPath(SQUARE, mpc.JT_MITER, mpc.ET_CLOSEDPOLYGON)
    tree = instance.Execute2(10)
    assert mpc.ClosedPathsFromPolyTree(tree) == instance.Execute(10)


def test_multiple_disjoint_offsets_match_upstream_area():
    paths = [SQUARE, [(200, 0), (300, 0), (300, 100), (200, 100)]]
    ours = mpc.PyclipperOffset()
    theirs = reference.PyclipperOffset()
    ours.AddPaths(paths, mpc.JT_MITER, mpc.ET_CLOSEDPOLYGON)
    theirs.AddPaths(paths, reference.JT_MITER, reference.ET_CLOSEDPOLYGON)
    assert total_area(ours.Execute(10), mpc) == total_area(
        theirs.Execute(10), reference
    )
    ours.Clear()
    assert ours.Execute(10) == []


@pytest.mark.parametrize("vertices", [32_767, 262_147])
def test_offset_parallel_threshold_matches_upstream(vertices):
    polygon = regular_polygon(vertices, 1_000_000_000_000, phase=0.003)
    got = offset(mpc, polygon, mpc.JT_MITER, 10_000)
    expected = offset(reference, polygon, reference.JT_MITER, 10_000)
    assert len(got) == len(expected) == 1
    assert total_area(got, mpc) == pytest.approx(
        total_area(expected, reference), rel=1e-12
    )


def test_offset_rejects_open_and_nonconvex_paths():
    instance = mpc.PyclipperOffset()
    with pytest.raises(NotImplementedError, match="ET_CLOSEDPOLYGON"):
        instance.AddPath(SQUARE, mpc.JT_MITER, mpc.ET_OPENBUTT)
    with pytest.raises(NotImplementedError, match="convex"):
        instance.AddPath(
            [(0, 0), (10, 0), (5, 5), (10, 10), (0, 10)],
            mpc.JT_MITER,
            mpc.ET_CLOSEDPOLYGON,
        )


def test_simplify_convex_polygon_matches_upstream_geometry():
    got = mpc.SimplifyPolygon(SQUARE)
    expected = reference.SimplifyPolygon(SQUARE)
    assert canonical(got[0]) == canonical(expected[0])


def test_simplify_polygons_matches_upstream_geometry():
    paths = [SQUARE, [(200, 0), (300, 0), (250, 100)]]
    got = mpc.SimplifyPolygons(paths)
    expected = reference.SimplifyPolygons(paths)
    assert sorted(map(canonical, got)) == sorted(map(canonical, expected))


def test_noninteger_coordinates_are_rejected_like_upstream():
    with pytest.raises(TypeError):
        mpc.Area([(0.5, 0), (1, 0), (0, 1)])


def test_scalar_query_rejects_coordinates_that_ctypes_would_narrow():
    with pytest.raises(OverflowError, match="62-bit"):
        mpc.PointInPolygon((1 << 70, 0), SQUARE)


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), "10"])
def test_offset_rejects_nonfinite_or_nonreal_delta(delta):
    instance = mpc.PyclipperOffset()
    instance.AddPath(SQUARE, mpc.JT_MITER, mpc.ET_CLOSEDPOLYGON)
    with pytest.raises(TypeError, match="finite real"):
        instance.Execute(delta)
