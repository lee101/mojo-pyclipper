"""End-to-end benchmarks against pyclipper on identical integer polygons."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_pyclipper as mpc  # noqa: E402
import pyclipper as reference  # noqa: E402


def regular_polygon(vertices, radius=1_000_000_000, phase=0.0):
    return [
        (
            round(radius * math.cos(phase + 2 * math.pi * index / vertices)),
            round(radius * math.sin(phase + 2 * math.pi * index / vertices)),
        )
        for index in range(vertices)
    ]


def time_best(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def offset(module, polygon, join_type, delta):
    instance = module.PyclipperOffset(miter_limit=2.0, arc_tolerance=0.25)
    instance.AddPath(polygon, join_type, module.ET_CLOSEDPOLYGON)
    return lambda: instance.Execute(delta)


def intersection(module, subject, clip):
    instance = module.Pyclipper()
    instance.AddPath(subject, module.PT_SUBJECT)
    instance.AddPath(clip, module.PT_CLIP)
    return lambda: instance.Execute(
        module.CT_INTERSECTION, module.PFT_NONZERO, module.PFT_NONZERO
    )


def machine():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            line = next(line for line in stream if line.startswith("model name"))
        model = line.split(":", 1)[1].strip()
    except (OSError, StopIteration):
        pass
    return f"{model or 'unknown CPU'}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def main():
    area_poly = regular_polygon(250_000)
    pip_poly = regular_polygon(256)
    rng = np.random.default_rng(42)
    queries = rng.integers(
        -1_200_000_000, 1_200_000_000, size=(25_000, 2), dtype=np.int64
    )
    clip_subject = regular_polygon(1_000)
    clip_clip = [
        (x + 175_000_000, y - 80_000_000)
        for x, y in regular_polygon(1_000, 850_000_000, 0.0017)
    ]
    offset_poly = regular_polygon(100_000)

    cases = [
        (
            "Area, 250k vertices",
            lambda: mpc.Area(area_poly),
            lambda: reference.Area(area_poly),
            7,
        ),
        (
            "PointInPolygon, 25k points x 256 vertices",
            lambda: mpc.PointsInPolygon(queries, pip_poly),
            lambda: [reference.PointInPolygon(point, pip_poly) for point in queries],
            5,
        ),
        (
            "Convex intersection, 1k x 1k vertices",
            intersection(mpc, clip_subject, clip_clip),
            intersection(reference, clip_subject, clip_clip),
            3,
        ),
        (
            "Miter offset, 100k vertices",
            offset(mpc, offset_poly, mpc.JT_MITER, 10_000),
            offset(reference, offset_poly, reference.JT_MITER, 10_000),
            5,
        ),
        (
            "Square offset, 100k vertices",
            offset(mpc, offset_poly, mpc.JT_SQUARE, 10_000),
            offset(reference, offset_poly, reference.JT_SQUARE, 10_000),
            5,
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| Case | mojo-pyclipper | pyclipper | Result |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs, repeat in cases:
        ours()
        theirs()
        mojo_time = time_best(ours, repeat)
        reference_time = time_best(theirs, repeat)
        ratio = reference_time / mojo_time
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{reference_time * 1e3:.3f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
