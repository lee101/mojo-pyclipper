# mojo-pyclipper

`mojo-pyclipper` is a standalone Mojo port of the compute-heavy convex-polygon
subset of [`pyclipper`](https://pypi.org/project/pyclipper/). It keeps
pyclipper's public names and call signatures for the supported operations while
moving edge classification, clipping, point queries, area accumulation, and
offset construction into one compiled Mojo shared library.

This is a deliberately bounded port, not a wrapper around pyclipper. The
runtime package has no pyclipper dependency; pyclipper 1.4 is installed only as
the test and benchmark oracle.

## Coverage

Implemented:

- `Pyclipper.AddPath`, `AddPaths`, `Clear`, `GetBounds`, `Execute`, and
  `Execute2`
- all four clip types: `CT_INTERSECTION`, `CT_UNION`, `CT_DIFFERENCE`, and
  `CT_XOR`
- simple convex closed integer paths; an operand may contain multiple
  interior-disjoint paths
- `PFT_EVENODD`, `PFT_NONZERO`, `PFT_POSITIVE`, and `PFT_NEGATIVE` for the
  supported path sets
- `PyclipperOffset` with convex `ET_CLOSEDPOLYGON` paths, positive and negative
  deltas, and `JT_MITER`, `JT_SQUARE`, and `JT_ROUND`
- `Area`, `Orientation`, `PointInPolygon`, `ReversePath`, `ReversePaths`,
  `CleanPolygon`, `CleanPolygons`, scaling helpers, convex `SimplifyPolygon`,
  and the closed-path `PolyTree` helpers
- `PointsInPolygon`, a vectorized extension that classifies a whole `(n, 2)`
  integer array in one Mojo call

Not implemented:

- non-convex or self-intersecting input, holes, open paths, or overlapping
  contours within one Boolean operand
- Minkowski operations and simplification of self-intersections
- nonzero `Pyclipper` initialization options; `PreserveCollinear`,
  `StrictlySimple`, and hierarchical hole/open-path `PolyTree` semantics
- Clipper's full-range integer arithmetic in Boolean and offset operations;
  those operations reject coordinates whose magnitude exceeds `2**52`

Unsupported inputs raise `NotImplementedError` when added. Boolean union,
difference, and XOR can return a geometry-equivalent convex partition rather
than pyclipper's canonical merged contour. The paths cover the same region but
may retain internal shared edges. Round joins enforce the requested chord-error
tolerance but may sample an arc more densely than pyclipper.

## Install and run

Install the pinned Mojo and Python environment, then build the native library:

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-pyclipper.so`. `pixi` also puts `python/` on
`PYTHONPATH`.

## Usage

```python
import mojo_pyclipper as pyclipper

subject = [(0, 0), (100, 0), (100, 100), (0, 100)]
clip = [(50, -10), (80, -10), (80, 110), (50, 110)]

pc = pyclipper.Pyclipper()
pc.AddPath(subject, pyclipper.PT_SUBJECT, True)
pc.AddPath(clip, pyclipper.PT_CLIP, True)
intersection = pc.Execute(
    pyclipper.CT_INTERSECTION,
    pyclipper.PFT_NONZERO,
    pyclipper.PFT_NONZERO,
)
assert sum(pyclipper.Area(path) for path in intersection) == 3000

offsetter = pyclipper.PyclipperOffset(miter_limit=2.0)
offsetter.AddPath(subject, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
expanded = offsetter.Execute(10)
assert abs(pyclipper.Area(expanded[0])) == 14400
```

Run it from this checkout with `pixi run python example.py`, or paste it into
`pixi run python`.

## Benchmarks

Measured with `pixi run bench`; times are the best of 3 to 7 complete public API
calls after warm-up. They include Python input conversion, FFI, allocation,
rounding, and output conversion.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.

| Case | mojo-pyclipper | pyclipper | Result |
|---|---:|---:|---:|
| Area, 250k vertices | 37.375 ms | 15.061 ms | 2.48x slower |
| PointInPolygon, 25k points x 256 vertices | 28.022 ms | 190.285 ms | 6.79x faster |
| Convex intersection, 1k x 1k vertices | 0.399 ms | 0.333 ms | 1.20x slower |
| Miter offset, 100k vertices | 34.863 ms | 49.719 ms | 1.43x faster |
| Square offset, 100k vertices | 50.241 ms | 71.445 ms | 1.42x faster |

The vectorized point-query case wins because one FFI call replaces 25,000
Python-to-C calls. SIMD normal construction and cached native-ready input
buffers put both offset cases ahead of pyclipper. Area still spends most of its
public-call time validating and converting the Python list; its native SIMD
reduction takes less than a millisecond, and an already-contiguous NumPy input
crosses the FFI boundary without a copy. Convex intersection uses a linear
edge-merge kernel above 64 vertices per input, with Sutherland-Hodgman retained
as a conservative fallback for collinear-edge cases.

No GPU path is included. Area, point classification, and linear intersection
perform less than one floating-point operation per byte moved, while offset
construction is a branch-heavy single pass whose host/device transfers would
dominate its square roots and divisions. None clears the roughly two-flop-per-byte
threshold needed to justify a GPU allocation and launch. No CPU parallel path is
included either: this pinned Mojo nightly does not export its documented
`parallelize` primitive, and the remaining large independent normal pass is
already a small fraction of the faster-than-upstream offset calls.

## How it works

`src/capi.mojo` is the single compilation unit. Python stores paths as
C-contiguous NumPy `int64` arrays with interleaved `x, y` coordinates. Buffers
cross the C ABI as integer addresses; exported Mojo functions reconstruct
concrete mutable pointers and write into NumPy-owned output and scratch arrays.
No allocator or ownership crosses the ABI.

Large intersections merge the two counterclockwise edge streams in linear time,
normalizing winding by indexed traversal without copying either input. Small or
collinear-edge cases clip against oriented half-planes with the original
Sutherland-Hodgman kernel. A `Pyclipper` instance reuses its two NumPy-owned work
buffers. Difference keeps each rejected half-plane piece, and union/XOR compose
those convex pieces without overlap. Offsetting computes normalized outward edge
normals with SIMD, then constructs miter, square, or tolerance-bounded round
joins. The Python layer handles pyclipper-compatible object state, validation,
integer rounding, result cleanup, and convex partition assembly.
