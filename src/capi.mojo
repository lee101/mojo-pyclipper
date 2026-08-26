"""Integer polygon kernels exported through a small C ABI."""

from std.math import acos, cos, sin, sqrt
from std.sys import simd_width_of as simdwidthof

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def signed_area_i64(points: IPtr, n: Int) -> Float64:
    var acc = 0.0
    if n < 2:
        return acc
    comptime W = simdwidthof[DType.float64]()
    comptime POINTS_PER_VECTOR = W // 2
    var signs = SIMD[DType.float64, W](1.0, -1.0, 1.0, -1.0)
    var i = 0
    while i + POINTS_PER_VECTOR < n:
        var current = SIMD[DType.float64, W](points.load[width=W](2 * i))
        var following = SIMD[DType.float64, W](points.load[width=W](2 * i + 2))
        acc += (current * following.shuffle[1, 0, 3, 2]() * signs).reduce_add()
        i += POINTS_PER_VECTOR
    while i + 1 < n:
        acc += Float64(points[2 * i]) * Float64(points[2 * i + 3])
        acc -= Float64(points[2 * i + 2]) * Float64(points[2 * i + 1])
        i += 1
    acc += Float64(points[2 * (n - 1)]) * Float64(points[1])
    acc -= Float64(points[0]) * Float64(points[2 * (n - 1) + 1])
    return 0.5 * acc


def signed_area_f64(points: FPtr, n: Int) -> Float64:
    var acc = 0.0
    if n < 2:
        return acc
    comptime W = simdwidthof[DType.float64]()
    comptime POINTS_PER_VECTOR = W // 2
    var signs = SIMD[DType.float64, W](1.0, -1.0, 1.0, -1.0)
    var i = 0
    while i + POINTS_PER_VECTOR < n:
        var current = points.load[width=W](2 * i)
        var following = points.load[width=W](2 * i + 2)
        acc += (current * following.shuffle[1, 0, 3, 2]() * signs).reduce_add()
        i += POINTS_PER_VECTOR
    while i + 1 < n:
        acc += points[2 * i] * points[2 * i + 3]
        acc -= points[2 * i + 2] * points[2 * i + 1]
        i += 1
    acc += points[2 * (n - 1)] * points[1]
    acc -= points[0] * points[2 * (n - 1) + 1]
    return 0.5 * acc


def point_in_polygon_i64(points: IPtr, n: Int, qx: Int64, qy: Int64) -> Int:
    var inside = False
    var j = n - 1
    for i in range(n):
        var xi = points[2 * i]
        var yi = points[2 * i + 1]
        var xj = points[2 * j]
        var yj = points[2 * j + 1]
        var dx = xi - xj
        var dy = yi - yj
        var qdx = qx - xj
        var qdy = qy - yj
        var cross = Float64(dx) * Float64(qdy) - Float64(dy) * Float64(qdx)
        if cross == 0.0 and qx >= min(xi, xj) and qx <= max(xi, xj) and qy >= min(yi, yj) and qy <= max(yi, yj):
            return -1
        if (yi > qy) != (yj > qy):
            var hit_x = Float64(xj) + Float64(qy - yj) * Float64(dx) / Float64(dy)
            if hit_x == Float64(qx):
                return -1
            if hit_x > Float64(qx):
                inside = not inside
        j = i
    return 1 if inside else 0


def is_convex_i64(points: IPtr, n: Int) -> Bool:
    if n < 3:
        return False
    var direction = 0
    for i in range(n):
        var j = 0 if i + 1 == n else i + 1
        var k = 0 if j + 1 == n else j + 1
        var ax = Float64(points[2 * j] - points[2 * i])
        var ay = Float64(points[2 * j + 1] - points[2 * i + 1])
        var bx = Float64(points[2 * k] - points[2 * j])
        var by = Float64(points[2 * k + 1] - points[2 * j + 1])
        var cross = ax * by - ay * bx
        if cross != 0.0:
            var current = 1 if cross > 0.0 else -1
            if direction != 0 and current != direction:
                return False
            direction = current
    return direction != 0


def line_value(
    x: Float64,
    y: Float64,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    orientation: Float64,
) -> Float64:
    return ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) * orientation


def emit_point(dst: FPtr, count: Int, x: Float64, y: Float64):
    dst[2 * count] = x
    dst[2 * count + 1] = y


def clip_halfplane(
    src: FPtr,
    n: Int,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    orientation: Float64,
    keep_inside: Bool,
    dst: FPtr,
    capacity: Int,
) -> Int:
    if n == 0:
        return 0
    var count = 0
    var px = src[2 * (n - 1)]
    var py = src[2 * (n - 1) + 1]
    var pv = line_value(px, py, ax, ay, bx, by, orientation)
    var pin = pv >= -1.0e-12 if keep_inside else pv <= 1.0e-12
    for i in range(n):
        var cx = src[2 * i]
        var cy = src[2 * i + 1]
        var cv = line_value(cx, cy, ax, ay, bx, by, orientation)
        var cin = cv >= -1.0e-12 if keep_inside else cv <= 1.0e-12
        if cin != pin:
            var denom = pv - cv
            if denom != 0.0:
                if count >= capacity:
                    return -1
                var t = pv / denom
                emit_point(dst, count, px + t * (cx - px), py + t * (cy - py))
                count += 1
        if cin:
            if count >= capacity:
                return -1
            emit_point(dst, count, cx, cy)
            count += 1
        px = cx
        py = cy
        pv = cv
        pin = cin
    return count


def copy_f64(src: FPtr, dst: FPtr, count: Int):
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= count:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < count:
        dst[i] = src[i]
        i += 1


def oriented_index(index: Int, n: Int, ccw: Bool) -> Int:
    return index if ccw else n - 1 - index


def emit_unique(dst: FPtr, count: Int, x: Float64, y: Float64, capacity: Int) -> Int:
    if count > 0 and dst[2 * (count - 1)] == x and dst[2 * (count - 1) + 1] == y:
        return count
    if count >= capacity:
        return -1
    emit_point(dst, count, x, y)
    return count + 1


def point_in_convex_f64(
    polygon: FPtr,
    n: Int,
    ccw: Bool,
    x: Float64,
    y: Float64,
) -> Bool:
    for edge in range(n):
        var following = 0 if edge + 1 == n else edge + 1
        var i = oriented_index(edge, n, ccw)
        var j = oriented_index(following, n, ccw)
        var cross = (polygon[2 * j] - polygon[2 * i]) * (y - polygon[2 * i + 1])
        cross -= (polygon[2 * j + 1] - polygon[2 * i + 1]) * (x - polygon[2 * i])
        if cross < -1.0e-12:
            return False
    return True


def intersect_convex_linear(
    subject: FPtr,
    subject_count: Int,
    cutter: FPtr,
    cutter_count: Int,
    target: FPtr,
    capacity: Int,
) -> Int:
    var subject_ccw = signed_area_f64(subject, subject_count) > 0.0
    var cutter_ccw = signed_area_f64(cutter, cutter_count) > 0.0
    var a = 0
    var b = 0
    var aa = 0
    var ba = 0
    var inside_flag = 0
    var first_crossing = True
    var count = 0
    while ((aa < subject_count) or (ba < cutter_count)) and aa < 2 * subject_count and ba < 2 * cutter_count:
        var a_previous = subject_count - 1 if a == 0 else a - 1
        var b_previous = cutter_count - 1 if b == 0 else b - 1
        var ai = oriented_index(a, subject_count, subject_ccw)
        var api = oriented_index(a_previous, subject_count, subject_ccw)
        var bi = oriented_index(b, cutter_count, cutter_ccw)
        var bpi = oriented_index(b_previous, cutter_count, cutter_ccw)
        var pax = subject[2 * api]
        var pay = subject[2 * api + 1]
        var ax = subject[2 * ai]
        var ay = subject[2 * ai + 1]
        var pbx = cutter[2 * bpi]
        var pby = cutter[2 * bpi + 1]
        var bx = cutter[2 * bi]
        var by = cutter[2 * bi + 1]
        var arx = ax - pax
        var ary = ay - pay
        var brx = bx - pbx
        var bry = by - pby
        var cross_value = arx * bry - ary * brx
        var cross = 1 if cross_value > 1.0e-12 else (-1 if cross_value < -1.0e-12 else 0)
        var ahb_value = brx * (ay - pby) - bry * (ax - pbx)
        var bha_value = arx * (by - pay) - ary * (bx - pax)
        var ahb = 1 if ahb_value > 1.0e-12 else (-1 if ahb_value < -1.0e-12 else 0)
        var bha = 1 if bha_value > 1.0e-12 else (-1 if bha_value < -1.0e-12 else 0)

        if cross == 0 and ahb == 0 and bha == 0:
            return -2

        if cross != 0:
            var dx = pbx - pax
            var dy = pby - pay
            var t = (dx * bry - dy * brx) / cross_value
            var u = (dx * ary - dy * arx) / cross_value
            if t >= -1.0e-12 and t <= 1.0 + 1.0e-12 and u >= -1.0e-12 and u <= 1.0 + 1.0e-12:
                var ix = pax + t * arx
                var iy = pay + t * ary
                if first_crossing:
                    aa = 0
                    ba = 0
                    first_crossing = False
                count = emit_unique(target, count, ix, iy, capacity)
                if count < 0:
                    return -1
                if ahb > 0:
                    inside_flag = 1
                elif bha > 0:
                    inside_flag = 2

        if cross == 0 and ahb < 0 and bha < 0:
            return 0
        if cross >= 0:
            if bha > 0:
                if inside_flag == 1:
                    count = emit_unique(target, count, ax, ay, capacity)
                    if count < 0:
                        return -1
                a = 0 if a + 1 == subject_count else a + 1
                aa += 1
            else:
                if inside_flag == 2:
                    count = emit_unique(target, count, bx, by, capacity)
                    if count < 0:
                        return -1
                b = 0 if b + 1 == cutter_count else b + 1
                ba += 1
        else:
            if ahb > 0:
                if inside_flag == 2:
                    count = emit_unique(target, count, bx, by, capacity)
                    if count < 0:
                        return -1
                b = 0 if b + 1 == cutter_count else b + 1
                ba += 1
            else:
                if inside_flag == 1:
                    count = emit_unique(target, count, ax, ay, capacity)
                    if count < 0:
                        return -1
                a = 0 if a + 1 == subject_count else a + 1
                aa += 1

    if inside_flag != 0:
        return count
    var subject_first = oriented_index(0, subject_count, subject_ccw)
    if point_in_convex_f64(
        cutter,
        cutter_count,
        cutter_ccw,
        subject[2 * subject_first],
        subject[2 * subject_first + 1],
    ):
        for i in range(subject_count):
            var index = oriented_index(i, subject_count, subject_ccw)
            emit_point(target, i, subject[2 * index], subject[2 * index + 1])
        return subject_count
    var cutter_first = oriented_index(0, cutter_count, cutter_ccw)
    if point_in_convex_f64(
        subject,
        subject_count,
        subject_ccw,
        cutter[2 * cutter_first],
        cutter[2 * cutter_first + 1],
    ):
        for i in range(cutter_count):
            var index = oriented_index(i, cutter_count, cutter_ccw)
            emit_point(target, i, cutter[2 * index], cutter[2 * index + 1])
        return cutter_count
    return 0


def intersect_convex(
    subject: FPtr,
    subject_count: Int,
    cutter: FPtr,
    cutter_count: Int,
    orientation: Float64,
    target: FPtr,
    scratch: FPtr,
    capacity: Int,
) -> Int:
    if subject_count < 3 or cutter_count < 3:
        return 0
    if subject_count >= 64 and cutter_count >= 64:
        var linear_count = intersect_convex_linear(
            subject, subject_count, cutter, cutter_count, target, capacity
        )
        if linear_count != -2:
            return linear_count
    var source = subject
    var destination = target
    var count = subject_count
    var in_target = False
    for i in range(cutter_count):
        var j = 0 if i + 1 == cutter_count else i + 1
        count = clip_halfplane(
            source,
            count,
            cutter[2 * i],
            cutter[2 * i + 1],
            cutter[2 * j],
            cutter[2 * j + 1],
            orientation,
            True,
            destination,
            capacity,
        )
        if count < 0:
            return -1
        if count < 3:
            return 0
        source = destination
        in_target = not in_target
        destination = scratch if in_target else target
    if not in_target:
        copy_f64(source, target, 2 * count)
    return count


def unit_outward_normal(points: FPtr, i: Int, n: Int, orientation: Float64, dst: FPtr):
    var j = 0 if i + 1 == n else i + 1
    var dx = points[2 * j] - points[2 * i]
    var dy = points[2 * j + 1] - points[2 * i + 1]
    var length = sqrt(dx * dx + dy * dy)
    dst[2 * i] = orientation * dy / length
    dst[2 * i + 1] = -orientation * dx / length


def outward_normals(points: FPtr, normals: FPtr, n: Int, orientation: Float64):
    comptime W = simdwidthof[DType.float64]()
    comptime POINTS_PER_VECTOR = W // 2
    var vector_count = (n - 1) // POINTS_PER_VECTOR

    @parameter
    def compute(chunk: Int):
        var offset = W * chunk
        var delta = points.load[width=W](offset + 2) - points.load[width=W](offset)
        var squared = delta * delta
        var lengths = sqrt(squared + squared.shuffle[1, 0, 3, 2]())
        var signs = SIMD[DType.float64, W](
            orientation, -orientation, orientation, -orientation
        )
        normals.store(
            offset, delta.shuffle[1, 0, 3, 2]() * signs / lengths
        )

    for chunk in range(vector_count):
        compute(chunk)
    for i in range(vector_count * POINTS_PER_VECTOR, n):
        unit_outward_normal(points, i, n, orientation, normals)


def offset_polygon(
    points: FPtr,
    normals: FPtr,
    n: Int,
    delta: Float64,
    join_type: Int,
    miter_limit: Float64,
    arc_tolerance: Float64,
    dst: FPtr,
    capacity: Int,
) -> Int:
    if n < 3:
        return 0
    var area = signed_area_f64(points, n)
    var orientation = 1.0 if area > 0.0 else -1.0
    outward_normals(points, normals, n, orientation)
    var count = 0
    for i in range(n):
        var previous = n - 1 if i == 0 else i - 1
        var n1x = normals[2 * previous]
        var n1y = normals[2 * previous + 1]
        var n2x = normals[2 * i]
        var n2y = normals[2 * i + 1]
        var dot = max(-1.0, min(1.0, n1x * n2x + n1y * n2y))
        var denom = 1.0 + dot
        var px = points[2 * i]
        var py = points[2 * i + 1]
        if delta <= 0.0 or join_type == 2:
            var ratio = 1.7976931348623157e308 if denom <= 1.0e-15 else sqrt(2.0 / denom)
            if delta <= 0.0 or ratio <= miter_limit:
                if denom > 1.0e-15 and count < capacity:
                    emit_point(dst, count, px + delta * (n1x + n2x) / denom, py + delta * (n1y + n2y) / denom)
                    count += 1
                continue
        if join_type == 1 and delta > 0.0:
            var angle = acos(dot)
            var tolerance = min(abs(delta) * 0.25, max(0.01, arc_tolerance))
            var step = 2.0 * acos(max(-1.0, min(1.0, 1.0 - tolerance / abs(delta))))
            if step <= 0.0:
                step = angle
            var intervals = max(1, Int(angle / step + 0.999999999))
            var signed_angle = orientation * angle
            for s in range(intervals + 1):
                if count >= capacity:
                    return -1
                var a = signed_angle * Float64(s) / Float64(intervals)
                var ca = cos(a)
                var sa = sin(a)
                var nx = n1x * ca - n1y * sa
                var ny = n1x * sa + n1y * ca
                emit_point(dst, count, px + delta * nx, py + delta * ny)
                count += 1
        else:
            var sin_half = sqrt(max(0.0, (1.0 - dot) * 0.5))
            var cos_half = sqrt(max(0.0, (1.0 + dot) * 0.5))
            var extension = 0.0 if cos_half == 0.0 else delta * sin_half / (1.0 + cos_half)
            var e1x = -orientation * n1y
            var e1y = orientation * n1x
            var e2x = -orientation * n2y
            var e2y = orientation * n2x
            if count + 1 >= capacity:
                return -1
            emit_point(dst, count, px + delta * n1x + extension * e1x, py + delta * n1y + extension * e1y)
            count += 1
            emit_point(dst, count, px + delta * n2x - extension * e2x, py + delta * n2y - extension * e2y)
            count += 1
    return count


@export("mpc_area")
def mpc_area(points_addr: Int, n: Int) abi("C") -> Float64:
    return signed_area_i64(iptr(points_addr), n)


@export("mpc_area_f64")
def mpc_area_f64(points_addr: Int, n: Int) abi("C") -> Float64:
    return signed_area_f64(fptr(points_addr), n)


@export("mpc_point_in_polygon")
def mpc_point_in_polygon(points_addr: Int, n: Int, x: Int64, y: Int64) abi("C") -> Int:
    return point_in_polygon_i64(iptr(points_addr), n, x, y)


@export("mpc_points_in_polygon")
def mpc_points_in_polygon(
    points_addr: Int, n: Int, queries_addr: Int, count: Int, dst_addr: Int
) abi("C"):
    var points = iptr(points_addr)
    var queries = iptr(queries_addr)
    var dst = iptr(dst_addr)
    for i in range(count):
        dst[i] = Int64(point_in_polygon_i64(points, n, queries[2 * i], queries[2 * i + 1]))


@export("mpc_is_convex")
def mpc_is_convex(points_addr: Int, n: Int) abi("C") -> Int:
    return 1 if is_convex_i64(iptr(points_addr), n) else 0


@export("mpc_clip_halfplane")
def mpc_clip_halfplane(
    src_addr: Int,
    n: Int,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    orientation: Float64,
    keep_inside: Int,
    dst_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    return clip_halfplane(
        fptr(src_addr), n, ax, ay, bx, by, orientation, keep_inside != 0, fptr(dst_addr), capacity
    )


@export("mpc_intersection")
def mpc_intersection(
    subject_addr: Int,
    subject_count: Int,
    cutter_addr: Int,
    cutter_count: Int,
    orientation: Float64,
    target_addr: Int,
    scratch_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    return intersect_convex(
        fptr(subject_addr),
        subject_count,
        fptr(cutter_addr),
        cutter_count,
        orientation,
        fptr(target_addr),
        fptr(scratch_addr),
        capacity,
    )


@export("mpc_offset_polygon")
def mpc_offset_polygon(
    points_addr: Int,
    normals_addr: Int,
    n: Int,
    delta: Float64,
    join_type: Int,
    miter_limit: Float64,
    arc_tolerance: Float64,
    dst_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    return offset_polygon(
        fptr(points_addr),
        fptr(normals_addr),
        n,
        delta,
        join_type,
        miter_limit,
        arc_tolerance,
        fptr(dst_addr),
        capacity,
    )
