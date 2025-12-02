from __future__ import annotations

"""Shared geometric helpers for layout rendering and path planning."""

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def _snap_trig(value: float, eps: float = 1e-12) -> float:
    for target in (-1.0, 0.0, 1.0):
        if abs(value - target) < eps:
            return target
    return value


def rotate_clockwise(point: Point, angle_deg: float, origin: Point = (0.0, 0.0)) -> Point:
    """Rotate ``point`` around ``origin`` by ``angle_deg`` clockwise."""

    ox, oy = origin
    px, py = point
    dx, dy = px - ox, py - oy
    rad = math.radians(angle_deg)
    cos_a = _snap_trig(math.cos(rad))
    sin_a = _snap_trig(math.sin(rad))
    rx = cos_a * dx + sin_a * dy
    ry = -sin_a * dx + cos_a * dy
    return (ox + rx, oy + ry)


def oriented_rectangle(x: float, y: float, length: float, width: float, angle_deg: float) -> List[Point]:
    """Return polygon vertices for a rectangle rotated around its lower-left corner."""

    base = [
        (0.0, 0.0),
        (length, 0.0),
        (length, width),
        (0.0, width),
    ]
    rotated = [rotate_clockwise(pt, angle_deg, origin=(0.0, 0.0)) for pt in base]
    return [(x + px, y + py) for px, py in rotated]


def oriented_rectangle_with_notch(
    x: float,
    y: float,
    length: float,
    width: float,
    angle_deg: float,
    notch_length: float = 0.0,
    notch_width: float = 0.0,
) -> List[Point]:
    """Rectangle polygon with optional top-right notch, rotated around lower-left corner."""

    notch_length = float(notch_length or 0.0)
    notch_width = float(notch_width or 0.0)
    if notch_length < 0 or notch_width < 0 or notch_length > length or notch_width > width:
        raise ValueError("notch_length/width must be within rectangle bounds")
    if notch_length == 0.0 or notch_width == 0.0:
        return oriented_rectangle(x, y, length, width, angle_deg)

    base = [
        (0.0, 0.0),
        (length - notch_length, 0.0),
        (length - notch_length, notch_width),
        (length, notch_width),
        (length, width),
        (0.0, width),
    ]
    rotated = [rotate_clockwise(pt, angle_deg, origin=(0.0, 0.0)) for pt in base]
    return [(x + px, y + py) for px, py in rotated]


def rotated_center(x: float, y: float, length: float, width: float, angle_deg: float) -> Point:
    """Return the world-space centre of a rectangle rotated around its lower-left corner."""

    local_center = (length / 2.0, width / 2.0)
    cx, cy = rotate_clockwise(local_center, angle_deg, origin=(0.0, 0.0))
    return (x + cx, y + cy)


def polygon_bounds(points: Sequence[Point]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return ((min_x, max_x), (min_y, max_y)) for a polygon."""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs)), (min(ys), max(ys))


def polygon_centroid(points: Sequence[Point]) -> Point:
    """Return centroid of a polygon (non-self-intersecting)."""

    pts = list(points)
    if len(pts) < 3:
        xs = [p[0] for p in pts] or [0.0]
        ys = [p[1] for p in pts] or [0.0]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-12:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    area *= 0.5
    return (cx / (6 * area), cy / (6 * area))


def point_in_polygon(point: Point, polygon: Sequence[Point], include_boundary: bool = True) -> bool:
    """Even-odd winding test with optional boundary inclusion."""

    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        # Boundary check
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9:
            if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return include_boundary
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1)
        if intersects:
            inside = not inside
    return inside


def min_distance_to_edges(point: Point, polygon: Sequence[Point]) -> float:
    """Return the minimum Euclidean distance from point to polygon edges."""

    if len(polygon) < 2:
        return float("inf")
    px, py = point
    min_dist = float("inf")
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            dist = math.hypot(px - closest_x, py - closest_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist


__all__ = [
    "Point",
    "rotate_clockwise",
    "oriented_rectangle",
    "oriented_rectangle_with_notch",
    "rotated_center",
    "polygon_bounds",
    "point_in_polygon",
    "min_distance_to_edges",
    "polygon_centroid",
]
