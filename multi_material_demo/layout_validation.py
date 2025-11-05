from __future__ import annotations

"""布局合法性检测工具，参考早期 workplace.py 实现。"""

from math import cos, sin, radians
from typing import Dict, Iterable, List, Optional, Tuple

Point = Tuple[float, float]


def _rotate_clockwise(point: Point, angle_deg: float, origin: Point = (0.0, 0.0)) -> Point:
    ox, oy = origin
    px, py = point
    dx, dy = px - ox, py - oy
    rad = radians(angle_deg)
    c, s = cos(rad), sin(rad)
    rx = c * dx + s * dy
    ry = -s * dx + c * dy
    return (rx + ox, ry + oy)


def _transform_polygon(points: Iterable[Point], rotate_deg: float, translate: Point) -> List[Point]:
    rotated = [_rotate_clockwise(p, rotate_deg, origin=(0.0, 0.0)) for p in points]
    tx, ty = translate
    return [(px + tx, py + ty) for px, py in rotated]


def _edges(poly: List[Point]) -> List[Tuple[Point, Point]]:
    return list(zip(poly, poly[1:] + poly[:1]))


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
    )


def _segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point, consider_endpoints: bool) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    if abs(o1) < 1e-12 and _on_segment(p1, q1, p2):
        return consider_endpoints
    if abs(o2) < 1e-12 and _on_segment(p1, q1, q2):
        return consider_endpoints
    if abs(o3) < 1e-12 and _on_segment(p2, q2, p1):
        return consider_endpoints
    if abs(o4) < 1e-12 and _on_segment(p2, q2, q1):
        return consider_endpoints

    return False


def _point_in_polygon(pt: Point, poly: List[Point], boundary_is_inside: bool) -> bool:
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in _edges(poly):
        o = _orientation((x1, y1), (x2, y2), (x, y))
        if abs(o) < 1e-12 and _on_segment((x1, y1), (x2, y2), (x, y)):
            return boundary_is_inside
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            if xinters > x:
                inside = not inside
    return inside


def _fu_polygon_local(fu: Dict[str, float]) -> List[Point]:
    length = float(fu.get("length", 0.0))
    width = float(fu.get("width", 0.0))
    notch_length = float(fu.get("notch_length", 0.0))
    notch_width = float(fu.get("notch_width", 0.0))
    if length <= 0 or width <= 0:
        raise ValueError("FU 的长宽必须为正数")
    if not (0 <= notch_length <= length) or not (0 <= notch_width <= width):
        raise ValueError("缺角尺寸必须在本体范围内")

    if notch_length == 0 or notch_width == 0:
        return [(0.0, 0.0), (length, 0.0), (length, width), (0.0, width)]
    return [
        (0.0, 0.0),
        (length - notch_length, 0.0),
        (length - notch_length, notch_width),
        (length, notch_width),
        (length, width),
        (0.0, width),
    ]


def _fu_polygon_world(fu: Dict[str, float], placement: Dict[str, float]) -> List[Point]:
    angle = float(placement.get("angle", placement.get("angle_deg", 0.0)))
    poly_local = _fu_polygon_local(fu)
    return _transform_polygon(poly_local, rotate_deg=angle, translate=(float(placement.get("x", 0.0)), float(placement.get("y", 0.0))))


def validate_layout_data(layout_data: Dict, allow_touching: bool = True) -> List[str]:
    """对布局 JSON 进行几何合法性校验。返回错误消息列表。"""

    if not layout_data:
        return ["布局数据为空"]

    factory = layout_data.get("factory", {})
    length = float(factory.get("length", 0.0))
    width = float(factory.get("width", 0.0))
    if length <= 0 or width <= 0:
        return ["factory.length 和 factory.width 必须为正数"]

    fus = layout_data.get("fus", [])
    if not isinstance(fus, list):
        return ["fus 字段必须为数组"]

    errors: List[str] = []
    polygons: List[List[Point]] = []
    names: List[str] = []

    for idx, fu_spec in enumerate(fus, start=1):
        label = str(fu_spec.get("id") or f"FU-{idx}")
        angle = float(fu_spec.get("angle", fu_spec.get("angle_deg", 0.0)))
        if abs(angle % 90.0) > 1e-6 and abs((angle % 90.0) - 90.0) > 1e-6:
            errors.append(f"{label}: angle {angle}° 非 90° 倍数")

        try:
            poly = _fu_polygon_world(fu_spec, fu_spec)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        polygons.append(poly)
        names.append(label)

        for (x, y) in poly:
            if not (-1e-9 <= x <= length + 1e-9 and -1e-9 <= y <= width + 1e-9):
                errors.append(f"{label}: 顶点({x:.3f},{y:.3f}) 超出工厂边界 [0,{length}]x[0,{width}]")
                break

    consider_endpoints = not allow_touching
    boundary_is_inside = not allow_touching
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly_a = polygons[i]
            poly_b = polygons[j]
            inter = False
            for edge_a in _edges(poly_a):
                for edge_b in _edges(poly_b):
                    if _segments_intersect(edge_a[0], edge_a[1], edge_b[0], edge_b[1], consider_endpoints):
                        inter = True
                        break
                if inter:
                    break

            if inter:
                errors.append(f"{names[i]} 与 {names[j]} 存在几何交叉")
                continue

            if _point_in_polygon(poly_a[0], poly_b, boundary_is_inside) or _point_in_polygon(poly_b[0], poly_a, boundary_is_inside):
                errors.append(f"{names[i]} 与 {names[j]} 存在包含重叠")

    return errors


__all__ = ["validate_layout_data"]

