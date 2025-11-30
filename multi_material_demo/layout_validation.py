from __future__ import annotations

"""布局合法性校验：允许边/点接触，不允许越界或面积重叠。"""

from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover
    from .geometry_utils import (
        oriented_rectangle_with_notch,
        polygon_bounds,
        point_in_polygon as _point_in_polygon,
    )
except ImportError:  # pragma: no cover
    from geometry_utils import (  # type: ignore
        oriented_rectangle_with_notch,
        polygon_bounds,
        point_in_polygon as _point_in_polygon,
    )

Point = Tuple[float, float]
EPS = 1e-9


def _edges(poly: List[Point]) -> List[Tuple[Point, Point]]:
    return list(zip(poly, poly[1:] + poly[:1]))


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    )


def _segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point, include_endpoints: bool) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    if include_endpoints:
        if abs(o1) < EPS and _on_segment(p1, q1, p2):
            return True
        if abs(o2) < EPS and _on_segment(p1, q1, q2):
            return True
        if abs(o3) < EPS and _on_segment(p2, q2, p1):
            return True
        if abs(o4) < EPS and _on_segment(p2, q2, q1):
            return True
    return False


def _polygon_from_spec(spec: Dict[str, float]) -> List[Point]:
    length = float(spec.get("length", 0.0))
    width = float(spec.get("width", 0.0))
    notch_length = float(spec.get("notch_length", 0.0))
    notch_width = float(spec.get("notch_width", 0.0))
    if length <= 0 or width <= 0:
        raise ValueError("length/width must be positive")
    if notch_length < 0 or notch_width < 0 or notch_length > length or notch_width > width:
        raise ValueError("notch_length/notch_width must be within rectangle bounds")

    x = float(spec.get("x", 0.0))
    y = float(spec.get("y", 0.0))
    angle = float(spec.get("angle", spec.get("angle_deg", 0.0)))
    return oriented_rectangle_with_notch(x, y, length, width, angle, notch_length, notch_width)


def validate_layout_data(layout_data: Dict, allow_touching: bool = True) -> List[str]:
    """返回布局错误列表；允许边/点接触，不允许面积重叠/越界。"""

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
    obstacles = layout_data.get("obstacles", [])
    if obstacles is not None and not isinstance(obstacles, list):
        return ["obstacles 字段必须为数组（可选）"]

    errors: List[str] = []
    shapes: List[Tuple[str, List[Point]]] = []

    def add_shape(label: str, spec: Dict[str, float]):
        try:
            poly = _polygon_from_spec(spec)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            return
        (min_x, max_x), (min_y, max_y) = polygon_bounds(poly)
        if min_x < -EPS or max_x > length + EPS or min_y < -EPS or max_y > width + EPS:
            errors.append(f"{label}: 越界 [0,{length}]x[0,{width}]")
        shapes.append((label, poly))

    for idx, fu in enumerate(fus, start=1):
        label = str(fu.get("id") or f"FU-{idx}")
        add_shape(label, fu)

    for idx, obs in enumerate(obstacles or [], start=1):
        label = str(obs.get("id") or f"OBS-{idx}")
        add_shape(label, obs)

    # 仅边/点接触允许：我们只判定内部交叉或包含（边界不算重叠）
    for i in range(len(shapes)):
        name_a, poly_a = shapes[i]
        edges_a = _edges(poly_a)
        for j in range(i + 1, len(shapes)):
            name_b, poly_b = shapes[j]
            edges_b = _edges(poly_b)
            overlap = False

            # 边相交（不含端点触碰）
            for ea in edges_a:
                for eb in edges_b:
                    if _segments_intersect(ea[0], ea[1], eb[0], eb[1], include_endpoints=False):
                        overlap = True
                        break
                if overlap:
                    break

            if not overlap:
                # 顶点在对方内部（排除边界）
                if _point_in_polygon(poly_a[0], poly_b, include_boundary=False) or _point_in_polygon(poly_b[0], poly_a, include_boundary=False):
                    overlap = True

            if overlap:
                errors.append(f"{name_a} 与 {name_b} 存在重叠")

    return errors


__all__ = ["validate_layout_data"]
