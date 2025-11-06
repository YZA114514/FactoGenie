from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import simpy


@dataclass(frozen=True)
class SourceConfig:
    """Definition of a raw-material source."""

    node: str
    material: str
    batch_size: float
    interval: float


@dataclass(frozen=True)
class RouteConfig:
    """Definition of a transport route between two locations."""

    from_node: str
    to_node: str
    material: str
    batch_size: float
    travel_time: float = 0.0
    transporter_id: Optional[str] = None
    path_points: Optional[List[Tuple[float, float]]] = None




@dataclass(frozen=True)
class TransporterConfig:
    """Definition of transporter pool."""

    transporter_id: str
    count: int = 1



@dataclass(frozen=True)
class TransporterConfig:
    """Definition of transporter pool."""

    transporter_id: str
    count: int = 1
    speed: float = 1.0

@dataclass(frozen=True)
class AssemblyConfig:
    """Definition of an assembly station."""

    station: str
    inputs: Dict[str, float]
    output: str
    output_quantity: float = 1.0
    process_time: float = 1.0


class MaterialStore:
    """Track quantities of multiple materials and provide blocking get/consume operations."""

    def __init__(self, env: simpy.Environment, node_id: str, log_callback):
        self.env = env
        self.node_id = node_id
        self._levels: Counter[str] = Counter()
        self._change_event: simpy.Event = env.event()
        self._log = log_callback

    # ------------------------------------------------------------------
    def put(self, material: str, quantity: float) -> None:
        if quantity <= 0:
            return
        self._levels[material] += quantity
        self._log(
            "inventory_put",
            node=self.node_id,
            material=material,
            quantity=quantity,
            level=self._levels[material],
        )
        self._trigger()

    # ------------------------------------------------------------------
    def get(self, material: str, quantity: float):
        if quantity <= 0:
            return
        while self._levels[material] < quantity:
            event = self._change_event
            yield event
        self._levels[material] -= quantity
        self._log(
            "inventory_get",
            node=self.node_id,
            material=material,
            quantity=quantity,
            level=self._levels[material],
        )

    # ------------------------------------------------------------------
    def get_up_to(self, material: str, max_quantity: float):
        if max_quantity <= 0:
            return 0.0
        while self._levels[material] <= 0:
            event = self._change_event
            yield event
        quantity = min(self._levels[material], max_quantity)
        self._levels[material] -= quantity
        self._log(
            "inventory_get",
            node=self.node_id,
            material=material,
            quantity=quantity,
            level=self._levels[material],
        )
        return quantity

    # ------------------------------------------------------------------
    def consume(self, requirements: Dict[str, float]):
        if not requirements:
            return
        while True:
            missing = [
                m
                for m, qty in requirements.items()
                if self._levels[m] < qty
            ]
            if not missing:
                break
            event = self._change_event
            yield event
        for material, qty in requirements.items():
            self._levels[material] -= qty
            self._log(
                "inventory_get",
                node=self.node_id,
                material=material,
                quantity=qty,
                level=self._levels[material],
            )

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, float]:
        return dict(self._levels)

    # ------------------------------------------------------------------
    def _trigger(self) -> None:
        event = self._change_event
        self._change_event = self.env.event()
        if not event.triggered:
            event.succeed()


class AssemblySim:
    """Light-weight multi-material assembly simulator."""

    def __init__(
        self,
        sources: Iterable[SourceConfig],
        routes: Iterable[RouteConfig],
        assemblies: Iterable[AssemblyConfig],
        transporters: Iterable[TransporterConfig] = (),
        initial_inventory: Optional[Dict[str, Dict[str, float]]] = None,
        transporter_paths: Optional[
            Dict[str, Dict[Tuple[str, str], Dict[str, object]]]
        ] = None,
    ):
        self.env = simpy.Environment()
        self.events: List[Dict[str, object]] = []

        self.sources: List[SourceConfig] = list(sources)
        self.routes: List[RouteConfig] = list(routes)
        self.assemblies: List[AssemblyConfig] = list(assemblies)
        self.transporters: List[TransporterConfig] = list(transporters)
        self._transporter_paths: Dict[
            str, Dict[Tuple[str, str], Dict[str, object]]
        ] = transporter_paths or {}

        self._stores: Dict[str, MaterialStore] = {}
        self._init_stores()
        self._apply_initial_inventory(initial_inventory)
        self._init_transporters()
        self._init_processes()

    # ------------------------------------------------------------------
    def _log(self, event: str, **payload) -> None:
        record = {"time": self.env.now, "event": event, **payload}
        self.events.append(record)

    # ------------------------------------------------------------------
    def _init_stores(self) -> None:
        nodes = set()
        for cfg in self.sources:
            nodes.add(cfg.node)
        for cfg in self.routes:
            nodes.add(cfg.from_node)
            nodes.add(cfg.to_node)
        for cfg in self.assemblies:
            nodes.add(cfg.station)
        for node_id in nodes:
            self._stores[node_id] = MaterialStore(self.env, node_id, self._log)

    # ------------------------------------------------------------------
    def _init_processes(self) -> None:
        for src in self.sources:
            self.env.process(self._run_source(src))
        for route in self.routes:
            self.env.process(self._run_route(route))
        for assembly in self.assemblies:
            self.env.process(self._run_assembly(assembly))

    # ------------------------------------------------------------------
    def _init_transporters(self) -> None:
        self._transporter_resources: Dict[str, simpy.Resource] = {}
        self._transporter_stores: Dict[str, simpy.Store] = {}
        self._transporter_meta: Dict[str, Dict[str, object]] = {}
        for cfg in self.transporters:
            transporter_id = cfg.transporter_id
            capacity = max(1, int(cfg.count))
            # Legacy resource kept for backward compatibility (unused in new logic).
            self._transporter_resources[transporter_id] = simpy.Resource(self.env, capacity=capacity)
            initial_node = self._infer_initial_node(transporter_id)
            store = simpy.Store(self.env, capacity=capacity)
            for index in range(capacity):
                vehicle = {
                    "vehicle_id": f"{transporter_id}_{index + 1}",
                    "location": initial_node,
                }
                store.items.append(vehicle)
            self._transporter_stores[transporter_id] = store
            self._transporter_meta[transporter_id] = {"initial_node": initial_node}

    # ------------------------------------------------------------------
    def _infer_initial_node(self, transporter_id: str) -> Optional[str]:
        for route in self.routes:
            if route.transporter_id == transporter_id:
                return route.from_node
        path_map = self._transporter_paths.get(transporter_id, {})
        if path_map:
            first_key = next(iter(path_map.keys()))
            if isinstance(first_key, tuple) and first_key:
                return first_key[0]
        if self._stores:
            return next(iter(self._stores.keys()))
        return None

    # ------------------------------------------------------------------
    def _lookup_route_entry(
        self,
        transporter_id: Optional[str],
        from_node: Optional[str],
        to_node: Optional[str],
    ) -> Optional[Dict[str, object]]:
        if not transporter_id or not from_node or not to_node:
            return None
        path_map = self._transporter_paths.get(transporter_id)
        if not path_map:
            return None
        return path_map.get((from_node, to_node))

    # ------------------------------------------------------------------
    def _acquire_vehicle(self, transporter_id: str, target_node: Optional[str]):
        store = self._transporter_stores[transporter_id]
        vehicle = yield store.get()
        current_node = vehicle.get("location")
        if target_node and current_node and current_node != target_node:
            reposition_entry = self._lookup_route_entry(transporter_id, current_node, target_node)
            if reposition_entry:
                travel_time = float(reposition_entry.get("travel_time", 0.0))
                path = reposition_entry.get("path_points")
            else:
                travel_time = 0.0
                path = None
            if travel_time > 0.0:
                self._log(
                    "transport_reposition",
                    transporter=transporter_id,
                    vehicle=vehicle.get("vehicle_id"),
                    from_node=current_node,
                    to_node=target_node,
                    path=path,
                    travel_time=travel_time,
                )
                yield self.env.timeout(travel_time)
        vehicle["location"] = target_node
        return vehicle

    # ------------------------------------------------------------------
    def _release_vehicle(self, transporter_id: str, vehicle: Dict[str, object], location: str):
        vehicle["location"] = location
        store = self._transporter_stores[transporter_id]
        yield store.put(vehicle)

    def _apply_initial_inventory(
        self,
        initial_inventory: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        if not initial_inventory:
            return
        for node, materials in initial_inventory.items():
            store = self._stores.get(node)
            if store is None:
                continue
            for material, qty in materials.items():
                store.put(material, float(qty))

    # ------------------------------------------------------------------
    def store_snapshot(self, node: str) -> Dict[str, float]:
        return self._stores[node].snapshot()

    # ------------------------------------------------------------------
    def run(self, until: Optional[float] = None) -> None:
        self.env.run(until=until)

    # ------------------------------------------------------------------
    def _run_source(self, cfg: SourceConfig):
        store = self._stores[cfg.node]
        batch = cfg.batch_size
        while True:
            yield self.env.timeout(cfg.interval)
            store.put(cfg.material, batch)
            self._log(
                "source_produce",
                source=cfg.node,
                material=cfg.material,
                quantity=batch,
            )

    # ------------------------------------------------------------------
    def _run_route(self, cfg: RouteConfig):
        source_store = self._stores[cfg.from_node]
        dest_store = self._stores[cfg.to_node]
        batch = cfg.batch_size
        transporter_id = cfg.transporter_id
        store = self._transporter_stores.get(transporter_id) if transporter_id else None
        while True:
            quantity = yield from source_store.get_up_to(cfg.material, batch)
            if quantity <= 0:
                continue
            vehicle = None
            if store is not None:
                vehicle = yield from self._acquire_vehicle(transporter_id, cfg.from_node)
            route_entry = self._lookup_route_entry(transporter_id, cfg.from_node, cfg.to_node)
            path_points = cfg.path_points or (route_entry.get('path_points') if route_entry else None)
            travel_time = float(route_entry.get('travel_time', cfg.travel_time)) if route_entry else cfg.travel_time
            vehicle_id = vehicle.get("vehicle_id") if vehicle else None
            self._log(
                "transport_depart",
                transporter=transporter_id,
                vehicle=vehicle_id,
                from_node=cfg.from_node,
                to_node=cfg.to_node,
                material=cfg.material,
                quantity=quantity,
                path=path_points,
            )
            yield self.env.timeout(travel_time)
            dest_store.put(cfg.material, quantity)
            self._log(
                "transport_arrive",
                transporter=transporter_id,
                vehicle=vehicle_id,
                from_node=cfg.from_node,
                to_node=cfg.to_node,
                material=cfg.material,
                quantity=quantity,
                path=path_points,
            )
            if vehicle is not None:
                yield from self._release_vehicle(transporter_id, vehicle, cfg.to_node)

    # ------------------------------------------------------------------
    def _run_assembly(self, cfg: AssemblyConfig):
        store = self._stores[cfg.station]
        while True:
            yield from store.consume(cfg.inputs)
            self._log(
                "assembly_start",
                station=cfg.station,
                inputs=cfg.inputs,
            )
            yield self.env.timeout(cfg.process_time)
            store.put(cfg.output, cfg.output_quantity)
            self._log(
                "assembly_finish",
                station=cfg.station,
                output=cfg.output,
                quantity=cfg.output_quantity,
            )
