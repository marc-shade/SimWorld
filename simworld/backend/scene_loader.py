"""City scene loader for the SceneKit backend.

Loads citygen JSON output (buildings, roads, elements) into the SceneKit
renderer, and supports the progen_world.json format used by the Unreal Engine
pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simworld.backend.renderer import SceneKitRenderer

logger = logging.getLogger(__name__)

# Deterministic color palette for building types.
# Seeded per type name so colors are stable across runs.
_BUILDING_BASE_COLORS: dict[str, tuple[int, int, int]] = {}
_ELEMENT_BASE_COLORS: dict[str, tuple[int, int, int]] = {}


def _color_for_type(name: str, cache: dict, seed_offset: int = 0) -> tuple[int, int, int]:
    """Return a deterministic RGB color for a type name."""
    if name in cache:
        return cache[name]
    rng = random.Random(hash(name) ^ seed_offset)
    r = rng.randint(100, 230)
    g = rng.randint(100, 230)
    b = rng.randint(100, 230)
    cache[name] = (r, g, b)
    return (r, g, b)


# -- Prefab classification for progen_world nodes --

_ROAD_PREFIXES = ('BP_Road',)
_BUILDING_PREFIXES = ('BP_Building',)
_TREE_PREFIXES = ('BP_Tree',)
_ELEMENT_PREFIXES = (
    'BP_Box', 'BP_Can', 'BP_Cart', 'BP_Couch', 'BP_Hydrant',
    'BP_Rabbish', 'BP_RoadBlocker', 'BP_RoadCone', 'BP_Scooter',
    'BP_Soda', 'BP_Table', 'BP_Trash', 'BP_Tree',
)


def _classify_instance(instance_name: str) -> str:
    """Classify a progen_world instance_name into a category."""
    for prefix in _BUILDING_PREFIXES:
        if instance_name.startswith(prefix):
            return 'building'
    for prefix in _ROAD_PREFIXES:
        if instance_name.startswith(prefix):
            return 'road'
    for prefix in _TREE_PREFIXES:
        if instance_name.startswith(prefix):
            return 'tree'
    for prefix in _ELEMENT_PREFIXES:
        if instance_name.startswith(prefix):
            return 'element'
    return 'element'


class CitySceneLoader:
    """Loads citygen city data into a SceneKitRenderer.

    Supports two input formats:
    1. Separate JSON files (buildings.json, roads.json, elements.json)
       produced by the citygen pipeline.
    2. Unified progen_world.json produced by the procedural generation
       pipeline for Unreal Engine.
    """

    def __init__(self, renderer: SceneKitRenderer):
        self._renderer = renderer
        self._loaded_buildings: list[str] = []
        self._loaded_roads: list[str] = []
        self._loaded_elements: list[str] = []
        self._stats = {'buildings': 0, 'roads': 0, 'elements': 0}

    @property
    def stats(self) -> dict[str, int]:
        """Return counts of loaded objects by category."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_city(self, output_dir: str) -> dict[str, int]:
        """Load a city from the standard citygen output directory.

        Expects ``buildings.json``, ``roads.json``, and ``elements.json``
        inside *output_dir*. Missing files are silently skipped.

        Returns:
            Dict with counts per category.
        """
        buildings_path = os.path.join(output_dir, 'buildings.json')
        roads_path = os.path.join(output_dir, 'roads.json')
        elements_path = os.path.join(output_dir, 'elements.json')

        if os.path.isfile(roads_path):
            self._load_roads_file(roads_path)
        else:
            logger.warning("roads.json not found in %s", output_dir)

        if os.path.isfile(buildings_path):
            self._load_buildings_file(buildings_path)
        else:
            logger.warning("buildings.json not found in %s", output_dir)

        if os.path.isfile(elements_path):
            self._load_elements_file(elements_path)
        else:
            logger.warning("elements.json not found in %s", output_dir)

        logger.info(
            "City loaded: %d buildings, %d roads, %d elements",
            self._stats['buildings'], self._stats['roads'],
            self._stats['elements'],
        )
        return self.stats

    def load_progen_world(self, json_path: str) -> dict[str, int]:
        """Load a progen_world.json file.

        Each node has an ``id``, ``instance_name``, and ``properties``
        with ``location``, ``orientation``, and ``scale`` sub-dicts.

        Returns:
            Dict with counts per category.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        base_map = data.get('base_map', {})
        map_width = base_map.get('width', 1000)
        map_height = base_map.get('height', 1000)

        nodes = data.get('nodes', [])
        for node in nodes:
            node_id = node.get('id', '')
            instance_name = node.get('instance_name', '')
            props = node.get('properties', {})
            loc = props.get('location', {})
            orient = props.get('orientation', {})
            scale = props.get('scale', {})

            x = loc.get('x', 0)
            y = loc.get('y', 0)
            z = loc.get('z', 0)
            yaw = orient.get('yaw', 0)
            sx = scale.get('x', 1.0)
            sy = scale.get('y', 1.0)
            sz = scale.get('z', 1.0)

            category = _classify_instance(instance_name)

            if category == 'building':
                color = _color_for_type(instance_name, _BUILDING_BASE_COLORS)
                # Approximate building footprint from scale
                bw = 100 * sx
                bh = 100 * sy
                self._renderer.add_building(
                    name=node_id,
                    x=x - bw / 2,
                    y=y - bh / 2,
                    width=bw,
                    height=bh,
                    elevation=z,
                    rotation=yaw,
                    color=color,
                )
                self._loaded_buildings.append(node_id)
                self._stats['buildings'] += 1

            elif category == 'road':
                # Road nodes represent segments along an axis.
                # Approximate a segment of standard length along the yaw direction.
                import math
                seg_length = 200 * sx
                rad = math.radians(yaw)
                half_dx = seg_length / 2 * math.cos(rad)
                half_dy = seg_length / 2 * math.sin(rad)
                self._renderer.add_road(
                    name=node_id,
                    start_x=x - half_dx,
                    start_y=y - half_dy,
                    end_x=x + half_dx,
                    end_y=y + half_dy,
                    width=50 * sy,
                    is_highway=False,
                )
                self._loaded_roads.append(node_id)
                self._stats['roads'] += 1

            elif category == 'tree':
                self._renderer.add_element(
                    name=node_id,
                    x=x,
                    y=y,
                    width=20 * sx,
                    height=20 * sy,
                    element_type=instance_name,
                    color=(60, 140, 60),
                )
                self._loaded_elements.append(node_id)
                self._stats['elements'] += 1

            else:
                color = _color_for_type(instance_name, _ELEMENT_BASE_COLORS, seed_offset=7)
                ew = max(10 * sx, 1)
                eh = max(10 * sy, 1)
                self._renderer.add_element(
                    name=node_id,
                    x=x,
                    y=y,
                    width=ew,
                    height=eh,
                    element_type=instance_name,
                    color=color,
                )
                self._loaded_elements.append(node_id)
                self._stats['elements'] += 1

        logger.info(
            "Progen world loaded (%s): %d buildings, %d roads, %d elements",
            json_path, self._stats['buildings'], self._stats['roads'],
            self._stats['elements'],
        )
        return self.stats

    def clear(self):
        """Remove all loaded objects from the renderer."""
        for name in self._loaded_buildings:
            self._renderer.remove_object(name)
        for name in self._loaded_roads:
            self._renderer.remove_object(name)
        for name in self._loaded_elements:
            self._renderer.remove_object(name)
        self._loaded_buildings.clear()
        self._loaded_roads.clear()
        self._loaded_elements.clear()
        self._stats = {'buildings': 0, 'roads': 0, 'elements': 0}

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def _load_roads_file(self, path: str):
        """Load roads.json and add segments to the renderer."""
        with open(path, 'r') as f:
            data = json.load(f)

        roads = data.get('roads', [])
        for idx, road in enumerate(roads):
            start = road.get('start', {})
            end = road.get('end', {})
            is_highway = road.get('is_highway', False)

            name = f"road_{idx}"
            self._renderer.add_road(
                name=name,
                start_x=start.get('x', 0),
                start_y=start.get('y', 0),
                end_x=end.get('x', 0),
                end_y=end.get('y', 0),
                width=50,
                is_highway=is_highway,
            )
            self._loaded_roads.append(name)
            self._stats['roads'] += 1

    def _load_buildings_file(self, path: str):
        """Load buildings.json and add boxes to the renderer."""
        with open(path, 'r') as f:
            data = json.load(f)

        buildings = data.get('buildings', [])
        for idx, bld in enumerate(buildings):
            bounds = bld.get('bounds', {})
            btype = bld.get('type', 'unknown')
            rotation = bld.get('rotation', 0)

            x = bounds.get('x', 0)
            y = bounds.get('y', 0)
            w = bounds.get('width', 10)
            h = bounds.get('height', 10)

            color = _color_for_type(btype, _BUILDING_BASE_COLORS)
            name = f"building_{idx}"

            self._renderer.add_building(
                name=name,
                x=x,
                y=y,
                width=w,
                height=h,
                rotation=rotation,
                color=color,
            )
            self._loaded_buildings.append(name)
            self._stats['buildings'] += 1

    def _load_elements_file(self, path: str):
        """Load elements.json and add elements to the renderer."""
        with open(path, 'r') as f:
            data = json.load(f)

        elements = data.get('elements', [])
        for idx, elem in enumerate(elements):
            bounds = elem.get('bounds', {})
            etype = elem.get('type', '')
            center = elem.get('center', {})

            cx = center.get('x', bounds.get('x', 0))
            cy = center.get('y', bounds.get('y', 0))
            ew = bounds.get('width', 2)
            eh = bounds.get('height', 2)

            color = _color_for_type(etype, _ELEMENT_BASE_COLORS, seed_offset=7)
            name = f"element_{idx}"

            self._renderer.add_element(
                name=name,
                x=cx,
                y=cy,
                width=ew,
                height=eh,
                element_type=etype,
                color=color,
            )
            self._loaded_elements.append(name)
            self._stats['elements'] += 1
