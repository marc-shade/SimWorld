"""Tests for the SimWorld macOS backend.

Tests the protocol server, command router, world state, and rendering
pipeline without requiring SceneKit (renderer falls back to software mode).
"""
import asyncio
import struct
import threading
import time

import numpy as np
import pytest

from simworld.backend.command_router import CommandRouter
from simworld.backend.protocol_server import MAGIC, ProtocolServer, _build_frame
from simworld.backend.renderer import SceneKitRenderer
from simworld.backend.scene_loader import CitySceneLoader
from simworld.backend.world_state import CameraState, SimObject, WorldState


# ---------------------------------------------------------------
# WorldState tests
# ---------------------------------------------------------------

class TestWorldState:
    def test_spawn_and_get_objects(self):
        ws = WorldState()
        ws.spawn('/Game/Vehicle', 'car1')
        ws.spawn('/Game/Pedestrian', 'ped1')
        objects = ws.get_objects()
        assert 'car1' in objects
        assert 'ped1' in objects

    def test_set_get_location(self):
        ws = WorldState()
        ws.spawn('/Game/Thing', 'obj1')
        ws.set_location('obj1', 100.0, 200.0, 300.0)
        loc = ws.get_location('obj1')
        np.testing.assert_array_almost_equal(loc, [100.0, 200.0, 300.0])

    def test_set_get_rotation(self):
        ws = WorldState()
        ws.spawn('/Game/Thing', 'obj1')
        ws.set_rotation('obj1', 10.0, 20.0, 30.0)
        rot = ws.get_rotation('obj1')
        np.testing.assert_array_almost_equal(rot, [10.0, 20.0, 30.0])

    def test_destroy(self):
        ws = WorldState()
        ws.spawn('/Game/Thing', 'obj1')
        assert 'obj1' in ws.get_objects()
        ws.destroy('obj1')
        assert 'obj1' not in ws.get_objects()

    def test_rename(self):
        ws = WorldState()
        ws.spawn('/Game/Thing', 'old_name')
        ws.rename_object('old_name', 'new_name')
        assert 'new_name' in ws.get_objects()
        assert 'old_name' not in ws.get_objects()

    def test_humanoid_step_forward(self):
        ws = WorldState()
        ws.spawn('/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C', 'agent')
        ws.set_location('agent', 0, 0, 0)
        ws.humanoid_set_speed('agent', 100.0)
        ws.humanoid_step_forward('agent', 1.0, 0)
        loc = ws.get_location('agent')
        # Should have moved from origin
        assert np.linalg.norm(loc) > 0

    def test_humanoid_sit_stand(self):
        ws = WorldState()
        ws.spawn('/Game/Agent', 'h1')
        assert ws.humanoid_sit_down('h1') is True
        assert ws.humanoid_sit_down('h1') is False  # already sitting
        assert ws.humanoid_stand_up('h1') is True
        assert ws.humanoid_stand_up('h1') is False  # already standing

    def test_camera_defaults(self):
        ws = WorldState()
        assert 0 in ws.cameras
        fov = ws.get_camera_fov(0)
        assert fov == 90.0

    def test_tick(self):
        ws = WorldState()
        ws.spawn('/Game/Agent', 'a1')
        ws.humanoid_set_speed('a1', 500.0)
        ws.humanoid_move_forward('a1')
        initial_pos = ws.get_location('a1').copy()
        ws.tick()
        new_pos = ws.get_location('a1')
        # Object was moving, position should have changed
        assert not np.array_equal(initial_pos, new_pos)

    def test_vehicle_u_turn(self):
        ws = WorldState()
        ws.spawn('/Game/Vehicle', 'v1')
        ws.set_rotation('v1', 0, 0, 0)
        ws.vehicle_u_turn('v1')
        rot = ws.get_rotation('v1')
        assert abs(rot[1] - 180.0) < 0.01


# ---------------------------------------------------------------
# CommandRouter tests
# ---------------------------------------------------------------

class TestCommandRouter:
    def setup_method(self):
        self.world = WorldState()
        self.router = CommandRouter(self.world)

    def test_vget_objects_empty(self):
        result = self.router.handle('vget /objects')
        assert result == ''  # no objects

    def test_spawn_and_list(self):
        self.router.handle('vset /objects/spawn /Game/Car car1')
        result = self.router.handle('vget /objects')
        assert 'car1' in result

    def test_set_get_location(self):
        self.router.handle('vset /objects/spawn /Game/Thing obj1')
        self.router.handle('vset /object/obj1/location 100 200 300')
        result = self.router.handle('vget /object/obj1/location')
        parts = result.split()
        assert len(parts) == 3
        assert abs(float(parts[0]) - 100) < 0.1

    def test_set_get_rotation(self):
        self.router.handle('vset /objects/spawn /Game/Thing obj1')
        self.router.handle('vset /object/obj1/rotation 10 20 30')
        result = self.router.handle('vget /object/obj1/rotation')
        parts = result.split()
        assert abs(float(parts[1]) - 20) < 0.1

    def test_destroy(self):
        self.router.handle('vset /objects/spawn /Game/Thing obj1')
        self.router.handle('vset /object/obj1/destroy')
        result = self.router.handle('vget /objects')
        assert 'obj1' not in result

    def test_pause_resume(self):
        self.router.handle('vset /action/game/pause')
        assert self.world.paused is True
        self.router.handle('vset /action/game/resume')
        assert self.world.paused is False

    def test_tick(self):
        result = self.router.handle('vset /action/tick')
        assert result == 'ok'

    def test_vrun_setres(self):
        self.router.handle('vrun setres 1920x1080w')
        assert self.world.resolution == (1920, 1080)

    def test_vbp_move_forward(self):
        self.router.handle('vset /objects/spawn /Game/Agent agent1')
        result = self.router.handle('vbp agent1 MoveForward')
        assert result == 'ok'

    def test_vbp_sit_down(self):
        self.router.handle('vset /objects/spawn /Game/Agent agent1')
        result = self.router.handle('vbp agent1 SitDown')
        assert '"Success"' in result
        assert '"true"' in result

    def test_camera_image_placeholder(self):
        """When no renderer is set, should return valid PNG bytes."""
        result = self.router.handle('vget /camera/0/lit png')
        assert isinstance(result, bytes)
        assert result[:4] == b'\x89PNG'

    def test_camera_image_bmp(self):
        result = self.router.handle('vget /camera/0/lit bmp')
        assert isinstance(result, bytes)
        assert result[:2] == b'BM'

    def test_camera_image_npy(self):
        result = self.router.handle('vget /camera/0/depth npy')
        assert isinstance(result, bytes)
        # npy files start with \x93NUMPY
        assert b'NUMPY' in result[:10]

    def test_unhandled_command(self):
        result = self.router.handle('vget /nonexistent')
        assert result == 'ok'  # graceful fallback

    def test_camera_vset(self):
        self.router.handle('vset /camera/0/location 100 200 300')
        loc = self.world.get_camera_location(0)
        np.testing.assert_array_almost_equal(loc, [100, 200, 300])

    def test_camera_fov(self):
        self.router.handle('vset /camera/0/fov 60')
        assert abs(self.world.get_camera_fov(0) - 60.0) < 0.01


# ---------------------------------------------------------------
# Protocol wire format tests
# ---------------------------------------------------------------

class TestProtocolWireFormat:
    def test_build_frame(self):
        payload = b'hello'
        frame = _build_frame(payload)
        magic, size = struct.unpack('<II', frame[:8])
        assert magic == MAGIC
        assert size == len(payload)
        assert frame[8:] == payload

    def test_frame_roundtrip(self):
        payload = b'0:vget /objects'
        frame = _build_frame(payload)
        # Parse back
        magic, size = struct.unpack('<II', frame[:8])
        extracted = frame[8:8 + size]
        assert extracted == payload


# ---------------------------------------------------------------
# Protocol server integration test
# ---------------------------------------------------------------

class TestProtocolServer:
    @pytest.mark.asyncio
    async def test_connect_and_request(self):
        """Start server, connect, send vget /objects, verify response."""
        world = WorldState()
        world.spawn('/Game/Test', 'TestObj')
        router = CommandRouter(world)
        server = ProtocolServer(
            command_callback=router.handle,
            host='127.0.0.1',
            port=0,  # OS picks a free port
        )
        await server.start()
        # Get the actual port
        actual_port = server._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection('127.0.0.1', actual_port)

        # Read greeting
        header = await reader.readexactly(8)
        magic, size = struct.unpack('<II', header)
        assert magic == MAGIC
        greeting = await reader.readexactly(size)
        assert greeting == b'connected'

        # Send request: "0:vget /objects"
        request_payload = b'0:vget /objects'
        writer.write(_build_frame(request_payload))
        await writer.drain()

        # Read response
        header = await reader.readexactly(8)
        magic, size = struct.unpack('<II', header)
        assert magic == MAGIC
        response = await reader.readexactly(size)
        # Response format: "0:{object list}"
        assert response.startswith(b'0:')
        assert b'TestObj' in response

        writer.close()
        await writer.wait_closed()
        await server.stop()


# ---------------------------------------------------------------
# Renderer tests (fallback mode — no SceneKit required)
# ---------------------------------------------------------------

class TestRenderer:
    def test_fallback_lit(self):
        renderer = SceneKitRenderer(width=320, height=240)
        img = renderer.render_camera(0, 'lit')
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8

    def test_fallback_depth(self):
        renderer = SceneKitRenderer(width=320, height=240)
        depth = renderer.render_camera(0, 'depth')
        assert depth.shape == (240, 320)
        assert depth.dtype == np.float32

    def test_encode_png(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        png_bytes = SceneKitRenderer.encode_png(img)
        assert png_bytes[:4] == b'\x89PNG'

    def test_encode_bmp(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bmp_bytes = SceneKitRenderer.encode_bmp(img)
        assert bmp_bytes[:2] == b'BM'

    def test_encode_npy(self):
        arr = np.ones((10, 10), dtype=np.float32)
        npy_bytes = SceneKitRenderer.encode_npy(arr)
        from io import BytesIO
        loaded = np.load(BytesIO(npy_bytes))
        np.testing.assert_array_equal(arr, loaded)

    def test_add_remove_object(self):
        renderer = SceneKitRenderer(width=100, height=100)
        renderer.add_object('test_obj', '/Game/Thing',
                            np.array([0, 0, 0]), np.array([1, 1, 1]))
        assert 'test_obj' in renderer._nodes
        renderer.remove_object('test_obj')
        assert 'test_obj' not in renderer._nodes


# ---------------------------------------------------------------
# Scene loader tests
# ---------------------------------------------------------------

class TestSceneLoader:
    def test_load_city(self, tmp_path):
        """Test loading from JSON files."""
        import json

        # Create test roads.json
        roads = {'roads': [
            {'start': {'x': 0, 'y': 0}, 'end': {'x': 100, 'y': 0}, 'is_highway': False},
            {'start': {'x': 0, 'y': 0}, 'end': {'x': 0, 'y': 100}, 'is_highway': True},
        ]}
        (tmp_path / 'roads.json').write_text(json.dumps(roads))

        # Create test buildings.json
        buildings = {'buildings': [
            {'type': 'house', 'bounds': {'x': 10, 'y': 10, 'width': 20, 'height': 20, 'rotation': 0}, 'rotation': 0},
        ]}
        (tmp_path / 'buildings.json').write_text(json.dumps(buildings))

        # Create test elements.json
        elements = {'elements': [
            {'type': 'tree', 'bounds': {'x': 50, 'y': 50, 'width': 5, 'height': 5, 'rotation': 0},
             'center': {'x': 52.5, 'y': 52.5}, 'rotation': 0},
        ]}
        (tmp_path / 'elements.json').write_text(json.dumps(elements))

        renderer = SceneKitRenderer(width=100, height=100)
        loader = CitySceneLoader(renderer)
        stats = loader.load_city(str(tmp_path))

        assert stats['roads'] == 2
        assert stats['buildings'] == 1
        assert stats['elements'] == 1

    def test_clear(self):
        renderer = SceneKitRenderer(width=100, height=100)
        loader = CitySceneLoader(renderer)
        # Manually add some tracked objects
        renderer.add_building('b1', 0, 0, 10, 10)
        loader._loaded_buildings.append('b1')
        loader._stats['buildings'] = 1

        loader.clear()
        assert loader.stats['buildings'] == 0
        assert 'b1' not in renderer._nodes
