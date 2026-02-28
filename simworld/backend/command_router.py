"""Command router for SimWorld macOS backend.

Routes UnrealCV text commands to handlers using regex pattern matching.
All commands from the original unrealcv.py client are supported.
"""
import json
import logging
import re
import struct
from typing import Callable, Union

import numpy as np
from io import BytesIO

from simworld.backend.world_state import WorldState

CommandHandler = Callable[[re.Match], Union[str, bytes]]

logger = logging.getLogger(__name__)


class CommandRouter:
    """Routes UnrealCV-style text commands to WorldState operations."""

    def __init__(self, world: WorldState):
        self.world = world
        self.renderer = None  # Set externally after construction
        self._routes: list[tuple[re.Pattern, CommandHandler]] = []
        self._register_all()

    def set_renderer(self, renderer):
        """Attach a renderer for camera image commands."""
        self.renderer = renderer

    def handle(self, command: str):
        """Dispatch a command string. Returns str or bytes."""
        command = command.strip()
        for pattern, handler in self._routes:
            m = pattern.match(command)
            if m:
                try:
                    return handler(m)
                except Exception:
                    logger.exception('Handler error for: %s', command[:120])
                    return 'error'
        logger.warning('Unhandled command: %s', command[:120])
        return 'ok'

    # ------------------------------------------------------------------
    # Internal: route registration
    # ------------------------------------------------------------------
    def _route(self, pattern: str, handler):
        self._routes.append((re.compile(pattern), handler))

    def _register_all(self):
        self._register_vget()
        self._register_vset()
        self._register_vbp()
        self._register_vrun()

    # ------------------------------------------------------------------
    # vget commands
    # ------------------------------------------------------------------
    def _register_vget(self):
        self._route(r'^vget /objects$', self._vget_objects)
        self._route(r'^vget /object/([\w.]+)/location$', self._vget_object_location)
        self._route(r'^vget /object/([\w.]+)/rotation$', self._vget_object_rotation)
        self._route(r'^vget /cameras$', self._vget_cameras)
        self._route(r'^vget /camera/(\d+)/location$', self._vget_camera_location)
        self._route(r'^vget /camera/(\d+)/rotation$', self._vget_camera_rotation)
        self._route(r'^vget /camera/(\d+)/fov$', self._vget_camera_fov)
        self._route(r'^vget /camera/(\d+)/size$', self._vget_camera_size)
        # Camera image: png/bmp/npy (binary) or filepath (text)
        self._route(r'^vget /camera/(\d+)/([\w]+) (png|bmp|npy)$', self._vget_camera_image_binary)
        self._route(r'^vget /camera/(\d+)/([\w]+) (.+)$', self._vget_camera_image_file)

    def _vget_objects(self, m):
        names = self.world.get_objects()
        return ' '.join(names)

    def _vget_object_location(self, m):
        name = m.group(1)
        loc = self.world.get_location(name)
        return f'{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f}'

    def _vget_object_rotation(self, m):
        name = m.group(1)
        rot = self.world.get_rotation(name)
        return f'{rot[0]:.2f} {rot[1]:.2f} {rot[2]:.2f}'

    def _vget_cameras(self, m):
        cam_ids = sorted(self.world.cameras.keys())
        return ' '.join(str(c) for c in cam_ids)

    def _vget_camera_location(self, m):
        cam_id = int(m.group(1))
        loc = self.world.get_camera_location(cam_id)
        return f'{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f}'

    def _vget_camera_rotation(self, m):
        cam_id = int(m.group(1))
        rot = self.world.get_camera_rotation(cam_id)
        return f'{rot[0]:.2f} {rot[1]:.2f} {rot[2]:.2f}'

    def _vget_camera_fov(self, m):
        cam_id = int(m.group(1))
        fov = self.world.get_camera_fov(cam_id)
        return f'{fov:.2f}'

    def _vget_camera_size(self, m):
        cam_id = int(m.group(1))
        w, h = self.world.get_camera_resolution(cam_id)
        return f'{w} {h}'

    def _vget_camera_image_binary(self, m):
        """Return binary image data (png, bmp, or npy)."""
        cam_id = int(m.group(1))
        viewmode = m.group(2)
        fmt = m.group(3)

        if self.renderer is not None:
            from simworld.backend.renderer import SceneKitRenderer
            arr = self.renderer.render_camera(cam_id, viewmode)
            if fmt == 'npy':
                return SceneKitRenderer.encode_npy(arr)
            elif fmt == 'bmp':
                return SceneKitRenderer.encode_bmp(arr)
            else:
                return SceneKitRenderer.encode_png(arr)

        # No renderer attached -- return a minimal valid placeholder
        return self._placeholder_image(cam_id, fmt)

    def _vget_camera_image_file(self, m):
        """Save image to filepath and return the path."""
        cam_id = int(m.group(1))
        viewmode = m.group(2)
        filepath = m.group(3).strip()

        if self.renderer is not None:
            from simworld.backend.renderer import SceneKitRenderer
            arr = self.renderer.render_camera(cam_id, viewmode)
            data = SceneKitRenderer.encode_png(arr)
        else:
            data = self._placeholder_image(cam_id, 'png')

        with open(filepath, 'wb') as f:
            f.write(data)
        return filepath

    def _placeholder_image(self, cam_id: int, fmt: str) -> bytes:
        """Generate a minimal placeholder image in the requested format."""
        cam_w, cam_h = self.world.get_camera_resolution(cam_id)
        if fmt == 'npy':
            arr = np.zeros((cam_h, cam_w), dtype=np.float32)
            buf = BytesIO()
            np.save(buf, arr)
            return buf.getvalue()
        elif fmt == 'bmp':
            return self._make_bmp(cam_w, cam_h)
        else:  # png
            return self._make_png(cam_w, cam_h)

    @staticmethod
    def _make_png(w: int, h: int) -> bytes:
        """Create a minimal valid PNG (black image)."""
        import zlib

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack('>I', len(data)) + c + crc

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
        ihdr = _chunk(b'IHDR', ihdr_data)

        # Raw image data: each row starts with filter byte 0, then w*3 zero bytes
        raw_rows = b''
        for _ in range(h):
            raw_rows += b'\x00' + b'\x00' * (w * 3)
        compressed = zlib.compress(raw_rows)
        idat = _chunk(b'IDAT', compressed)
        iend = _chunk(b'IEND', b'')
        return sig + ihdr + idat + iend

    @staticmethod
    def _make_bmp(w: int, h: int) -> bytes:
        """Create a minimal valid 24-bit BMP (black image)."""
        row_stride = ((24 * w + 31) // 32) * 4
        pixel_data_size = row_stride * h
        file_size = 54 + pixel_data_size

        header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
        info = struct.pack('<IiiHHIIiiII', 40, w, h, 1, 24, 0, pixel_data_size, 0, 0, 0, 0)
        pixels = b'\x00' * pixel_data_size
        return header + info + pixels

    # ------------------------------------------------------------------
    # vset commands
    # ------------------------------------------------------------------
    def _register_vset(self):
        self._route(r'^vset /objects/spawn ([\S]+) ([\w.]+)$', self._vset_spawn)
        self._route(r'^vset /objects/spawn_bp_asset ([\S]+) ([\w.]+)$', self._vset_spawn_bp)
        self._route(r'^vset /object/([\w.]+)/location ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vset_object_location)
        self._route(r'^vset /object/([\w.]+)/rotation ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vset_object_rotation)
        self._route(r'^vset /object/([\w.]+)/scale ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vset_object_scale)
        self._route(r'^vset /object/([\w.]+)/color (\d+) (\d+) (\d+)$', self._vset_object_color)
        self._route(r'^vset /object/([\w.]+)/physics (\w+)$', self._vset_object_physics)
        self._route(r'^vset /object/([\w.]+)/collision (\w+)$', self._vset_object_collision)
        self._route(r'^vset /object/([\w.]+)/object_mobility (\w+)$', self._vset_object_mobility)
        self._route(r'^vset /object/([\w.]+)/name ([\w.]+)$', self._vset_object_name)
        self._route(r'^vset /object/([\w.]+)/destroy$', self._vset_object_destroy)
        self._route(r'^vset /action/set_fixed_frame_rate (\d+)$', self._vset_fps)
        self._route(r'^vset /action/game/pause$', self._vset_pause)
        self._route(r'^vset /action/game/resume$', self._vset_resume)
        self._route(r'^vset /action/tick_intervel ([-\d.e+]+)$', self._vset_tick_interval)
        self._route(r'^vset /action/tick$', self._vset_tick)
        self._route(r'^vset /action/clean_garbage$', self._vset_clean_garbage)
        # Camera vset
        self._route(r'^vset /camera/(\d+)/location ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vset_camera_location)
        self._route(r'^vset /camera/(\d+)/rotation ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vset_camera_rotation)
        self._route(r'^vset /camera/(\d+)/fov ([-\d.e+]+)$', self._vset_camera_fov)
        self._route(r'^vset /camera/(\d+)/size (\d+) (\d+)$', self._vset_camera_size)

    def _vset_spawn(self, m):
        prefab, name = m.group(1), m.group(2)
        self.world.spawn(prefab, name)
        if self.renderer is not None:
            obj = self.world.objects[name]
            self.renderer.add_object(name, prefab, obj.position, obj.scale)
        return 'ok'

    def _vset_spawn_bp(self, m):
        prefab_path, name = m.group(1), m.group(2)
        self.world.spawn(prefab_path, name)
        if self.renderer is not None:
            obj = self.world.objects[name]
            self.renderer.add_object(name, prefab_path, obj.position, obj.scale)
        return 'ok'

    def _vset_object_location(self, m):
        name = m.group(1)
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        self.world.set_location(name, x, y, z)
        if self.renderer is not None and name in self.world.objects:
            obj = self.world.objects[name]
            self.renderer.update_object(name, position=obj.position, rotation=obj.rotation)
        return 'ok'

    def _vset_object_rotation(self, m):
        name = m.group(1)
        p, y, r = float(m.group(2)), float(m.group(3)), float(m.group(4))
        self.world.set_rotation(name, p, y, r)
        if self.renderer is not None and name in self.world.objects:
            obj = self.world.objects[name]
            self.renderer.update_object(name, position=obj.position, rotation=obj.rotation)
        return 'ok'

    def _vset_object_scale(self, m):
        name = m.group(1)
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        self.world.set_scale(name, x, y, z)
        return 'ok'

    def _vset_object_color(self, m):
        name = m.group(1)
        r, g, b = int(m.group(2)), int(m.group(3)), int(m.group(4))
        self.world.set_color(name, r, g, b)
        return 'ok'

    def _vset_object_physics(self, m):
        name = m.group(1)
        enabled = m.group(2).lower() in ('true', '1', 'yes')
        self.world.set_physics(name, enabled)
        return 'ok'

    def _vset_object_collision(self, m):
        name = m.group(1)
        enabled = m.group(2).lower() in ('true', '1', 'yes')
        self.world.set_collision(name, enabled)
        return 'ok'

    def _vset_object_mobility(self, m):
        name = m.group(1)
        movable = m.group(2).lower() in ('true', '1', 'yes')
        self.world.set_movable(name, movable)
        return 'ok'

    def _vset_object_name(self, m):
        old_name, new_name = m.group(1), m.group(2)
        self.world.rename_object(old_name, new_name)
        return 'ok'

    def _vset_object_destroy(self, m):
        name = m.group(1)
        self.world.destroy(name)
        if self.renderer is not None:
            self.renderer.remove_object(name)
        return 'ok'

    def _vset_fps(self, m):
        fps = int(m.group(1))
        self.world.set_fps(fps)
        return 'ok'

    def _vset_pause(self, m):
        self.world.set_paused(True)
        return 'ok'

    def _vset_resume(self, m):
        self.world.set_paused(False)
        return 'ok'

    def _vset_tick_interval(self, m):
        interval = float(m.group(1))
        self.world.set_tick_interval(interval)
        return 'ok'

    def _vset_tick(self, m):
        self.world.tick()
        return 'ok'

    def _vset_clean_garbage(self, m):
        return 'ok'

    def _vset_camera_location(self, m):
        cam_id = int(m.group(1))
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        self.world.set_camera_location(cam_id, x, y, z)
        if self.renderer is not None:
            self.renderer.update_camera(cam_id, position=np.array([x, y, z]))
        return 'ok'

    def _vset_camera_rotation(self, m):
        cam_id = int(m.group(1))
        p, y, r = float(m.group(2)), float(m.group(3)), float(m.group(4))
        self.world.set_camera_rotation(cam_id, p, y, r)
        if self.renderer is not None:
            self.renderer.update_camera(cam_id, rotation=np.array([p, y, r]))
        return 'ok'

    def _vset_camera_fov(self, m):
        cam_id = int(m.group(1))
        fov = float(m.group(2))
        self.world.set_camera_fov(cam_id, fov)
        return 'ok'

    def _vset_camera_size(self, m):
        cam_id = int(m.group(1))
        w, h = int(m.group(2)), int(m.group(3))
        self.world.set_camera_resolution(cam_id, w, h)
        return 'ok'

    # ------------------------------------------------------------------
    # vbp commands
    # ------------------------------------------------------------------
    def _register_vbp(self):
        # Humanoid movement
        self._route(r'^vbp ([\w.]+) MoveForward$', self._vbp_move_forward)
        self._route(r'^vbp ([\w.]+) TurnAround ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_turn_around)
        self._route(r'^vbp ([\w.]+) StopAgent$', self._vbp_stop_agent)
        self._route(r'^vbp ([\w.]+) StepForward ([-\d.e+]+) (\d+)$', self._vbp_step_forward)
        self._route(r'^vbp ([\w.]+) SetMaxSpeed ([-\d.e+]+)$', self._vbp_set_max_speed)
        self._route(r'^vbp ([\w.]+) SetSpeed ([-\d.e+]+)$', self._vbp_set_max_speed)

        # Humanoid actions
        self._route(r'^vbp ([\w.]+) SitDown$', self._vbp_sit_down)
        self._route(r'^vbp ([\w.]+) StandUp$', self._vbp_stand_up)
        self._route(r'^vbp ([\w.]+) PickUp ([\w.]+)$', self._vbp_pick_up)
        self._route(r'^vbp ([\w.]+) DropOff$', self._vbp_drop_off)
        self._route(r'^vbp ([\w.]+) EnterVehicle ([\w.]+)$', self._vbp_enter_vehicle)
        self._route(r'^vbp ([\w.]+) ExitVehicle ([\w.]+)$', self._vbp_exit_vehicle)
        self._route(r'^vbp ([\w.]+) GetOnScooter$', self._vbp_get_on_scooter)
        self._route(r'^vbp ([\w.]+) GetOffScooter$', self._vbp_get_off_scooter)

        # Humanoid social animations
        self._route(r'^vbp ([\w.]+) Discussion (\d+)$', self._vbp_discussion)
        self._route(r'^vbp ([\w.]+) Arguing (\d+)$', self._vbp_arguing)
        self._route(r'^vbp ([\w.]+) Listening$', self._vbp_listening)
        self._route(r'^vbp ([\w.]+) Wave2Dog$', self._vbp_wave2dog)
        self._route(r'^vbp ([\w.]+) Directing$', self._vbp_directing)
        self._route(r'^vbp ([\w.]+) StopAction$', self._vbp_stop_action)

        # Path / waypoints
        self._route(r'^vbp ([\w.]+) SetPath (.+)$', self._vbp_set_path)
        self._route(r'^vbp ([\w.]+) FollowPath$', self._vbp_follow_path)
        self._route(r'^vbp ([\w.]+) SetWaypoints (.+)$', self._vbp_set_waypoints)
        self._route(r'^vbp ([\w.]+) MovementSimulation$', self._vbp_movement_simulation)

        # Vehicle
        self._route(r'^vbp ([\w.]+) SetState ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_set_state)
        self._route(r'^vbp ([\w.]+) MakeUTurn$', self._vbp_make_u_turn)
        self._route(r'^vbp ([\w.]+) VSetState (.+)$', self._vbp_vset_state)
        self._route(r'^vbp ([\w.]+) PSetState (.+)$', self._vbp_pset_state)

        # Pedestrian
        self._route(r'^vbp ([\w.]+) StopPedestrian$', self._vbp_stop_pedestrian)
        self._route(r'^vbp ([\w.]+) Rotate_Angle ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_rotate_angle)
        self._route(r'^vbp ([\w.]+) SetPedestrianWalk$', self._vbp_set_pedestrian_walk)

        # Traffic light
        self._route(r'^vbp ([\w.]+) SwitchVehicleFrontGreen$', self._vbp_switch_vehicle_green)
        self._route(r'^vbp ([\w.]+) SetDuration ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_set_duration)
        self._route(r'^vbp ([\w.]+) GetInformation$', self._vbp_get_information)
        self._route(r'^vbp ([\w.]+) UpdateObjects$', self._vbp_update_objects)
        self._route(r'^vbp ([\w.]+) AddVehicleSignal ([\w.]+)$', self._vbp_add_vehicle_signal)
        self._route(r'^vbp ([\w.]+) AddPedSignal ([\w.]+)$', self._vbp_add_ped_signal)
        self._route(r'^vbp ([\w.]+) StartSimulation$', self._vbp_start_simulation)

        # Controller
        self._route(r'^vbp ([\w.]+) EnableController (\w+)$', self._vbp_enable_controller)
        self._route(r'^vbp ([\w.]+) GetCollisionNum$', self._vbp_get_collision_num)

        # Robot / dog
        self._route(r'^vbp ([\w.]+) Move_Speed ([-\d.e+]+) ([-\d.e+]+) (\d+)$',
                    self._vbp_move_speed)
        # Note: Rotate_Angle is already registered above (shared by pedestrian and robot)
        self._route(r'^vbp ([\w.]+) lookup$', self._vbp_lookup)
        self._route(r'^vbp ([\w.]+) lookdown$', self._vbp_lookdown)

        # Weather
        self._route(r'^vbp ([\w.]+) SetSunDirection ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_set_sun_direction)
        self._route(r'^vbp ([\w.]+) GetSunDirection$', self._vbp_get_sun_direction)
        self._route(r'^vbp ([\w.]+) SetSunIntensity ([-\d.e+]+)$', self._vbp_set_sun_intensity)
        self._route(r'^vbp ([\w.]+) GetSunIntensity$', self._vbp_get_sun_intensity)
        self._route(r'^vbp ([\w.]+) SetFog ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_set_fog)
        self._route(r'^vbp ([\w.]+) GetFog$', self._vbp_get_fog)
        self._route(r'^vbp ([\w.]+) SetAtmosphere ([-\d.e+]+) ([-\d.e+]+)$',
                    self._vbp_set_atmosphere)
        self._route(r'^vbp ([\w.]+) GetAtmosphere$', self._vbp_get_atmosphere)

    # --- Humanoid movement ---
    def _vbp_move_forward(self, m):
        name = m.group(1)
        self.world.humanoid_move_forward(name)
        return 'ok'

    def _vbp_turn_around(self, m):
        name = m.group(1)
        angle = float(m.group(3))
        clockwise = int(float(m.group(4)))
        self.world.humanoid_rotate(name, angle, clockwise)
        return 'ok'

    def _vbp_stop_agent(self, m):
        name = m.group(1)
        self.world.humanoid_stop(name)
        return 'ok'

    def _vbp_step_forward(self, m):
        name = m.group(1)
        duration = float(m.group(2))
        direction = int(m.group(3))
        self.world.humanoid_step_forward(name, duration, direction)
        return 'ok'

    def _vbp_set_max_speed(self, m):
        name = m.group(1)
        speed = float(m.group(2))
        self.world.humanoid_set_speed(name, speed)
        return 'ok'

    # --- Humanoid actions (JSON response) ---
    def _vbp_sit_down(self, m):
        name = m.group(1)
        result = self.world.humanoid_sit_down(name)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_stand_up(self, m):
        name = m.group(1)
        result = self.world.humanoid_stand_up(name)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_pick_up(self, m):
        humanoid = m.group(1)
        obj = m.group(2)
        result = self.world.humanoid_pick_up(humanoid, obj)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_drop_off(self, m):
        humanoid = m.group(1)
        result = self.world.humanoid_drop(humanoid)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_enter_vehicle(self, m):
        humanoid = m.group(1)
        vehicle = m.group(2)
        result = self.world.humanoid_enter_vehicle(humanoid, vehicle)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_exit_vehicle(self, m):
        humanoid = m.group(1)
        vehicle = m.group(2)
        result = self.world.humanoid_exit_vehicle(humanoid, vehicle)
        return json.dumps({'Success': 'true' if result else 'false'})

    def _vbp_get_on_scooter(self, m):
        return 'ok'

    def _vbp_get_off_scooter(self, m):
        return 'ok'

    # --- Social animations ---
    def _vbp_discussion(self, m):
        return 'ok'

    def _vbp_arguing(self, m):
        return 'ok'

    def _vbp_listening(self, m):
        return 'ok'

    def _vbp_wave2dog(self, m):
        return 'ok'

    def _vbp_directing(self, m):
        return 'ok'

    def _vbp_stop_action(self, m):
        name = m.group(1)
        self.world.humanoid_stop(name)
        return 'ok'

    # --- Path / waypoints ---
    def _vbp_set_path(self, m):
        name = m.group(1)
        path_str = m.group(2)
        self.world.humanoid_set_path(name, path_str)
        return 'ok'

    def _vbp_follow_path(self, m):
        name = m.group(1)
        self.world.humanoid_move_forward(name)
        return 'ok'

    def _vbp_set_waypoints(self, m):
        name = m.group(1)
        waypoints_str = m.group(2)
        self.world.humanoid_set_waypoints(name, waypoints_str)
        return 'ok'

    def _vbp_movement_simulation(self, m):
        name = m.group(1)
        self.world.humanoid_move_forward(name)
        return 'ok'

    # --- Vehicle ---
    def _vbp_set_state(self, m):
        name = m.group(1)
        throttle = float(m.group(2))
        brake = float(m.group(3))
        steering = float(m.group(4))
        self.world.vehicle_set_state(name, throttle, brake, steering)
        return 'ok'

    def _vbp_make_u_turn(self, m):
        name = m.group(1)
        self.world.vehicle_u_turn(name)
        return 'ok'

    def _vbp_vset_state(self, m):
        # Batch vehicle states -- store manager name for reference
        name = m.group(1)
        self.world.ue_manager_name = name
        return 'ok'

    def _vbp_pset_state(self, m):
        # Batch pedestrian states
        name = m.group(1)
        self.world.ue_manager_name = name
        return 'ok'

    # --- Pedestrian ---
    def _vbp_stop_pedestrian(self, m):
        name = m.group(1)
        self.world.pedestrian_stop(name)
        return 'ok'

    def _vbp_rotate_angle(self, m):
        name = m.group(1)
        angle = float(m.group(3))
        clockwise = int(float(m.group(4)))
        self.world.pedestrian_rotate(name, angle, clockwise)
        return 'ok'

    def _vbp_set_pedestrian_walk(self, m):
        return 'ok'

    # --- Traffic light ---
    def _vbp_switch_vehicle_green(self, m):
        return 'ok'

    def _vbp_set_duration(self, m):
        return 'ok'

    def _vbp_get_information(self, m):
        name = m.group(1)
        self.world.ue_manager_name = name
        objects = self.world.get_objects()
        info = {}
        for obj_name in objects:
            loc = self.world.get_location(obj_name)
            rot = self.world.get_rotation(obj_name)
            info[obj_name] = {
                'location': [float(loc[0]), float(loc[1]), float(loc[2])],
                'rotation': [float(rot[0]), float(rot[1]), float(rot[2])],
            }
        return json.dumps(info)

    def _vbp_update_objects(self, m):
        name = m.group(1)
        self.world.ue_manager_name = name
        return 'ok'

    def _vbp_add_vehicle_signal(self, m):
        return 'ok'

    def _vbp_add_ped_signal(self, m):
        return 'ok'

    def _vbp_start_simulation(self, m):
        return 'ok'

    # --- Controller ---
    def _vbp_enable_controller(self, m):
        name = m.group(1)
        enabled = m.group(2).lower() in ('true', '1', 'yes')
        with self.world.lock:
            if name in self.world.objects:
                self.world.objects[name].controller_enabled = enabled
        return 'ok'

    def _vbp_get_collision_num(self, m):
        name = m.group(1)
        counts = self.world.get_collision_num(name)
        return json.dumps(counts)

    # --- Robot / dog ---
    def _vbp_move_speed(self, m):
        name = m.group(1)
        speed = abs(float(m.group(2)))
        duration = float(m.group(3))
        direction = int(m.group(4))
        self.world.humanoid_set_speed(name, speed)
        self.world.humanoid_step_forward(name, duration, direction)
        return 'ok'

    def _vbp_lookup(self, m):
        name = m.group(1)
        with self.world.lock:
            if name in self.world.objects:
                self.world.objects[name].rotation[0] -= 15.0  # pitch up
        return 'ok'

    def _vbp_lookdown(self, m):
        name = m.group(1)
        with self.world.lock:
            if name in self.world.objects:
                self.world.objects[name].rotation[0] += 15.0  # pitch down
        return 'ok'

    # --- Weather ---
    def _vbp_set_sun_direction(self, m):
        return 'ok'

    def _vbp_get_sun_direction(self, m):
        return '0.0 0.0'

    def _vbp_set_sun_intensity(self, m):
        return 'ok'

    def _vbp_get_sun_intensity(self, m):
        return '1.0'

    def _vbp_set_fog(self, m):
        return 'ok'

    def _vbp_get_fog(self, m):
        return '0.0 0.0 0.0'

    def _vbp_set_atmosphere(self, m):
        return 'ok'

    def _vbp_get_atmosphere(self, m):
        return '1.0 1.0'

    # ------------------------------------------------------------------
    # vrun commands
    # ------------------------------------------------------------------
    def _register_vrun(self):
        self._route(r'^vrun setres (\d+)x(\d+)w$', self._vrun_setres)
        self._route(r'^vrun Editor\.AsyncSkinnedAssetCompilation (\d+)$',
                    self._vrun_async_skinned)

    def _vrun_setres(self, m):
        w, h = int(m.group(1)), int(m.group(2))
        self.world.set_resolution(w, h)
        return 'ok'

    def _vrun_async_skinned(self, m):
        return 'ok'
