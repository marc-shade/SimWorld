"""World state manager for SimWorld macOS backend."""
import numpy as np
from dataclasses import dataclass, field
from threading import Lock
import math
from typing import Optional


@dataclass
class SimObject:
    name: str
    prefab: str
    position: np.ndarray       # [x, y, z]
    rotation: np.ndarray       # [pitch, yaw, roll]
    scale: np.ndarray          # [x, y, z]
    color: tuple = (255, 255, 255)
    has_physics: bool = False
    has_collision: bool = True
    is_movable: bool = True
    object_type: str = 'static'   # 'humanoid', 'vehicle', 'pedestrian', 'static', 'robot', 'scooter'
    speed: float = 200.0
    is_moving: bool = False
    waypoints: list = field(default_factory=list)
    path: list = field(default_factory=list)
    attached_object: Optional[str] = None    # for pickup/drop
    vehicle: Optional[str] = None            # for enter/exit vehicle
    is_sitting: bool = False
    controller_enabled: bool = True


@dataclass
class CameraState:
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [pitch, yaw, roll]
    fov: float = 90.0
    width: int = 1280
    height: int = 720


class WorldState:
    """Tracks all objects, cameras, and simulation state."""

    def __init__(self):
        self.objects: dict[str, SimObject] = {}
        self.cameras: dict[int, CameraState] = {
            0: CameraState(
                position=np.array([0.0, 0.0, 500.0]),
                rotation=np.array([0.0, 0.0, 0.0]),
            )
        }
        self.paused: bool = False
        self.tick_interval: float = 0.05
        self.fps: int = 30
        self.resolution: tuple = (1280, 720)
        self.lock = Lock()
        self._collision_counts: dict = {}  # name -> {Human:0, Object:0, Building:0, Vehicle:0}
        self.ue_manager_name: Optional[str] = None

    # Object management
    def spawn(self, prefab: str, name: str) -> SimObject:
        """Spawn a new object at origin."""
        with self.lock:
            obj = SimObject(
                name=name, prefab=prefab,
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.array([0.0, 0.0, 0.0]),
                scale=np.array([1.0, 1.0, 1.0]),
            )
            # Determine object type from prefab
            prefab_lower = prefab.lower()
            if 'pedestrian' in prefab_lower or 'user_agent' in prefab_lower:
                obj.object_type = 'humanoid'
            elif 'vehicle' in prefab_lower:
                obj.object_type = 'vehicle'
            elif 'scooter' in prefab_lower:
                obj.object_type = 'scooter'
            elif 'robot' in prefab_lower or 'dog' in prefab_lower:
                obj.object_type = 'robot'
            self.objects[name] = obj
            self._collision_counts[name] = {
                'HumanCollision': 0, 'ObjectCollision': 0,
                'BuildingCollision': 0, 'VehicleCollision': 0,
            }
            return obj

    def destroy(self, name: str):
        with self.lock:
            self.objects.pop(name, None)
            self._collision_counts.pop(name, None)

    def get_objects(self) -> list[str]:
        with self.lock:
            return list(self.objects.keys())

    def set_location(self, name: str, x: float, y: float, z: float):
        with self.lock:
            if name in self.objects:
                self.objects[name].position = np.array([x, y, z])

    def get_location(self, name: str) -> np.ndarray:
        with self.lock:
            if name in self.objects:
                return self.objects[name].position.copy()
            return np.array([0.0, 0.0, 0.0])

    def set_rotation(self, name: str, pitch: float, yaw: float, roll: float):
        with self.lock:
            if name in self.objects:
                self.objects[name].rotation = np.array([pitch, yaw, roll])

    def get_rotation(self, name: str) -> np.ndarray:
        with self.lock:
            if name in self.objects:
                return self.objects[name].rotation.copy()
            return np.array([0.0, 0.0, 0.0])

    def set_scale(self, name: str, x: float, y: float, z: float):
        with self.lock:
            if name in self.objects:
                self.objects[name].scale = np.array([x, y, z])

    def set_color(self, name: str, r: int, g: int, b: int):
        with self.lock:
            if name in self.objects:
                self.objects[name].color = (r, g, b)

    def set_physics(self, name: str, enabled: bool):
        with self.lock:
            if name in self.objects:
                self.objects[name].has_physics = enabled

    def set_collision(self, name: str, enabled: bool):
        with self.lock:
            if name in self.objects:
                self.objects[name].has_collision = enabled

    def set_movable(self, name: str, movable: bool):
        with self.lock:
            if name in self.objects:
                self.objects[name].is_movable = movable

    def rename_object(self, old_name: str, new_name: str):
        with self.lock:
            if old_name in self.objects:
                obj = self.objects.pop(old_name)
                obj.name = new_name
                self.objects[new_name] = obj

    def get_collision_num(self, name: str) -> dict:
        with self.lock:
            return self._collision_counts.get(name, {
                'HumanCollision': 0, 'ObjectCollision': 0,
                'BuildingCollision': 0, 'VehicleCollision': 0,
            })

    # Simulation control
    def set_paused(self, paused: bool):
        self.paused = paused

    def set_tick_interval(self, interval: float):
        self.tick_interval = interval

    def set_fps(self, fps: int):
        self.fps = fps

    def set_resolution(self, width: int, height: int):
        self.resolution = (width, height)

    def tick(self):
        """Advance simulation by one tick."""
        with self.lock:
            dt = self.tick_interval
            for obj in self.objects.values():
                if obj.is_moving and obj.is_movable:
                    # Move in facing direction
                    yaw_rad = math.radians(obj.rotation[1])  # yaw
                    dx = math.cos(yaw_rad) * obj.speed * dt
                    dy = math.sin(yaw_rad) * obj.speed * dt
                    obj.position[0] += dx
                    obj.position[1] += dy

    # Humanoid actions
    def humanoid_move_forward(self, name: str):
        with self.lock:
            if name in self.objects:
                self.objects[name].is_moving = True

    def humanoid_stop(self, name: str):
        with self.lock:
            if name in self.objects:
                self.objects[name].is_moving = False

    def humanoid_step_forward(self, name: str, duration: float, direction: int = 0):
        """Step forward for a duration. direction: 0=forward, 1=backward, 2=left, 3=right."""
        with self.lock:
            if name not in self.objects:
                return
            obj = self.objects[name]
            yaw_rad = math.radians(obj.rotation[1])
            speed = obj.speed
            if direction == 1:
                speed = -speed
            elif direction == 2:
                yaw_rad -= math.pi / 2
            elif direction == 3:
                yaw_rad += math.pi / 2
            obj.position[0] += math.cos(yaw_rad) * speed * duration
            obj.position[1] += math.sin(yaw_rad) * speed * duration

    def humanoid_rotate(self, name: str, angle: float, clockwise: int):
        with self.lock:
            if name in self.objects:
                self.objects[name].rotation[1] += angle  # yaw

    def humanoid_set_speed(self, name: str, speed: float):
        with self.lock:
            if name in self.objects:
                self.objects[name].speed = speed

    def humanoid_sit_down(self, name: str) -> bool:
        with self.lock:
            if name in self.objects and not self.objects[name].is_sitting:
                self.objects[name].is_sitting = True
                self.objects[name].is_moving = False
                return True
            return False

    def humanoid_stand_up(self, name: str) -> bool:
        with self.lock:
            if name in self.objects and self.objects[name].is_sitting:
                self.objects[name].is_sitting = False
                return True
            return False

    def humanoid_pick_up(self, humanoid_name: str, object_name: str) -> bool:
        with self.lock:
            if humanoid_name in self.objects and object_name in self.objects:
                self.objects[humanoid_name].attached_object = object_name
                return True
            return False

    def humanoid_drop(self, humanoid_name: str) -> bool:
        with self.lock:
            if humanoid_name in self.objects and self.objects[humanoid_name].attached_object:
                self.objects[humanoid_name].attached_object = None
                return True
            return False

    def humanoid_enter_vehicle(self, humanoid_name: str, vehicle_name: str) -> bool:
        with self.lock:
            if humanoid_name in self.objects and vehicle_name in self.objects:
                self.objects[humanoid_name].vehicle = vehicle_name
                return True
            return False

    def humanoid_exit_vehicle(self, humanoid_name: str, vehicle_name: str) -> bool:
        with self.lock:
            if humanoid_name in self.objects and self.objects[humanoid_name].vehicle == vehicle_name:
                self.objects[humanoid_name].vehicle = None
                return True
            return False

    def humanoid_set_path(self, name: str, path_str: str):
        """Set path from 'x1,y1;x2,y2;...' format."""
        with self.lock:
            if name in self.objects:
                points = []
                for pt in path_str.split(';'):
                    coords = pt.strip().split(',')
                    if len(coords) >= 2:
                        points.append((float(coords[0]), float(coords[1])))
                self.objects[name].path = points

    def humanoid_set_waypoints(self, name: str, waypoints_str: str):
        """Set waypoints from 'x1,y1;x2,y2;...' format."""
        with self.lock:
            if name in self.objects:
                points = []
                for pt in waypoints_str.split(';'):
                    coords = pt.strip().split(',')
                    if len(coords) >= 2:
                        points.append((float(coords[0]), float(coords[1])))
                self.objects[name].waypoints = points

    # Camera management
    def set_camera_location(self, cam_id: int, x: float, y: float, z: float):
        with self.lock:
            if cam_id not in self.cameras:
                self.cameras[cam_id] = CameraState(
                    position=np.array([x, y, z]),
                    rotation=np.array([0.0, 0.0, 0.0]),
                )
            else:
                self.cameras[cam_id].position = np.array([x, y, z])

    def get_camera_location(self, cam_id: int) -> np.ndarray:
        with self.lock:
            if cam_id in self.cameras:
                return self.cameras[cam_id].position.copy()
            return np.array([0.0, 0.0, 0.0])

    def set_camera_rotation(self, cam_id: int, pitch: float, yaw: float, roll: float):
        with self.lock:
            if cam_id not in self.cameras:
                self.cameras[cam_id] = CameraState(
                    position=np.array([0.0, 0.0, 0.0]),
                    rotation=np.array([pitch, yaw, roll]),
                )
            else:
                self.cameras[cam_id].rotation = np.array([pitch, yaw, roll])

    def get_camera_rotation(self, cam_id: int) -> np.ndarray:
        with self.lock:
            if cam_id in self.cameras:
                return self.cameras[cam_id].rotation.copy()
            return np.array([0.0, 0.0, 0.0])

    def set_camera_fov(self, cam_id: int, fov: float):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id].fov = fov

    def get_camera_fov(self, cam_id: int) -> float:
        with self.lock:
            if cam_id in self.cameras:
                return self.cameras[cam_id].fov
            return 90.0

    def set_camera_resolution(self, cam_id: int, w: int, h: int):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id].width = w
                self.cameras[cam_id].height = h

    def get_camera_resolution(self, cam_id: int) -> tuple:
        with self.lock:
            if cam_id in self.cameras:
                return (self.cameras[cam_id].width, self.cameras[cam_id].height)
            return (1280, 720)

    # Vehicle actions
    def vehicle_set_state(self, name: str, throttle: float, brake: float, steering: float):
        with self.lock:
            if name in self.objects:
                obj = self.objects[name]
                if throttle > 0:
                    obj.is_moving = True
                    obj.speed = throttle * 1000  # scale
                if brake > 0:
                    obj.is_moving = False

    def vehicle_u_turn(self, name: str):
        with self.lock:
            if name in self.objects:
                self.objects[name].rotation[1] += 180.0

    # Pedestrian actions (reuse humanoid logic)
    def pedestrian_move_forward(self, name: str):
        self.humanoid_move_forward(name)

    def pedestrian_stop(self, name: str):
        self.humanoid_stop(name)

    def pedestrian_rotate(self, name: str, angle: float, clockwise: int):
        self.humanoid_rotate(name, angle, clockwise)

    def pedestrian_set_speed(self, name: str, speed: float):
        self.humanoid_set_speed(name, speed)

    def pedestrian_set_waypoints(self, name: str, waypoints_str: str):
        self.humanoid_set_waypoints(name, waypoints_str)
