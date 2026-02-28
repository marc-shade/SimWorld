"""SimWorld macOS native backend.

Provides a TCP server implementing the UnrealCV protocol backed by
SceneKit + Metal for GPU-accelerated offscreen rendering on Apple Silicon.
"""
from simworld.backend.world_state import CameraState, SimObject, WorldState
from simworld.backend.protocol_server import ProtocolServer
from simworld.backend.command_router import CommandRouter
from simworld.backend.renderer import SceneKitRenderer
from simworld.backend.scene_loader import CitySceneLoader

__all__ = [
    'CameraState',
    'CitySceneLoader',
    'CommandRouter',
    'ProtocolServer',
    'SceneKitRenderer',
    'SimObject',
    'WorldState',
]
