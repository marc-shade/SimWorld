"""Entry point for the SimWorld macOS backend server.

Initializes world state, SceneKit renderer, loads a city scene,
wires the command router, and starts the UnrealCV protocol server
on 127.0.0.1:9000.

Usage:
    python -m simworld.backend.launch [--port 9000] [--city-dir output]
"""
import argparse
import asyncio
import logging
import os
import signal
from typing import Optional

from simworld.backend.command_router import CommandRouter
from simworld.backend.protocol_server import ProtocolServer
from simworld.backend.renderer import SceneKitRenderer
from simworld.backend.scene_loader import CitySceneLoader
from simworld.backend.world_state import WorldState

logger = logging.getLogger('simworld.backend')


def build_server(
    host: str = '127.0.0.1',
    port: int = 9000,
    width: int = 1280,
    height: int = 720,
    city_dir: Optional[str] = None,
    progen_world: Optional[str] = None,
) -> ProtocolServer:
    """Construct and wire all backend components.

    Returns a ProtocolServer ready to be started.
    """
    # 1. World state
    world = WorldState()
    world.set_resolution(width, height)

    # 2. Renderer
    renderer = SceneKitRenderer(width=width, height=height)

    # 3. Command router (routes commands → world state + renderer)
    router = CommandRouter(world)
    router.set_renderer(renderer)

    # 4. Load city scene if paths provided
    if city_dir or progen_world:
        loader = CitySceneLoader(renderer)
        if city_dir and os.path.isdir(city_dir):
            stats = loader.load_city(city_dir)
            logger.info('Loaded city from %s: %s', city_dir, stats)
        if progen_world and os.path.isfile(progen_world):
            stats = loader.load_progen_world(progen_world)
            logger.info('Loaded progen world from %s: %s', progen_world, stats)

    # 5. Protocol server
    server = ProtocolServer(
        command_callback=router.handle,
        host=host,
        port=port,
    )
    return server


async def _run(server: ProtocolServer):
    """Run the server until interrupted."""
    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info('Received shutdown signal')
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    logger.info('SimWorld macOS backend ready')

    await stop_event.wait()
    await server.stop()
    logger.info('Server shut down cleanly')


def main():
    parser = argparse.ArgumentParser(description='SimWorld macOS backend server')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address')
    parser.add_argument('--port', type=int, default=9000, help='TCP port')
    parser.add_argument('--width', type=int, default=1280, help='Render width')
    parser.add_argument('--height', type=int, default=720, help='Render height')
    parser.add_argument('--city-dir', default=None,
                        help='Path to citygen output directory (buildings.json, roads.json, etc.)')
    parser.add_argument('--progen-world', default=None,
                        help='Path to progen_world.json')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    server = build_server(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        city_dir=args.city_dir,
        progen_world=args.progen_world,
    )

    asyncio.run(_run(server))


if __name__ == '__main__':
    main()
