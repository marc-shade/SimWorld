"""Asyncio TCP server implementing the UnrealCV wire protocol for SimWorld macOS backend."""
import asyncio
import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

MAGIC = 0x9E2B83C1
HEADER_SIZE = 8  # 4 bytes magic + 4 bytes payload size


def _build_frame(payload: bytes) -> bytes:
    """Build a wire-protocol frame: [magic_u32_le][size_u32_le][payload]."""
    return struct.pack('<II', MAGIC, len(payload)) + payload


class ClientHandler:
    """Handles a single connected UnrealCV client."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 command_callback):
        self.reader = reader
        self.writer = writer
        self.command_callback = command_callback
        self._peer = writer.get_extra_info('peername')

    async def run(self):
        """Main loop: send greeting, then read requests and write responses."""
        try:
            # Send "connected" greeting on connect
            greeting = _build_frame(b'connected')
            self.writer.write(greeting)
            await self.writer.drain()
            logger.info('Client %s connected, greeting sent', self._peer)

            while True:
                # Read frame header
                header = await self.reader.readexactly(HEADER_SIZE)
                magic, payload_size = struct.unpack('<II', header)
                if magic != MAGIC:
                    logger.warning('Bad magic 0x%08X from %s, dropping', magic, self._peer)
                    break

                # Read payload
                payload = await self.reader.readexactly(payload_size)

                # Parse "msg_id:command"
                payload_str = payload.decode('utf-8', errors='replace')
                colon_idx = payload_str.find(':')
                if colon_idx < 0:
                    logger.warning('Malformed request (no colon): %r', payload_str[:80])
                    continue
                msg_id = payload_str[:colon_idx]
                command = payload_str[colon_idx + 1:]

                logger.debug('Request %s: %s', msg_id, command[:120])

                # Dispatch command
                response = self.command_callback(command)

                # Build response frame
                if isinstance(response, bytes):
                    # Binary response (e.g. camera image)
                    resp_payload = msg_id.encode('utf-8') + b':' + response
                else:
                    # Text response
                    resp_payload = f'{msg_id}:{response}'.encode('utf-8')

                frame = _build_frame(resp_payload)
                self.writer.write(frame)
                await self.writer.drain()

        except asyncio.IncompleteReadError:
            logger.info('Client %s disconnected (incomplete read)', self._peer)
        except ConnectionResetError:
            logger.info('Client %s connection reset', self._peer)
        except Exception:
            logger.exception('Error handling client %s', self._peer)
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            logger.info('Client %s handler finished', self._peer)


class ProtocolServer:
    """Asyncio TCP server that speaks the UnrealCV wire protocol.

    Usage:
        server = ProtocolServer(command_callback=router.handle, host='127.0.0.1', port=9000)
        await server.start()
        # ... run until shutdown ...
        await server.stop()
    """

    def __init__(self, command_callback, host: str = '127.0.0.1', port: int = 9000):
        self.command_callback = command_callback
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self._clients: list[asyncio.Task] = []

    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
        handler = ClientHandler(reader, writer, self.command_callback)
        task = asyncio.current_task()
        if task is not None:
            self._clients.append(task)
        try:
            await handler.run()
        finally:
            if task is not None and task in self._clients:
                self._clients.remove(task)

    async def start(self):
        """Start listening for connections."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
        )
        addrs = ', '.join(str(s.getsockname()) for s in self._server.sockets)
        logger.info('ProtocolServer listening on %s', addrs)

    async def stop(self):
        """Gracefully shut down the server and all client connections."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info('ProtocolServer stopped listening')

        # Cancel outstanding client handlers
        for task in list(self._clients):
            task.cancel()
        if self._clients:
            await asyncio.gather(*self._clients, return_exceptions=True)
        logger.info('ProtocolServer shutdown complete')

    async def serve_forever(self):
        """Start and serve until cancelled."""
        await self.start()
        if self._server is None:
            raise RuntimeError('Server failed to start')
        try:
            await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
