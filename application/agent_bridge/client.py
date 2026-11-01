"""Synchronous single-client attachment, usable without Qt or the MCP SDK."""
import socket
from threading import RLock

from .protocol import VERSION, MAX_RESPONSE, BridgeError, find_discovery, read_discovery, read_message, write_message


class BridgeClient:
    def __init__(self, discovery_file=None, *, timeout=125):
        identity = read_discovery(find_discovery(discovery_file))
        self.socket = socket.create_connection((identity['host'], identity['port']), timeout=timeout)
        self.stream = self.socket.makefile('rwb')
        self._lock = RLock()
        try:
            write_message(self.stream, {key: identity[key] for key in ('version', 'session_id', 'token')})
            self.info = read_message(self.stream, 4096)
            if not self.info.get('ok'):
                raise BridgeError(self.info.get('message', 'bridge connection refused'))
            if self.info.get('version') != VERSION or self.info.get('session_id') != identity['session_id']:
                raise BridgeError('GUI bridge identity changed')
        except Exception:
            self.close()
            raise

    def call(self, name, arguments):
        with self._lock:
            try:
                write_message(self.stream, {'name': name, 'arguments': arguments}, limit=262144)
                return read_message(self.stream, MAX_RESPONSE)
            except (OSError, EOFError, ValueError) as error:
                self.close()
                raise BridgeError('GUI bridge disconnected; no headless fallback during an attached session') from error

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.stream.close()
        self.socket.close()
