"""Loopback listener; dispatch futures keep network IO off the Qt thread."""
from concurrent.futures import CancelledError, TimeoutError
import hmac
import json
import os
from pathlib import Path
import secrets
import select
import socket
from threading import BoundedSemaphore, Event, Lock, Thread
import time
from uuid import uuid4

from .protocol import VERSION, BridgeError, discovery_directory, read_message, write_message


class BridgeServer:
    def __init__(self, submit, *, directory=None, info=None, connection_changed=None, call_timeout=120):
        self.submit = submit
        self.info = dict(info or {})
        self.connection_changed = connection_changed or (lambda connected: None)
        self.call_timeout = call_timeout
        self.session_id = uuid4().hex
        self.token = secrets.token_urlsafe(32)
        self.directory = Path(directory) if directory is not None else discovery_directory()
        self.path = self.directory / f'{os.getpid()}-{self.session_id}.json'
        self._stop = Event()
        self._lock = Lock()
        self._slots = BoundedSemaphore(4)
        self._client = None
        self._cancellation = None
        self.listener = None

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.stat()
        if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise BridgeError('bridge discovery directory must be owner-only (0700)')
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(('127.0.0.1', 0))
            listener.listen(4)
            listener.settimeout(0.2)
            self.listener = listener
            discovery = {'version': VERSION, 'session_id': self.session_id, 'token': self.token,
                         'host': '127.0.0.1', 'port': listener.getsockname()[1]}
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as handle:
                json.dump(discovery, handle)
            self.thread = Thread(target=self._accept, name='ghostgui-bridge-listener', daemon=True)
            self.thread.start()
        except Exception:
            listener.close()
            self.path.unlink(missing_ok=True)
            raise
        return self.path

    def _notify(self, connected):
        try:
            self.connection_changed(connected)
        except RuntimeError:
            # The GUI may already have destroyed its signal owner at shutdown.
            pass

    def _accept(self):
        while not self._stop.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                connection.close()
                continue
            Thread(target=self._handle, args=(connection,), daemon=True,
                   name='ghostgui-bridge-client').start()

    def _handle(self, connection):
        claimed = False
        cancel = Event()
        try:
            connection.settimeout(5)
            with connection, connection.makefile('rwb') as stream:
                hello = read_message(stream, 4096)
                if (hello.get('version') != VERSION or hello.get('session_id') != self.session_id
                        or not isinstance(hello.get('token'), str)
                        or not hmac.compare_digest(hello['token'], self.token)):
                    write_message(stream, {'ok': False, 'message': 'bridge authentication failed'})
                    return
                with self._lock:
                    if self._client is not None or self._stop.is_set():
                        write_message(stream, {'ok': False, 'message': 'a GUI agent client is already active'})
                        return
                    self._client, self._cancellation = connection, cancel
                    claimed = True
                self._notify(True)
                write_message(stream, {'ok': True, 'version': VERSION, 'session_id': self.session_id,
                                       **self.info, 'workspace': 'gui', 'attach': 'required'})
                connection.settimeout(None)
                while not self._stop.is_set():
                    request = read_message(stream)
                    if set(request) != {'name', 'arguments'}:
                        raise BridgeError('expected a tool name and arguments')
                    future = self.submit(request['name'], request['arguments'], cancel)
                    deadline = time.monotonic() + self.call_timeout
                    while True:
                        if cancel.is_set() or self._stop.is_set() or time.monotonic() >= deadline:
                            cancel.set()
                            future.cancel()
                            raise BridgeError('bridge invocation cancelled or timed out')
                        readable, _, _ = select.select([connection], [], [], 0)
                        if readable and not connection.recv(1, socket.MSG_PEEK):
                            cancel.set()
                            future.cancel()
                            return
                        try:
                            outcome = future.result(timeout=0.05)
                            break
                        except TimeoutError:
                            continue
                    write_message(stream, outcome)
        except (OSError, EOFError, ValueError, BridgeError, CancelledError):
            pass
        finally:
            cancel.set()
            if claimed:
                with self._lock:
                    if self._client is connection:
                        self._client = self._cancellation = None
                self._notify(False)
            self._slots.release()

    def stop(self):
        self._stop.set()
        self.path.unlink(missing_ok=True)
        with self._lock:
            if self._cancellation is not None:
                self._cancellation.set()
            if self._client is not None:
                try:
                    self._client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if self.listener is not None:
            self.listener.close()
