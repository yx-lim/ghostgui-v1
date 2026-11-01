"""Bounded newline JSON framing and private per-process discovery."""
import json
import os
from pathlib import Path
import stat

from application.project_manager import ghostgui_config_dir

VERSION = 1
MAX_REQUEST = 262144
MAX_RESPONSE = 4 * 1024 * 1024


class BridgeError(RuntimeError):
    pass


def discovery_directory():
    return ghostgui_config_dir() / 'agent-bridges'


def read_message(stream, limit=MAX_REQUEST):
    line = stream.readline(limit + 1)
    if not line:
        raise EOFError('bridge disconnected')
    if len(line) > limit or not line.endswith(b'\n'):
        raise BridgeError('bridge message exceeds the byte limit')
    value = json.loads(line)
    if not isinstance(value, dict):
        raise BridgeError('bridge message must be an object')
    return value


def write_message(stream, value, limit=MAX_RESPONSE):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(data) > limit:
        raise BridgeError('bridge response exceeds the byte limit')
    stream.write(data)
    stream.flush()


def read_discovery(path):
    path = Path(path).expanduser()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BridgeError('discovery must be a regular file')
        if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise BridgeError('discovery file must be owner-only (0600)')
        raw = handle.read(4097)
    if len(raw) > 4096:
        raise BridgeError('discovery file is too large')
    result = json.loads(raw)
    if (not isinstance(result, dict) or result.get('version') != VERSION or result.get('host') != '127.0.0.1'
            or type(result.get('port')) is not int or not 0 < result['port'] < 65536
            or not isinstance(result.get('token'), str) or len(result['token']) < 32
            or not isinstance(result.get('session_id'), str)):
        raise BridgeError('invalid bridge discovery identity')
    return result


def find_discovery(path=None):
    if path is not None:
        return Path(path)
    paths = sorted(discovery_directory().glob('*.json'))
    if not paths:
        raise BridgeError('no GUI has enabled external agent control')
    if len(paths) != 1:
        raise BridgeError('multiple GUI bridges found; select --discovery-file explicitly')
    return paths[0]
