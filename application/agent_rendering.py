"""Bounded offscreen images, independent of Qt and the interactive viewer."""
from dataclasses import dataclass
import struct
import zlib

import numpy as np


@dataclass(frozen=True)
class RenderedImage:
    data: bytes
    mime_type: str = "image/png"


@dataclass(frozen=True)
class ImageResult:
    payload: dict
    images: tuple[RenderedImage, ...] = ()


class RenderingUnavailable(RuntimeError):
    pass


def png_rgb(pixels):
    """Encode small RGB arrays without adding an imaging dependency."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("RGB pixels required")
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


# Small timestamp glyphs keep image creation dependency-free.
_GLYPHS = dict(zip('0123456789.fs', (
    '111101101101111', '010110010010111', '111001111100111',
    '111001111001111', '101101111001001', '111100111001111',
    '111100111101111', '111001001001001', '111101111101111',
    '111101111001111', '000000000000010', '111100110100100',
    '111100111001111')))


def label_pixels(pixels, label):
    pixels = np.asarray(pixels, dtype=np.uint8).copy()
    pixels[:20, :] = 20
    for index, character in enumerate(label):
        glyph = _GLYPHS.get(character, '0' * 15)
        for row in range(5):
            for column in range(3):
                if glyph[row * 3 + column] == '1':
                    x, y = 5 + index * 8 + column * 2, 5 + row * 2
                    pixels[y:y + 2, x:x + 2] = 255
    return pixels


class MuJoCoViewRenderer:
    def render(self, model, samples, views, width, height):
        import mujoco
        renderer = None
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            data = mujoco.MjData(model)
            images = []
            for time, qpos in samples:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                for view in views:
                    camera = mujoco.MjvCamera()
                    mujoco.mjv_defaultFreeCamera(model, camera)
                    camera.lookat[:] = data.subtree_com[0]
                    camera.distance = max(float(model.stat.extent) * 1.6, 1.0)
                    camera.azimuth = 0 if view == 'front' else 90
                    camera.elevation = -10
                    renderer.update_scene(data, camera=camera)
                    pixels = label_pixels(renderer.render(), f'{view[0]} {time:.3f}s')
                    images.append(RenderedImage(png_rgb(pixels)))
            return tuple(images)
        except Exception as error:
            raise RenderingUnavailable(f'offscreen renderer unavailable: {error}') from error
        finally:
            if renderer is not None:
                renderer.close()


def render_workspace(workspace, args, renderer):
    document = workspace.document(args.get('source', 'candidate'))
    times = args.get('times_s', [0.0])
    views = args.get('views', ['front', 'side'])
    if len(times) * len(views) > 4:
        raise ValueError('render_views permits at most four images')
    if len(set(views)) != len(views):
        raise ValueError('render views must be unique')
    if any(time > document.timeline_duration for time in times):
        raise ValueError('render time exceeds the motion duration')
    identity = workspace.identity(document)
    samples = [(time, document.qpos_timeline.sample_state(time).copy()) for time in times]
    generated = workspace.generated
    if generated is not None and generated.source == identity:
        # Exact sampled generated poses use the existing manifold timeline sampler.
        from core.models import RobotStateTimeline
        timeline = RobotStateTimeline(workspace.adapter)
        for time, qpos in zip(generated.export.times, generated.export.qposes):
            timeline.set_state(time, qpos)
        samples = [(time, timeline.sample_state(time).copy()) for time in times]
    try:
        images = renderer.render(workspace.adapter.mj_model, samples, views,
                                 args.get('width', 480), args.get('height', 360))
    except RenderingUnavailable as error:
        return ImageResult({'available': False, 'code': 'rendering_unavailable', 'message': str(error)[:1000]})
    if len(images) != len(times) * len(views):
        raise ValueError('renderer returned an unexpected image count')
    if any(image.mime_type != 'image/png' or not image.data.startswith(b'\x89PNG\r\n\x1a\n') for image in images):
        raise ValueError('renderer must return PNG images')
    if sum(len(image.data) for image in images) > 2 * 1024 * 1024:
        raise ValueError('rendered images exceed the 2 MiB limit')
    return ImageResult({'available': True, 'source_revision': document.revision,
                        'motion_source': 'generated' if generated is not None and generated.source == identity else 'editable',
                        'images': [{'time_s': time, 'view': view} for time in times for view in views]}, images)
