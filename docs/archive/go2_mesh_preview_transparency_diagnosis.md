# Go2 mesh and preview transparency diagnosis

> Historical note: this diagnosis may not describe the current implementation.
> See [Models](../models.md) for supported behavior.

G1 is registered directly to `models/g1_29dof.xml`. Its MJCF compiler points
`meshdir` at `models/assets-g1`, declares 35 STL mesh assets, and assigns those
meshes to visual group 2. `RobotModel3D` lets MuJoCo resolve those paths relative
to the XML file, while `RobotCanvas3D` renders MuJoCo's compiled vertex/face
arrays.

Go2 was instead registered to the hand-written `models/go2.xml`. That model
contains no mesh assets (`nmesh == 0`): its body, hips, thighs, calves, and feet
are boxes, cylinders, capsules, and spheres. The repository does contain the
real ROS description and seven COLLADA files under `models/assets-go2`, and all
17 `package://go2_description/dae/...` references resolve to those files. They
were never reached by the registry. The older generic URDF conversion also
removed every visual block because MuJoCo rejects the URDF's repeated material
elements and does not directly accept DAE as a mesh asset. The result was an
intentional collision-primitive fallback, but it appeared without an actionable
asset warning.

The corrected path registers the Go2 URDF, resolves package and relative mesh
URIs without using the process working directory, validates every reference,
and converts its triangulated Z-up DAE material parts to cached OBJ files before
MuJoCo compiles the URDF. MuJoCo is explicitly told not to discard URDF visuals;
the renderer selects its non-colliding group-1 visual meshes rather than the
primitive collision geoms. Missing paths, unsupported formats, malformed DAE,
and conversion failures are now load errors rather than silent fallback.

The preview artifact was separate. The OpenGL clear color already had alpha 1,
and no top-level translucent Qt flag was set. However, blending was enabled
globally with the same factors for color and alpha. Drawing the orange preview
or ghosts therefore reduced the destination framebuffer alpha below 1. On
desktops where `QOpenGLWidget`'s backing texture is alpha-composited, those
pixels exposed the application or desktop behind the widget.

The viewer now requests an opaque surface, marks opaque paint events, renders
opaque scene/main geometry first, and enables blending only around ghost and
preview passes. Those passes disable alpha-channel writes and restore blending,
depth-mask, and color-mask state afterward. Preview alpha affects only the
orange geom colors and is user-adjustable; it cannot change window alpha.
