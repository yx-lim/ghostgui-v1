"""OpenGL material, transform, visual-selection, and blend-state helpers."""

from OpenGL import GL


class RobotRenderer:
    @staticmethod
    def render_geom_ids(model):
        visual_ids = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == 2
        }
        if not visual_ids:
            visual_ids = {
                geom_id for geom_id in range(model.ngeom)
                if int(model.geom_group[geom_id]) == 1
                and int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            }
        return visual_ids or set(range(model.ngeom))

    @staticmethod
    def transform_matrix(position, rotation):
        matrix = [0.0] * 16
        rotation = rotation.reshape(3, 3)
        for row in range(3):
            for column in range(3):
                matrix[column * 4 + row] = float(rotation[row, column])
        matrix[15] = 1.0
        matrix[12:15] = [float(value) for value in position]
        return matrix

    @staticmethod
    def apply_geom_material(model, geom_id):
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            specular = float(model.mat_specular[material_id])
            shininess = float(model.mat_shininess[material_id]) * 128.0
            emission = float(model.mat_emission[material_id])
        else:
            specular, shininess, emission = 0.2, 32.0, 0.0
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SPECULAR,
            (specular, specular, specular, 1.0),
        )
        GL.glMaterialf(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SHININESS,
            max(0.0, min(128.0, shininess)),
        )
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_EMISSION,
            (emission, emission, emission, 1.0),
        )

    @staticmethod
    def begin_transparent_pass():
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ZERO,
            GL.GL_ONE,
        )
        GL.glDepthMask(GL.GL_FALSE)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)

    @staticmethod
    def end_transparent_pass():
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
