class CameraData:
    def __init__(self, focal_length, matrix_width, matrix_height, x, y, z, az, sphere_diameter):
        self.focal_length = focal_length
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        self.x = x
        self.y = y
        self.z = z
        self.az = az
        self.sphere_diameter = sphere_diameter

    def __str__(self):
        return f"CameraData({self.focal_length=}, {self.matrix_width=}, {self.matrix_height=}, {self.x=}, {self.y=}, {self.z=}, {self.az=}, {self.sphere_diameter=})"

    def __repr__(self):
        return str(self)
