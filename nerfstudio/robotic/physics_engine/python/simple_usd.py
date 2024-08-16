from pxr import Sdf, Usd, UsdGeom, Vt, Gf

class AnimatedMesh:
    def __init__(self):
        self.m_layer = None
        self.m_stage = None
        self.m_mesh = None
        self.m_points_attribute = None
        self.m_extent = Gf.Range3d()
        self.m_last_frame = -1
        self.m_mesh_elements = []
        self.m_particle_x = []

    def initialize_usd(self, filename):
        # Create the layer to populate.
        self.m_layer = Sdf.Layer.CreateNew(filename)

        # Create a UsdStage with that root layer.
        self.m_stage = Usd.Stage.Open(self.m_layer)

    def initialize_topology(self):
        # Create a mesh for this surface
        self.m_mesh = UsdGeom.Mesh.Define(self.m_stage, Sdf.Path("/MeshSurface"))

        # Create appropriate buffers for vertex counts and indices, and populate them
        face_vertex_counts = []
        face_vertex_indices = []
        for element in self.m_mesh_elements:
            face_vertex_counts.append(len(element))
            for vertex in element:
                face_vertex_indices.extend(vertex)

        # Now set the attributes
        self.m_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        self.m_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    def initialize_particles(self):
        # Grab the points (Positions) attribute, and indicate it is time-varying
        self.m_points_attribute = self.m_mesh.GetPointsAttr()
        self.m_points_attribute.SetVariability(Sdf.VariabilityVarying)

    def write_frame(self, frame):
        print(f"Writing frame {frame} ...")

        # Check that there are any particles to write at all
        if not self.m_particle_x:
            raise ValueError("Empty array of input vertices")

        # Check that frames have been written in sequence
        if frame != self.m_last_frame + 1:Isaac_previous 
            raise ValueError("Non-consecutive frame sequence requested in writeFrame()")
        self.m_last_frame = frame

        # Update extent
        for pt in self.m_particle_x:
            self.m_extent.UnionWith(Gf.Vec3d(pt[0], pt[1], 0.))

        # Copy particleX into VtVec3fArray for Usd
        usd_points = Vt.Vec3fArray(len(self.m_particle_x))
        for p, particle in enumerate(self.m_particle_x):
            usd_points[p] = Gf.Vec3f(particle[0], particle[1], 0.)

        # Write the points attribute for the given frame
        self.m_points_attribute.Set(usd_points, float(frame))

    def write_usd(self):
        # Set up the timecode
        self.m_stage.SetStartTimeCode(0.0)
        self.m_stage.SetEndTimeCode(float(self.m_last_frame))

        # Set the effective extent
        extent_array = Vt.Vec3fArray(2)
        extent_array[0] = self.m_extent.GetMin()
        extent_array[1] = self.m_extent.GetMax()
        self.m_mesh.GetExtentAttr().Set(extent_array)

        # Save USD file
        self.m_stage.GetRootLayer().Save()
        print("USD file saved!")


# Example usage
if __name__ == "__main__":
    # Create an instance of the AnimatedMesh
    animated_mesh = AnimatedMesh()

    # Initialize USD, topology, and particles
    animated_mesh.initialize_usd("animated_mesh.usda")
    animated_mesh.initialize_topology()
    animated_mesh.initialize_particles()

    # Simulate and add data for frames
    for frame_number in range(10):
        # Simulated particle data for the frame
        simulated_particles = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

        # Write the frame
        animated_mesh.m_particle_x = simulated_particles
        animated_mesh.write_frame(frame_number)

    # Write the USD file
    animated_mesh.write_usd()