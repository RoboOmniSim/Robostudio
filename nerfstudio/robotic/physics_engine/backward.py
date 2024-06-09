# cite from Jatavallabhula and Macklin et al., "gradSim: Differentiable simulation for system identification and visuomotor control", ICLR 2021.



# @inproceedings{zhan2023pypose,
#   title = {{PyPose} v0.6: The Imperative Programming Interface for Robotics},
#   author = {Zitong Zhan and Xiangfu Li and Qihang Li and Haonan He and Abhinav Pandey and Haitao Xiao and Yangmengfei Xu and Xiangyu Chen and Kuan Xu and Kun Cao and Zhipeng Zhao and Zihan Wang and Huan Xu and Zihang Fang and Yutian Chen and Wentao Wang and Xu Fang and Yi Du and Tianhao Wu and Xiao Lin and Yuheng Qiu and Fan Yang and Jingnan Shi and Shaoshu Su and Yiren Lu and Taimeng Fu and Karthik Dantu and Jiajun Wu and Lihua Xie and Marco Hutter and Luca Carlone and Sebastian Scherer and Daning Huang and Yaoyu Hu and Junyi Geng and Chen Wang},
#   year = {2023},
#   booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) Workshop},
# }



import numpy as np
import pypose as pp
import torch
import torch.nn.functional as F


# usd gradsim based backward optimization

import numpy as np
from scipy.spatial.transform import Rotation as R

import open3d as o3d







def get_uv_depth(xyz,projection_matrx,view_mat,transformation_t0=None,transformation_recenter=None):
    if transformation_t0 is not None:
        # assume points are not yet in cam coord
        if transformation_recenter is not None:

            xyz_forward=pp.mat2SE3(transformation_t0).Act(xyz)
            xyz_recenter=pp.mat2SE3(transformation_recenter).Act(xyz_forward)
            
            pts_c = pp.mat2SE3(view_mat).Act(xyz_recenter)
    else:
        pts_c = xyz

    depth = pts_c[..., -1]
    return depth






def optimize_photometrix(xyz,transformation_t0,projection_matrix,view_mat,image,delta_t):
    """
    use photo metric loss here


    need the rasterize gaussian part and reserve the ctx 
    
    
    """
    xyz_optimized=0
    velocity=0


    return xyz_optimized, velocity


def optimize(xyz,transformation_t0,projection_matrix,view_mat,uv_gt,depth_gt,delta_t ):


    """
    xyz world coodinate 
    transformation_t0 (the initial transformation) from t0 to t1
    uv_gt  (h,w) in timestamp1
    depth_gt (h,w,1)  in timestamp1
    projection_matrix intrinsic
    view_mat c2w
    delta_t timestamp
    
    
    
    THis is the optimizer aims for uv optical flow based optimize
    """
    # see usage of pypose LM at https://pypose.readthedocs.io/latest/generated/pypose.optim.LevenbergMarquardt.html#pypose.optim.LevenbergMarquardt
    
    class Reproj(nn.Module):
        def __init__(self, t):
            super().__init__()
            self.pose = pp.Parameter(pp.randn_se3(t))

        def forward(self, xyz):
            # the last dimension of the output is 6,
            # which will be the residual dimension.
            pw_t1 = pp.mat2SE3(self.pose).Act(xyz) # initial t1 pts world 

    
            # pts reporject, get depth and uv
            pc_t1 = pp.mat2SE3(view_mat).Act(pw_t1)
            uv_t1 = pp.homo2cart(pc_t1)

            
            # compare with gt
            assert uv_gt.shape == uv_t1.shape
            loss = (uv_t1 - uv_gt).flatten()
            return loss

    objective_func = Reproj(transformation_t0)
    strategy = pp.optim.strategy.Adaptive(damping=1e-6)
    optimizer = pp.optim.LM(objective_func, strategy=strategy)

    for idx in range(10):
        loss = optimizer.step(xyz)
        print('Pose loss %.7f @ %d it'%(loss, idx))
        
        if loss < 1e-5:
            print('Early Stopping with loss:', loss.item())
            break

    optimized = objective_func.pose
    pts_optimized = pp.mat2SE3(optimized).Act(xyz)



    # Compute displacement vectors for each vertex
    displacements = pts_optimized - xyz

    # Compute individual velocities for each vertex
    velocities = displacements / delta_t

    # Compute average displacement and velocity for the mesh
    average_displacement = np.mean(displacements, axis=0)
    average_velocity = average_displacement / delta_t

    



    return pts_optimized,velocities,optimized





def backward_rigid(xyz,transformation_t0,transformation_recenter,uv_gt,depth_gt,projection_matrix,view_mat,delta_t):
    """
    xyz: 3D points in world coordinate
    transformation_t0: transformation matrix at time t0
    uv_gt: 2D pixel in image plane  (h,w)
    depth_gt: depth value of the 3D points reproject to the image plane (h,w,1)
    projection_matrix: projection matrix
    view_mat: view matrix in raw ns colmap format
    delta_t: time interval between t and t+1 # normal determined by fps
    transformation_recenter: transformation matrix to recenter the 3D points to the world coordinate center


    return:
    pts_optimized: optimized 3D points in t+1
    u_optimized: optimized u value in t+1
    v_optimized: optimized v value in t+1
    d_optimized: optimized depth value in t+1
    dx_dt: gradient of x
    dy_dt: gradient of y
    dz_dt: gradient of z



    this is for a fixed camera view (no batchify)
    """



    u,v,d =get_uv_depth(xyz,projection_matrix,view_mat)

    u_f,v_f,d_f = get_uv_depth(xyz,projection_matrix,view_mat,transformation_t0=transformation_t0,transformation_recenter=transformation_recenter)




    # select sam_mask

    # follow semantic mask on uv and xyz to compute only the first xyz intersect on this pixel with same semantic mask(frist depth)

    


    # compute the gradient of the loss function
    pts_optimized,velocity,optimized_pose = optimize(xyz,transformation_t0,projection_matrix,view_mat,uv_gt,depth_gt,delta_t)

    






    return pts_optimized,velocity,optimized_pose







# from gradsim 




# if pypose have this, please replace 
def normalize(quaternion):
    r"""Normalizes a quaternion to unit norm.

    Args:
        quaternion (torch.Tensor): Quaternion to normalize (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Normalized quaternion (shape: :math:`(4)`).
    """
    norm = quaternion.norm(p=2, dim=0) + 1e-5
    return quaternion / norm

# equivalent to:
# pp.SO3(quaternion).matrix()
# pp.SE3([tx, ty, tz, qx, qy, qz, qw]).matrix()
def quaternion_to_rotmat(quaternion):
    r"""Converts a quaternion to a :math:`3 \times 3` rotation matrix.

    Args:
        quaternion (torch.Tensor): Quaternion to convert (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): rotation matrix (shape: :math:`(3, 3)`).
    """
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    rotmat = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
    twoisq = 2 * i * i
    twojsq = 2 * j * j
    twoksq = 2 * k * k
    twoij = 2 * i * j
    twoik = 2 * i * k
    twojk = 2 * j * k
    twori = 2 * r * i
    tworj = 2 * r * j
    twork = 2 * r * k
    rotmat[0, 0] = 1 - twojsq - twoksq
    rotmat[0, 1] = twoij - twork
    rotmat[0, 2] = twoik + tworj
    rotmat[1, 0] = twoij + twork
    rotmat[1, 1] = 1 - twoisq - twoksq
    rotmat[1, 2] = twojk - twori
    rotmat[2, 0] = twoik - tworj
    rotmat[2, 1] = twojk + twori
    rotmat[2, 2] = 1 - twoisq - twojsq
    return rotmat


def multiply(q1, q2):
    r"""Multiply two quaternions `q1`, `q2`.

    Args:
        q1 (torch.Tensor): First quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
        q2 (torch.Tensor): Second quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Quaternion product (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
    """
    r1 = q1[0]
    v1 = q1[1:]
    r2 = q2[0]
    v2 = q2[1:]
    return torch.cat(
        (
            r1 * r2 - torch.matmul(v1.view(1, 3), v2.view(3, 1)).view(-1),
            r1 * v2 + r2 * v1 + torch.cross(v1, v2),
        ),
        dim=0,
    )


import math
from abc import ABCMeta, abstractmethod


class ODEIntegrator(metaclass=ABCMeta):
    """Abstract base class for ODE integrators. Integrators have a
    `integrate()` method that computes time derivatives of the
    state vector.
    """

    @abstractmethod
    def integrate(self, *args, **kwargs):
        pass





# if need novel time interpolation, the following code can be used

class EulerIntegrator(ODEIntegrator):
    """Performs semi-implicit Euler integration to solve the ODE. """

    def __init__(self):
        super().__init__()

    def integrate(self, simulator, dtime):
        # Compute forces (and torques).
        derivatives = simulator.compute_state_derivatives()
        # Compute state updates.
        for body, dstate in zip(simulator.bodies, derivatives):
            # body.position = body.position + dstate[0] * dtime
            body.orientation = body.orientation + dstate[1] * dtime
            body.linear_momentum = body.linear_momentum + dstate[2] * dtime
            body.angular_momentum = body.angular_momentum + dstate[3] * dtime
            body.orientation = normalize(body.orientation)
            body.linear_velocity = body.linear_momentum / body.masses.sum()
            inertia_world_inv = body.compute_inertia_world(
                body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
            )
            body.angular_velocity = torch.matmul(
                inertia_world_inv, body.angular_momentum.view(-1, 1)
            ).view(-1)
            # Update the position in the end, as that's when linear velocity is
            # available.
            body.position = body.position + body.linear_velocity * dtime

        return dtime
    
    

def compute_state_derivatives(self, time):
    r"""Compute the time-derivatives of the state vector, adopting the convention
        from Witkin and Baraff's SIGGRAPH '97 course.
        http://www.cs.cmu.edu/~baraff/sigcourse/index.html
        """

        # Derivative of position :math:`x(t)` is velocity :math:`v(t)`.
        # See Eq. (2-43) from above source.
    dposition = self.linear_velocity  # this from backward_rigid
        # Derivative of orientation :math:`q(t)` (quaternion representation) is
        # :math:`0.5 \omega(t) \circle q(t)`, where :math:`\circle` denotes
        # quaternion multiplication, where :math:`\omega(t)` is the angular velocity
        # converted to a quaternion (with `0` as the scalar component).
        # See Eq. (4-2) from above source.
    angular_velocity_quat = torch.zeros(4, dtype=self.dtype, device=self.device)
    angular_velocity_quat[1:] = self.angular_velocity
    dorientation = 0.5 * multiply(angular_velocity_quat, self.orientation)
        # Derivative of linear momentum :math:`P(t)` is force :math:`F(t)`.
        # See Eq. (2-43) from above source.
        # Derivative of angular momentum :math:`L(t)` is torque :math:`\tau(t)`.
        # See Eq. (2-43) from above source.
    dlinear_momentum, dangular_momentum = self.apply_external_forces(time)

    return dposition, dorientation, dlinear_momentum, dangular_momentum




def apply_external_forces(self, time):
    """Apply the external forces (includes torques) at the current timestep. """
    force_per_point = torch.zeros_like(self.vertices)
    torque_per_point = torch.zeros_like(self.vertices)

    for force, application_points in zip(self.forces, self.application_points):
            # Compute the force vector.
        force_vector = force.apply(time)
        torque = torch.zeros(3, dtype=self.dtype, device=self.device)
        if application_points is not None:
            mask = torch.zeros_like(self.vertices)
            inds = (
                    torch.tensor(
                        application_points, dtype=torch.long, device=self.device
                    )
                    .view(-1, 1)
                    .repeat(1, 3)
                )
            mask = mask.scatter_(0, inds, 1.0)
            force_per_point = force_per_point + mask * force_vector.view(1, 3)
            torque_per_point = torque_per_point + torch.cross(
                    self.vertices - self.position.view(1, 3), force_per_point
                )
        else:
            force_per_point = force_per_point + force_vector.view(1, 3)
                # Torque is 0 this case; axis of force passes through center of mass.

    return force_per_point.sum(0), torque_per_point.sum(0)


def compute_angular_velocity(xyz_0,xyz_optimized,delta_t):





    # angular velocity very time consuming so replace it in future 



    # Example vertex positions before and after movement
    # Each row represents a vertex. Column are the x, y, z coordinates
    vertices_initial = xyz_0
    vertices_final = xyz_optimized

    # Using SVD to find the optimal rotation matrix
    U, S, Vt = np.linalg.svd(vertices_initial.T @ vertices_final)
    rotation_matrix = U @ Vt

    # Ensure a proper rotation matrix (handling possible reflection)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = U @ Vt

    # Convert rotation matrix to axis-angle
    rot = R.from_matrix(rotation_matrix)
    angle = rot.magnitude()
    axis = rot.as_rotvec() / angle

    # Time interval (example: 1 second)
    delta_t = 1.0

    # Angular velocity
    angular_velocity = axis * angle / delta_t


    return angular_velocity



if __name__ == '__main__':
    


    # test the forward simulation and backward optimization


    # load the point cloud data 

    mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object/convex_mesh.ply'
    
    object=o3d.io.read_triangle_mesh(mesh_path)

    # down sample the object by only backward optimize several points that interpolated by the corner of object since the rigid body shares uniform motion




    

    # adapt axis 
    gravity = torch.tensor((0.0,0.0 ,-9.8), dtype=torch.float32)

    # x,y,z   gravity is z down x forward is the target movement, y pos is right
    
    # add external force

    Force = torch.tensor((10, 0.0, 0.0), dtype=torch.float32) # force in x direction

  


    # use start and final position for lever modeling



    # and also with time and mass, momentum conservation here 