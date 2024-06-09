# %%
from render_util.gaussian_model import *
from render_util.gaussian_fuse import *
gau = GaussianModel(sh_degree=3)
gau.load_ply("data/gs_cup.ply")

# %%
rt = np.array([[0.134, -0.013, 0.111, 0.004],
               [-0.112, -0.008, 0.134, 0.018],
               [-0.004, -0.174, -0.014, -0.040],
               [0.0, 0.0, 0.0, 1.0]])
scale = 0.174504

# %%
rotation = rt[:3, :3]
translation = rt[None, :3, 3]
rotation = rotation / scale
gau._xyz = torch.tensor(gau._xyz @ rotation.T * scale + translation)
gau._scaling = torch.tensor(gau._scaling + np.log(scale))

rot_matrix_2_transform = torch.matmul(torch.tensor(rotation[None,:,:], dtype=torch.float), quaternion_to_matrix(torch.tensor(gau._rotation, dtype=torch.float)))
gau._rotation = torch.tensor(matrix_to_quaternion(rot_matrix_2_transform).cpu().numpy())


# %%
from render_util.gaussian_fuse import *
gau._features_rest = torch.tensor(sh_rotation(gau._features_rest, gau._features_dc, rotation))

# %%

gau.save_ply("test.ply")

