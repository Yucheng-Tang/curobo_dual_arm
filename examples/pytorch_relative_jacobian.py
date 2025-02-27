import time

import torch
import math
import pytorch_kinematics as pk
from curobo.util_file import  join_path, load_yaml
from curobo.util_file import get_robot_configs_path

import pytorch3d.transforms

import numpy as np

def inverse_homogeneous_matrix(T):
    # Extract the rotation matrix (3x3) and translation vector (3x1)
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute the inverse of the rotation matrix (transpose of the rotation matrix)
    R_inv = R.T

    # Compute the inverse of the translation vector
    t_inv = -torch.matmul(R_inv, t)

    # Create the inverse homogeneous transformation matrix
    T_inv = torch.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

# Generate random joint angles within the limits
def generate_random_joint_angles(joint_limits, batch_size):
    random_joint_angles = []
    for _ in range(batch_size):
        batch_sample = [torch.distributions.uniform.Uniform(low, high).sample().item() for low, high in joint_limits]
        random_joint_angles.append(batch_sample)
    return torch.tensor(random_joint_angles)

# config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
# urdf_file = config_file["robot_cfg"]["kinematics"][
#     "urdf_path"
# ]

urdf_file = "robot/franka_description/dual_franka_panda.urdf"
urdf_file_test = "robot/franka_description/dual_franka_panda_test.urdf"

# ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

# ee_link_left = "left_panda_hand"
# ee_link_right = "right_panda_hand"

ee_link_left = "left_ee_link"
ee_link_right = "right_ee_link"

chain_cpu_right = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/" + urdf_file).read(), ee_link_right)
chain_cpu_left = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/" + urdf_file).read(), ee_link_left)
chain_cpu_relative = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/" + urdf_file_test).read(), ee_link_right)


d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

chain_right = chain_cpu_right.to(dtype=dtype, device=d)
chain_left = chain_cpu_left.to(dtype=dtype, device=d)
chain_relative = chain_cpu_relative.to(dtype=dtype, device=d)

joint_limits = [
    (-2.8973, 2.8973),          # Joint 1
    (-1.7628, 1.7628),      # Joint 2
    (-2.8973, 2.8973),      # Joint 3
    (-3.0718, -0.0698),          # Joint 4
    (-2.8973, 2.8973),          # Joint 5
    (-0.0175, 3.7525),      # Joint 6
    (-2.8973, 2.8973)           # Joint 7
]

# require gradient through the input joint values
N = 1
th_right = generate_random_joint_angles(joint_limits, N).to(dtype=dtype, device=d)
th_left = generate_random_joint_angles(joint_limits, N).to(dtype=dtype, device=d)

# th_right = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)
# th_left = th_right
# th_left = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)

# th_right = torch.zeros(N, 7, dtype=dtype, device=d, requires_grad=True)
# th_left = torch.zeros(N, 7, dtype=dtype, device=d, requires_grad=True)
# with torch.no_grad():
#     th_left[0, 6] = math.pi / 4
#     th_right[0, 6] = math.pi/4
#     th_left[0, 3] = -math.pi / 2
#     th_right[0, 3] = -math.pi / 2
#     # th_left[0, 2] = math.pi / 2
#     # th_right[0, 2] = math.pi / 2



th_relative = torch.cat((-th_left.flip(1), th_right), dim=1)

print(th_left, th_left.flip(1), th_right, th_relative)

# th = torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], [0.0, -math.pi / 3.0, 0.0, math.pi / 6.0, 0.0, math.pi / 5.0, 0.0]], requires_grad=True)
tg_left = chain_left.forward_kinematics(th_left)
tg_right = chain_right.forward_kinematics(th_right)
tg_relative = chain_relative.forward_kinematics(th_relative)

print(tg_left.get_matrix().round(decimals=2), tg_right.get_matrix().round(decimals=2))

print("____________________________inverse Matrix_______________________________________")
print(inverse_homogeneous_matrix(tg_right.get_matrix()[0]).round(decimals=2), tg_right._get_matrix_inverse().round(decimals=2))

# tg_relative_gt = torch.matmul(tg_left.get_matrix(), tg_right._get_matrix_inverse())
tg_relative_gt = torch.matmul(tg_left._get_matrix_inverse(), tg_right.get_matrix())

# quat1 = pytorch3d.transforms.quaternion_to_matrix(torch.tensor([0.4103,  0.3481, -0.6309, -0.5590]))
# quat2 = pytorch3d.transforms.quaternion_to_matrix(torch.tensor([0.2204, -0.6938, -0.4928, -0.4766]))
# pos1 = torch.tensor([0.8262,  0.9111, -0.7027])
# pos2 = torch.tensor([-1.5438,  1.3756,  0.0537])
#
# T1 = torch.eye(4)
# T1[:3, :3] = quat1
# T1[:3, 3] = pos1
#
# T2 = torch.eye(4)
# T2[:3, :3] = quat2
# T2[:3, 3] = pos2
#
# relative_pos = torch.matmul(inverse_homogeneous_matrix(T2), T1)
#
# print("check: ", relative_pos[:3, 3], pytorch3d.transforms.matrix_to_quaternion(relative_pos[:3, :3]))


print("____________________________relative Transform_______________________________________")
print(tg_relative.get_matrix().round(decimals=2))
print(tg_relative_gt.round(decimals=2))

J_rel = chain_relative.jacobian(th_relative)
J_left = chain_left.jacobian(th_left)
J_right = chain_right.jacobian(th_right)

diag_R_ra = torch.block_diag(tg_left.get_matrix().squeeze()[:3, :3], tg_left.get_matrix().squeeze()[:3, :3])
diag_R_rt = torch.block_diag(tg_relative.get_matrix().squeeze()[:3, :3], tg_relative.get_matrix().squeeze()[:3, :3])
diag_R_tb = torch.block_diag(tg_right.get_matrix().squeeze()[:3, :3], tg_right.get_matrix().squeeze()[:3, :3])



J_rel_cal = torch.cat((-diag_R_ra @ J_right, diag_R_rt @ diag_R_tb @ J_left), dim=2)

print(J_rel.round(decimals=2),(-diag_R_ra @ J_right).round(decimals=2), (diag_R_rt @ diag_R_tb @ J_left).round(decimals=2))

# m = tg.get_matrix()
# pos = m[:, :3, 3]
# rot = m[:, :3, :3]
# euler_angle = pytorch3d.transforms.matrix_to_euler_angles(rot, convention="XYX")
#
# jacobian = []
# grad_hessian = []
#
# cal_time = time.time()
# hessian = torch.zeros(N, 6, 7, 7, device="cuda", dtype=torch.float32)
# for i in range(pos.shape[0]):
#     for j in range(pos.shape[1]):
#         # print("lin_jac:", i, j, torch.autograd.grad(pos[i, j], th, retain_graph=True)[0][i])
#         jac_tmp = torch.autograd.grad(pos[i, j], th, retain_graph=True, create_graph=True)[0][i]
#         jacobian.append(jac_tmp)
#         for k in range(jac_tmp.shape[0]):
#             # print(torch.autograd.grad(jac_tmp[k], th, retain_graph=True)[0][i])
#             hessian[i, j, k, :] = torch.autograd.grad(jac_tmp[k], th, retain_graph=True)[0][i]
# #     pos[:, i].backward()
# # jacobian = torch.stack(jacobian, dim=-1)
# # axis_in_eef = cur_transform[:, :3, :3].transpose(1, 2) @ f.joint.axis
# cal_time_end = time.time() - cal_time
# print("grad cal time:", cal_time_end)
#
# # for i in range(euler_angle.shape[0]):
# #     for j in range(euler_angle.shape[1]):
# #         print("ang_jac: ", i, j, torch.autograd.grad(euler_angle[i, j], th, retain_graph=True)[0][i])
# #         jacobian.append(torch.autograd.grad(euler_angle[i, j], th, retain_graph=True)[0][i])
# # print(jacobian)
#
# cal_time = time.time()
# J = chain.jacobian(th)
#
#
# print(J.requires_grad)
# hessian = torch.zeros(N, 6, 7, 7, device="cuda", dtype=torch.float32)
# for i in range(J.shape[0]):
#     for j in range(J.shape[1]):
#         for k in range(J.shape[2]):
#             hessian[i, j, k, :] = torch.autograd.grad(J[i, j, k], th, retain_graph=True)[0][i]
#             # print("hessian", i, j, k, torch.autograd.grad(J[i, j, k], th, retain_graph=True)[0][i])
# cal_time = time.time() - cal_time
# print("torch cal time:", cal_time)

#
# print(th.grad)