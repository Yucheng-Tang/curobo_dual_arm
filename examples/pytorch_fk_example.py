import time

import torch
import math
import pytorch_kinematics as pk
from curobo.util_file import  join_path, load_yaml
from curobo.util_file import get_robot_configs_path

import pytorch3d.transforms



config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
urdf_file = config_file["robot_cfg"]["kinematics"][
    "urdf_path"
]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

chain_cpu = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/" + urdf_file).read(), ee_link)

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

chain = chain_cpu.to(dtype=dtype, device=d)

# require gradient through the input joint values
N = 1000
th = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)
print(th.H, th.T)

# th = torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], [0.0, -math.pi / 3.0, 0.0, math.pi / 6.0, 0.0, math.pi / 5.0, 0.0]], requires_grad=True)
tg = chain.forward_kinematics(th)
m = tg.get_matrix()
pos = m[:, :3, 3]
rot = m[:, :3, :3]
euler_angle = pytorch3d.transforms.matrix_to_euler_angles(rot, convention="XYX")

jacobian = []
grad_hessian = []

cal_time = time.time()
hessian = torch.zeros(N, 6, 7, 7, device="cuda", dtype=torch.float32)
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        # print("lin_jac:", i, j, torch.autograd.grad(pos[i, j], th, retain_graph=True)[0][i])
        jac_tmp = torch.autograd.grad(pos[i, j], th, retain_graph=True, create_graph=True)[0][i]
        jacobian.append(jac_tmp)
        for k in range(jac_tmp.shape[0]):
            # print(torch.autograd.grad(jac_tmp[k], th, retain_graph=True)[0][i])
            hessian[i, j, k, :] = torch.autograd.grad(jac_tmp[k], th, retain_graph=True)[0][i]
#     pos[:, i].backward()
# jacobian = torch.stack(jacobian, dim=-1)
# axis_in_eef = cur_transform[:, :3, :3].transpose(1, 2) @ f.joint.axis
cal_time_end = time.time() - cal_time
print("grad cal time:", cal_time_end)

# for i in range(euler_angle.shape[0]):
#     for j in range(euler_angle.shape[1]):
#         print("ang_jac: ", i, j, torch.autograd.grad(euler_angle[i, j], th, retain_graph=True)[0][i])
#         jacobian.append(torch.autograd.grad(euler_angle[i, j], th, retain_graph=True)[0][i])
# print(jacobian)

cal_time = time.time()
J = chain.jacobian(th)


print(J.requires_grad)
hessian = torch.zeros(N, 6, 7, 7, device="cuda", dtype=torch.float32)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        for k in range(J.shape[2]):
            hessian[i, j, k, :] = torch.autograd.grad(J[i, j, k], th, retain_graph=True)[0][i]
            # print("hessian", i, j, k, torch.autograd.grad(J[i, j, k], th, retain_graph=True)[0][i])
cal_time = time.time() - cal_time
print("torch cal time:", cal_time)

#
# print(th.grad)