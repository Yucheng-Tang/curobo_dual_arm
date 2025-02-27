#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler
from torch.autograd import grad

# CuRobo
from curobo.cuda_robot_model.cuda_robot_generator import (
    CudaRobotGenerator,
    CudaRobotGeneratorConfig,
)
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser
from curobo.cuda_robot_model.types import KinematicsTensorConfig, SelfCollisionKinematicsConfig
from curobo.curobolib.kinematics import get_cuda_kinematics
from curobo.geom.types import Mesh, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util_file import get_robot_path, join_path, load_yaml

import pytorch_kinematics as pk
from curobo.util_file import get_robot_configs_path
import pytorch3d.transforms as transforms

import time
import scipy.optimize as O
import numpy as np

import math

from copy import deepcopy



@dataclass
class CudaRobotModelConfig:
    tensor_args: TensorDeviceType
    link_names: List[str]
    kinematics_config: KinematicsTensorConfig
    self_collision_config: Optional[SelfCollisionKinematicsConfig] = None
    kinematics_parser: Optional[KinematicsParser] = None
    compute_jacobian: bool = False
    use_global_cumul: bool = False
    generator_config: Optional[CudaRobotGeneratorConfig] = None

    def get_joint_limits(self):
        return self.kinematics_config.joint_limits

    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        ee_link: str,
        link_names: Optional[List[str]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            ee_link : Name of end-effector link.
            tensor_args : Device to load robot model. Defaults to TensorDeviceType().

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        config = CudaRobotGeneratorConfig(base_link, ee_link, tensor_args, link_names, urdf_path=urdf_path)
        return CudaRobotModelConfig.from_config(config)

    @staticmethod
    def from_basic_usd(
        usd_path: str,
        usd_robot_root: str,
        base_link: str,
        ee_link: str,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            ee_link : Name of end-effector link.
            tensor_args : Device to load robot model. Defaults to TensorDeviceType().

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        config = CudaRobotGeneratorConfig(
            tensor_args,
            base_link,
            ee_link,
            usd_path=usd_path,
            usd_robot_root=usd_robot_root,
            use_usd_kinematics=True,
        )
        return CudaRobotModelConfig.from_config(config)

    @staticmethod
    def from_robot_yaml_file(
        file_path: str,
        ee_link: Optional[str] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        config_file = load_yaml(join_path(get_robot_path(), file_path))["robot_cfg"]["kinematics"]
        if ee_link is not None:
            config_file["ee_link"] = ee_link
        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**config_file, tensor_args=tensor_args)
        )

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**data_dict, tensor_args=tensor_args)
        )

    @staticmethod
    def from_config(config: CudaRobotGeneratorConfig):
        # create a config generator and load all values
        generator = CudaRobotGenerator(config)
        return CudaRobotModelConfig(
            tensor_args=generator.tensor_args,
            link_names=generator.link_names,
            kinematics_config=generator.kinematics_config,
            self_collision_config=generator.self_collision_config,
            kinematics_parser=generator.kinematics_parser,
            use_global_cumul=generator.use_global_cumul,
            compute_jacobian=generator.compute_jacobian,
            generator_config=config,
        )

    @property
    def cspace(self):
        return self.kinematics_config.cspace

    @property
    def dof(self) -> int:
        return self.kinematics_config.n_dof


@dataclass
class CudaRobotModelState:
    """Dataclass that stores kinematics information."""

    #: End-effector position stored as x,y,z in meters [b, 3]. End-effector is defined by
    #: :attr:`CudaRobotModel.ee_link`.
    ee_position: torch.Tensor

    #: End-effector orientaiton stored as quaternion qw, qx, qy, qz [b,4]. End-effector is defined
    #: by :attr:`CudaRobotModel.ee_link`.
    ee_quaternion: torch.Tensor

    #: Linear Jacobian. Currently not supported.
    lin_jacobian: Optional[torch.Tensor] = None

    #: Angular Jacobian. Currently not supported.
    ang_jacobian: Optional[torch.Tensor] = None

    #: Position of links specified by link_names  (:attr:`CudaRobotModel.link_names`).
    links_position: Optional[torch.Tensor] = None

    #: Quaternions of links specified by link names (:attr:`CudaRobotModel.link_names`).
    links_quaternion: Optional[torch.Tensor] = None

    #: Position of spheres specified by collision spheres (:attr:`CudaRobotModel.robot_spheres`)
    #: in x, y, z, r format [b,n,4].
    link_spheres_tensor: Optional[torch.Tensor] = None

    #: Names of links that each index in :attr:`links_position` and :attr:`links_quaternion`
    #: corresponds to.
    link_names: Optional[str] = None

    @property
    def ee_pose(self) -> Pose:
        """Get end-effector pose as a Pose object."""
        return Pose(self.ee_position, self.ee_quaternion)

    def get_link_spheres(self) -> torch.Tensor:
        """Get spheres representing robot geometry as a tensor with [batch,4],  [x,y,z,radius]."""
        return self.link_spheres_tensor

    @property
    def link_pose(self) -> Union[None, Dict[str, Pose]]:
        """Deprecated, use link_poses."""
        return self.link_poses

    @property
    def link_poses(self) -> Union[None, Dict[str, Pose]]:
        """Get link poses as a dictionary of link name to Pose object."""
        link_poses = None
        if self.link_names is not None:
            link_poses = {}
            link_pos = self.links_position.contiguous()
            link_quat = self.links_quaternion.contiguous()
            for i, v in enumerate(self.link_names):
                link_poses[v] = Pose(link_pos[..., i, :], link_quat[..., i, :])
        return link_poses


class CudaRobotModel(CudaRobotModelConfig):
    """
    CUDA Accelerated Robot Model

    Currently dof is created only for links that we need to compute kinematics. E.g., for robots
    with many serial chains, add all links of the robot to get the correct dof. This is not an
    issue if you are loading collision spheres as that will cover the full geometry of the robot.
    """

    def __init__(self, config: CudaRobotModelConfig):
        super().__init__(**vars(config))
        self._batch_size = 0
        self.update_batch_size(1, reset_buffers=True)

    def update_batch_size(self, batch_size, force_update=False, reset_buffers=False):
        if batch_size == 0:
            log_error("batch size is zero")
        if force_update and self._batch_size == batch_size and self.compute_jacobian:
            self.lin_jac = self.lin_jac.detach()  # .requires_grad_(True)
            self.ang_jac = self.ang_jac.detach()  # .requires_grad_(True)
        elif self._batch_size != batch_size or reset_buffers:
            self._batch_size = batch_size
            self._link_pos_seq = torch.zeros(
                (self._batch_size, len(self.link_names), 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._link_quat_seq = torch.zeros(
                (self._batch_size, len(self.link_names), 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )

            self._batch_robot_spheres = torch.zeros(
                (self._batch_size, self.kinematics_config.total_spheres, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.collision_geometry_dtype,
            )
            self._grad_out_q = torch.zeros(
                (self._batch_size, self.get_dof()),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._global_cumul_mat = torch.zeros(
                (self._batch_size, self.kinematics_config.link_map.shape[0], 4, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            if self.compute_jacobian:
                self.lin_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self.ang_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

    @profiler.record_function("cuda_robot_model/forward_kinematics")
    def forward(self, q, link_name=None, calculate_jacobian=True, robot_jac=None):
        # pos, rot = self.compute_forward_kinematics(q, qd, link_name)
        if len(q.shape) > 2:
            raise ValueError("q shape should be [batch_size, dof]")
        batch_size = q.shape[0]
        self.update_batch_size(batch_size, force_update=q.requires_grad)

        # do fused forward:
        link_pos_seq, link_quat_seq, link_spheres_tensor = self._cuda_forward(q)

        # print("_________________________________", self._cuda_backward(q))

        if len(self.link_names) == 1:
            ee_pos = link_pos_seq.squeeze(1)
            ee_quat = link_quat_seq.squeeze(1)
        else:
            link_idx = self.kinematics_config.ee_idx
            if link_name is not None:
                link_idx = self.link_names.index(link_name)
            ee_pos = link_pos_seq.contiguous()[..., link_idx, :]
            ee_quat = link_quat_seq.contiguous()[..., link_idx, :]
        lin_jac = ang_jac = None

        # print("___debug robot_jac___: ", robot_jac)

        # compute jacobians?
        if calculate_jacobian and robot_jac is not None:
            # link_idx = self.kinematics_config.ee_idx

            q_grad = torch.tensor(q, dtype=torch.float32, device="cuda", requires_grad=True)
            # q_grad = torch.rand(2, 7, dtype=torch.float32, device="cuda", requires_grad=True)

            lin_jac, ang_jac = robot_jac.get_jacobian(q_grad, link_pos_seq, link_quat_seq)
            # lin_jac, ang_jac = robot_jac.get_jacobian_test(q, link_pos_seq, link_quat_seq)

            # start_time = time.time()
            # hessian = robot_jac.cal_hessian(q_grad)
            # end_time = time.time()

            # # Calculate the elapsed time
            # elapsed_time = end_time - start_time
            #
            # print("Hessian tensor calculation time: {:.4f} seconds".format(elapsed_time))
            # print(lin_jac, ang_jac, hessian)


            # raise NotImplementedError
        return (
            ee_pos,
            ee_quat,
            lin_jac,
            ang_jac,
            link_pos_seq,
            link_quat_seq,
            link_spheres_tensor,
        )

    def get_state(self, q, link_name=None, calculate_jacobian=False) -> CudaRobotModelState:
        out = self.forward(q, link_name, calculate_jacobian)
        state = CudaRobotModelState(
            out[0],
            out[1],
            None,
            None,
            out[4],
            out[5],
            out[6],
            self.link_names,
        )
        return deepcopy(state)

    def get_robot_link_meshes(self):
        m_list = [self.get_link_mesh(l) for l in self.kinematics_config.mesh_link_names]

        return m_list

    def get_robot_as_mesh(self, q: torch.Tensor):
        # get all link meshes:
        m_list = self.get_robot_link_meshes()
        pose = self.get_link_poses(q, self.kinematics_config.mesh_link_names)
        for li, l in enumerate(self.kinematics_config.mesh_link_names):
            m_list[li].pose = (
                pose.get_index(0, li).multiply(Pose.from_list(m_list[li].pose)).tolist()
            )

        return m_list

    def get_robot_as_spheres(self, q: torch.Tensor, filter_valid: bool = True):
        state = self.get_state(q)

        # state has sphere position and radius

        sph_all = state.get_link_spheres().cpu().numpy()

        sph_traj = []
        for j in range(sph_all.shape[0]):
            sph = sph_all[j, :, :]
            if filter_valid:
                sph_list = [
                    Sphere(
                        name="robot_curobo_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                    if (sph[i, 3] > 0.0)
                ]
            else:
                sph_list = [
                    Sphere(
                        name="robot_curobo_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                ]
            sph_traj.append(sph_list)
        return sph_traj

    def get_link_poses(self, q: torch.Tensor, link_names: List[str]) -> Pose:
        state = self.get_state(q)
        position = torch.zeros(
            (q.shape[0], len(link_names), 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        quaternion = torch.zeros(
            (q.shape[0], len(link_names), 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        for li, l in enumerate(link_names):
            i = self.link_names.index(l)
            position[:, li, :] = state.links_position[:, i, :]
            quaternion[:, li, :] = state.links_quaternion[:, i, :]
        return Pose(position=position, quaternion=quaternion)

    def _cuda_forward(self, q):
        link_pos, link_quat, robot_spheres = get_cuda_kinematics(
            self._link_pos_seq,
            self._link_quat_seq,
            self._batch_robot_spheres,
            self._global_cumul_mat,
            q,
            self.kinematics_config.fixed_transforms,
            self.kinematics_config.link_spheres,
            self.kinematics_config.link_map,  # tells which link is attached to which link i
            self.kinematics_config.joint_map,  # tells which joint is attached to a link i
            self.kinematics_config.joint_map_type,  # joint type
            self.kinematics_config.store_link_map,
            self.kinematics_config.link_sphere_idx_map,  # sphere idx map
            self.kinematics_config.link_chain_map,
            self.kinematics_config.joint_offset_map,
            self._grad_out_q,
            self.use_global_cumul,
        )
        return link_pos, link_quat, robot_spheres

    def _cuda_backward(self, q):
        out = get_cuda_jacobian()
        return out

    @property
    def all_articulated_joint_names(self):
        return self.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self) -> SelfCollisionKinematicsConfig:
        return self.self_collision_config

    def get_link_mesh(self, link_name: str) -> Mesh:
        mesh = self.kinematics_parser.get_link_mesh(link_name)
        return mesh

    def get_link_transform(self, link_name: str) -> Pose:
        mat = self._kinematics_config.fixed_transforms[self._name_to_idx_map[link_name]]
        pose = Pose(position=mat[:3, 3], rotation=mat[:3, :3])
        return pose

    def get_all_link_transforms(self) -> Pose:
        pose = Pose(
            self.kinematics_config.fixed_transforms[:, :3, 3],
            rotation=self.kinematics_config.fixed_transforms[:, :3, :3],
        )
        return pose

    def get_dof(self) -> int:
        return self.kinematics_config.n_dof

    @property
    def dof(self) -> int:
        return self.kinematics_config.n_dof

    @property
    def joint_names(self) -> List[str]:
        return self.kinematics_config.joint_names

    @property
    def total_spheres(self) -> int:
        return self.kinematics_config.total_spheres

    @property
    def lock_jointstate(self):
        return self.kinematics_config.lock_jointstate

    def get_full_js(self, js: JointState):
        all_joint_names = self.all_articulated_joint_names
        lock_joint_state = self.lock_jointstate

        new_js = js.get_augmented_joint_state(all_joint_names, lock_joint_state)
        return new_js

    def get_mimic_js(self, js: JointState):
        if self.kinematics_config.mimic_joints is None:
            return None
        extra_joints = {"position": [], "joint_names": []}
        # for every joint in mimic_joints, get active joint name
        for j in self.kinematics_config.mimic_joints:
            active_q = js.position[..., js.joint_names.index(j)]
            for k in self.kinematics_config.mimic_joints[j]:
                extra_joints["joint_names"].append(k["joint_name"])
                extra_joints["position"].append(
                    k["joint_offset"][0] * active_q + k["joint_offset"][1]
                )
        extra_js = JointState.from_position(
            position=torch.stack(extra_joints["position"]), joint_names=extra_joints["joint_names"]
        )
        new_js = js.get_augmented_joint_state(js.joint_names + extra_js.joint_names, extra_js)
        return new_js

    @property
    def ee_link(self):
        return self.kinematics_config.ee_link

    @property
    def base_link(self):
        return self.kinematics_config.base_link

    @property
    def robot_spheres(self):
        return self.kinematics_config.link_spheres

    def update_kinematics_config(self, new_kin_config: KinematicsTensorConfig):
        self.kinematics_config.copy_(new_kin_config)

    @property
    def retract_config(self):
        return self.kinematics_config.cspace.retract_config


class TorchJacobian():
    # def __init__(self, urdf_file: str, ee_link: str):
    def __init__(self):
        config_file = load_yaml(join_path(get_robot_configs_path(), "dual_ur10e.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"][
            "urdf_path"
        ]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        # for dual_ur10e
        ee_link_1 = "tool1"
        self.chain_cpu_1 = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/"+urdf_file).read(), ee_link_1)

        self.chain_cpu = pk.build_serial_chain_from_urdf(open("../src/curobo/content/assets/"+urdf_file).read(), ee_link)

        d = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        self.chain = self.chain_cpu.to(dtype=dtype, device=d)
        self.chain_1 = self.chain_cpu_1.to(dtype=dtype, device=d)

        self.transform = pk.transforms.Transform3d(device=d, dtype=dtype)
        # self.transform = transforms.Transform3d(device=d, dtype=dtype)
        self.j_w = torch.rand(100, 6, 7, device=d, dtype=dtype)


    def get_jacobian(self, q: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        # print(ee_pos, ee_quat, self.transform.translate(ee_pos))
        # self.transform = self.transform.translate(ee_pos)
        J = self.chain.jacobian(q)
        # lin_jac = ang_jac = None

        # hessian = J[1].backward()
        # print("_____________Hessian__________", hessian)

        # print("before split")
        #
        lin_jac, ang_jac = torch.split(J, 3, dim=1)
        # lin_jac, ang_jac = torch.split(self.j_w, 3, dim=1)

        #
        # print("after split")
        if lin_jac is not None and ang_jac is not None:
            print("___debug lin_jac, ang_jac___", lin_jac.shape, ang_jac.shape)

        return lin_jac, ang_jac
        # return J

    def get_jacobian_test(self, q: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        print("__________require grad____________", q.requires_grad)

        q_grad = q.requires_grad_()

        fk_result = self.chain.forward_kinematics(q_grad, end_only=True)

        m = fk_result.get_matrix()
        pos = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])

        lin_jac = []

        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                print(pos[i, j], q_grad[i])
                jac_tmp = grad(pos[i, j], q_grad, retain_graph=True)[0][i]
                lin_jac.append(jac_tmp)

        # lin_jac = pos.backward()
        print("_______tcp pose______", lin_jac)

        J = self.chain.jacobian(q, self.transform)

        print("_______tcp pose______", J[:, :3, :], lin_jac)

        jacobian = []

        for i in range(fk_result):
            jacobian.append(grad(fk_result[:, i], q, retain_graph=True, create_graph=True)[0])

        jacobian = torch.stack(jacobian, dim=-1)
        return jacobian

    def cal_hessian(self, q: torch.Tensor) -> torch.Tensor:
        # q_grad = q.requires_grad_(True)
        # q_grad = q[1].requires_grad_(True)
        J = self.chain.jacobian(q)
        # print("jacobian require_grad", q.requires_grad, J.requires_grad)
        hessian = torch.zeros(J.shape[0], J.shape[1], J.shape[2], J.shape[2], device="cuda", dtype=torch.float32)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                for k in range(J.shape[2]):
                    # print("hessian", i, j, k, torch.autograd.grad(J[i, j, k], q, retain_graph=True)[0][i])
                    hessian[i, j, k, :] = torch.autograd.grad(J[i, j, k], q, retain_graph=True)[0][i]

        # q_cpu = q.cpu().detach().numpy()
        # J_cpu = J.cpu().detach().numpy()
        #
        #
        #
        # hessian_list = []
        # for i in range(q.shape[0]):
        #
        #     g = O.approx_fprime(J, self.objective_function, 7 * [0.001])
        #     hessian_list.append(g)
        #
        # print(hessian_list)

        return hessian

    def objective_function(self, J):
        jT = np.transpose(J)
        jjT = np.dot(J, jT)
        det = abs(np.linalg.det(jjT))
        manipulability = math.sqrt(det)

        return float(manipulability)

    # def calc_hessian(self, q, tool=None):
    #     """
    #     Return the Hessian of the robot's end-effector pose with respect to the joint variables.
    #     The Hessian is a tensor of shape (N, 6, DOF, DOF), where N is the batch size, 6 represents
    #     the end-effector's position and orientation, and DOF is the number of joint variables.
    #
    #     tool is the transformation wrt the end effector; default is identity. If specified, will have to
    #     specify for each of the N inputs
    #     """
    #     if len(q.shape) <= 1:
    #         N = 1
    #         th = q.reshape(1, -1)
    #     else:
    #         N = q.shape[0]
    #     ndof = q.shape[1]
    #
    #     h_eef = torch.zeros((N, 6, ndof, ndof), dtype=self.chain.dtype, device=self.chain.device)
    #
    #     if tool is None:
    #         cur_transform = transforms.Transform3d(device=self.chain.device,
    #                                                dtype=self.chain.dtype).get_matrix().repeat(N, 1, 1)
    #     else:
    #         if tool.dtype != self.chain.dtype or tool.device != self.chain.device:
    #             tool = tool.to(device=self.chain.device, copy=True, dtype=self.chain.dtype)
    #         cur_transform = tool.get_matrix()
    #
    #     for i in range(ndof):
    #         for j in range(i, ndof):
    #             cur_transform_i = cur_transform.clone()
    #             cur_transform_j = cur_transform.clone()
    #
    #             for k, f in enumerate(reversed(self.chain._serial_frames)):
    #                 if f.joint.joint_type == "revolute":
    #                     axis_i = cur_transform_i[:, :3, :3].transpose(1, 2) @ f.joint.axis
    #                     eef2joint_pos_i = cur_transform_i[:, :3, 3].unsqueeze(2)
    #                     joint2eef_rot_i = cur_transform_i[:, :3, :3].transpose(1, 2)
    #                     eef2joint_pos_in_eef_i = joint2eef_rot_i @ eef2joint_pos_i
    #
    #                     axis_j = cur_transform_j[:, :3, :3].transpose(1, 2) @ f.joint.axis
    #                     eef2joint_pos_j = cur_transform_j[:, :3, 3].unsqueeze(2)
    #                     joint2eef_rot_j = cur_transform_j[:, :3, :3].transpose(1, 2)
    #                     eef2joint_pos_in_eef_j = joint2eef_rot_j @ eef2joint_pos_j
    #
    #                     if i == j:
    #                         h_eef[:, :3, i, j] = torch.cross(axis_i, eef2joint_pos_in_eef_i.squeeze(2), dim=1)
    #                         h_eef[:, 3:, i, j] = axis_i
    #                     else:
    #                         d2x_dqjdqi = torch.cross(axis_j,
    #                                                  torch.cross(axis_i, eef2joint_pos_in_eef_i.squeeze(2), dim=1),
    #                                                  dim=1)
    #                         h_eef[:, :3, i, j] = d2x_dqjdqi.squeeze()
    #                         h_eef[:, 3:, i, j] = torch.cross(axis_j, axis_i, dim=1).squeeze()
    #
    #                 elif f.joint.joint_type == "prismatic":
    #                     if i == j:
    #                         h_eef[:, :3, i, j] = f.joint.axis.repeat(N, 1) @ cur_transform_i[:, :3, :3]
    #
    #                 cur_frame_transform_i = f.get_transform(q[:, i]).get_matrix()
    #                 cur_frame_transform_j = f.get_transform(q[:, j]).get_matrix()
    #                 cur_transform_i = cur_frame_transform_i @ cur_transform_i
    #                 cur_transform_j = cur_frame_transform_j @ cur_transform_j
    #
    #         # Transform Hessian to base frame
    #     pose = self.chain.forward_kinematics(q).get_matrix()
    #     rotation = pose[:, :3, :3]
    #     h_tr = torch.zeros((N, 6, 6, 6), dtype=self.chain.dtype, device=self.chain.device)
    #     h_tr[:, :3, :3, :3] = rotation.unsqueeze(-1)
    #     h_tr[:, 3:, 3:, 3:] = rotation.unsqueeze(-1)
    #     h_w = h_tr @ h_eef @ h_tr.transpose(-1, -2)
    #
    #     return h_w

    def forward_kinematics(self, q):
        cur_transform = transforms.Transform3d(device=self.chain.device, dtype=self.chain.dtype).get_matrix()
        for f in reversed(self.chain._serial_frames):
            cur_frame_transform = f.get_transform(q).get_matrix()
            cur_transform = cur_frame_transform @ cur_transform
        return cur_transform