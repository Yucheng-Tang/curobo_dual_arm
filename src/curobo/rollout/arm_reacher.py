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
# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.rollout.cost.cost_base import CostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseCostMetric
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.cost.zero_cost import ZeroCost
from curobo.rollout.cost.manipulability_cost import ManipulabilityCost, ManipulabilityCostConfig
from curobo.rollout.cost.trajectory_execution_cost import TrajExecCost, TrajExecCostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.tensor import T_BValue_float, T_BValue_int
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_max
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .arm_base import ArmBase, ArmBaseConfig, ArmCostConfig

import pytorch3d.transforms

import pandas as pd


@dataclass
class ArmReacherMetrics(RolloutMetrics):
    cspace_error: Optional[T_BValue_float] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    pose_error: Optional[T_BValue_float] = None
    goalset_index: Optional[T_BValue_int] = None
    null_space_error: Optional[T_BValue_float] = None
    exec_time: Optional[T_BValue_float] = None

    def __getitem__(self, idx):
        d_list = [
            self.cost,
            self.constraint,
            self.feasible,
            self.state,
            self.cspace_error,
            self.position_error,
            self.rotation_error,
            self.pose_error,
            self.goalset_index,
            self.null_space_error,
            self.exec_time,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return ArmReacherMetrics(*idx_vals)

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return ArmReacherMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
            cspace_error=None if self.cspace_error is None else self.cspace_error.clone(),
            position_error=None if self.position_error is None else self.position_error.clone(),
            rotation_error=None if self.rotation_error is None else self.rotation_error.clone(),
            pose_error=None if self.pose_error is None else self.pose_error.clone(),
            goalset_index=None if self.goalset_index is None else self.goalset_index.clone(),
            null_space_error=(
                None if self.null_space_error is None else self.null_space_error.clone()
            ),
            exec_time=None if self.exec_time is None else self.exec_time.clone(),
        )


@dataclass
class ArmReacherCostConfig(ArmCostConfig):
    pose_cfg: Optional[PoseCostConfig] = None
    cspace_cfg: Optional[DistCostConfig] = None
    straight_line_cfg: Optional[CostConfig] = None
    zero_acc_cfg: Optional[CostConfig] = None
    zero_vel_cfg: Optional[CostConfig] = None
    zero_jerk_cfg: Optional[CostConfig] = None
    link_pose_cfg: Optional[PoseCostConfig] = None
    manipulability_cfg: Optional[CostConfig] = None
    stiffness_cfg: Optional[CostConfig] = None
    relative_pose_cfg: Optional[PoseCostConfig] = None
    traj_exec_cfg: Optional[TrajExecCostConfig] = None

    @staticmethod
    def _get_base_keys():
        base_k = ArmCostConfig._get_base_keys()
        # add new cost terms:
        new_k = {
            "pose_cfg": PoseCostConfig,
            "cspace_cfg": DistCostConfig,
            "straight_line_cfg": CostConfig,
            "zero_acc_cfg": CostConfig,
            "zero_vel_cfg": CostConfig,
            "zero_jerk_cfg": CostConfig,
            "link_pose_cfg": PoseCostConfig,
            "manipulability_cfg": ManipulabilityCostConfig,
            "stiffness_cfg": ManipulabilityCostConfig,
            "relative_pose_cfg": PoseCostConfig,
            "traj_exec_cfg": TrajExecCostConfig,
        }
        new_k.update(base_k)
        return new_k

    @staticmethod
    def from_dict(
            data_dict: Dict,
            robot_cfg: RobotConfig,
            world_coll_checker: Optional[WorldCollision] = None,
            tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        k_list = ArmReacherCostConfig._get_base_keys()
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        return ArmReacherCostConfig(**data)


@dataclass
class ArmReacherConfig(ArmBaseConfig):
    cost_cfg: ArmReacherCostConfig
    constraint_cfg: ArmReacherCostConfig
    convergence_cfg: ArmReacherCostConfig

    @staticmethod
    def cost_from_dict(
            cost_data_dict: Dict,
            robot_cfg: RobotConfig,
            world_coll_checker: Optional[WorldCollision] = None,
            tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return ArmReacherCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )


@get_torch_jit_decorator()
def _compute_g_dist_jit(rot_err_norm, goal_dist):
    # goal_cost = goal_cost.view(cost.shape)
    # rot_err_norm = rot_err_norm.view(cost.shape)
    # goal_dist = goal_dist.view(cost.shape)
    g_dist = goal_dist.unsqueeze(-1) + 10.0 * rot_err_norm.unsqueeze(-1)
    return g_dist


class ArmReacher(ArmBase, ArmReacherConfig):
    """
    .. inheritance-diagram:: curobo.rollout.arm_reacher.ArmReacher
    """

    @profiler.record_function("arm_reacher/init")
    def __init__(self, config: Optional[ArmReacherConfig] = None):
        if config is not None:
            ArmReacherConfig.__init__(self, **vars(config))
        ArmBase.__init__(self)

        # self.goal_state = None
        # self.goal_ee_pos = None
        # self.goal_ee_rot = None
        # self.goal_ee_quat = None
        self._compute_g_dist = False
        self._n_goalset = 1

        if self.cost_cfg.cspace_cfg is not None:
            self.cost_cfg.cspace_cfg.dof = self.d_action
            # self.cost_cfg.cspace_cfg.update_vec_weight(self.dynamics_model.cspace_distance_weight)
            self.dist_cost = DistCost(self.cost_cfg.cspace_cfg)
        if self.cost_cfg.relative_pose_cfg is not None:
            print("______________relative pose cfg_______________")
            # print(self.cost_cfg.relative_pose_cfg)
            self.goal_cost = PoseCost(self.cost_cfg.relative_pose_cfg)

        else:  # standard Curobo
            if self.cost_cfg.pose_cfg is not None:
                self.goal_cost = PoseCost(self.cost_cfg.pose_cfg)
                if self.cost_cfg.link_pose_cfg is None:
                    log_info(
                        "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                    )
                    self.cost_cfg.link_pose_cfg = self.cost_cfg.pose_cfg
            self._link_pose_costs = {}

            if self.cost_cfg.link_pose_cfg is not None:
                for i in self.kinematics.link_names:
                    if i != self.kinematics.ee_link:
                        self._link_pose_costs[i] = PoseCost(self.cost_cfg.link_pose_cfg)
        if self.cost_cfg.straight_line_cfg is not None:
            self.straight_line_cost = StraightLineCost(self.cost_cfg.straight_line_cfg)
        if self.cost_cfg.zero_vel_cfg is not None:
            self.zero_vel_cost = ZeroCost(self.cost_cfg.zero_vel_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_vel_cost.hinge_value is not None:
                self._compute_g_dist = True
        if self.cost_cfg.zero_acc_cfg is not None:
            self.zero_acc_cost = ZeroCost(self.cost_cfg.zero_acc_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_acc_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.zero_jerk_cfg is not None:
            self.zero_jerk_cost = ZeroCost(self.cost_cfg.zero_jerk_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_jerk_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.manipulability_cfg is not None:
            self.manipulability_cost = ManipulabilityCost(self.cost_cfg.manipulability_cfg)

        if self.cost_cfg.traj_exec_cfg is not None:
            self.traj_exec_cost = TrajExecCost(self.cost_cfg.traj_exec_cfg)

        self.z_tensor = torch.tensor(
            0, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_pose_convergence = {}

        # TODO: add convergence cost for relative pose
        if self.convergence_cfg.pose_cfg is not None:
            self.pose_convergence = PoseCost(self.convergence_cfg.pose_cfg)
            if self.convergence_cfg.link_pose_cfg is None:
                log_warn(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.convergence_cfg.link_pose_cfg = self.convergence_cfg.pose_cfg

        if self.convergence_cfg.link_pose_cfg is not None:
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_convergence[i] = PoseCost(self.convergence_cfg.link_pose_cfg)
        if self.convergence_cfg.cspace_cfg is not None:
            self.convergence_cfg.cspace_cfg.dof = self.d_action
            self.cspace_convergence = DistCost(self.convergence_cfg.cspace_cfg)

        # check if g_dist is required in any of the cost terms:
        self.update_params(Goal(current_state=self._start_state))

    def cost_fn(self, state: KinematicModelState, action_batch=None, robot_jac=None, q_init=None):
        """
        Compute cost given that state dictionary and actions


        :class:`curobo.rollout.cost.PoseCost`
        :class:`curobo.rollout.cost.DistCost`

        """
        state_batch = state.state_seq
        with profiler.record_function("cost/base"):
            cost_list = super(ArmReacher, self).cost_fn(state, action_batch, return_list=True, robot_jac=robot_jac, q_init=q_init)
        ee_pos_batch, ee_quat_batch = state.ee_pos_seq, state.ee_quat_seq
        g_dist = None
        # px = pd.DataFrame(state_batch[0].position.cpu().detach().numpy())
        if self.cost_cfg.relative_pose_cfg is not None:
            with (((profiler.record_function("cost/relative_pose")))):
                if (
                        self._goal_buffer.goal_pose.position is not None
                        and self.cost_cfg.relative_pose_cfg is not None
                        and self.goal_cost.enabled
                ):
                    # print("__________goal buffer__________",  self._goal_buffer)
                    link_poses = state.link_pose

                    # TODO: only support 2 TCP relative pose and check the calculation is correct
                    for k in self._goal_buffer.links_goal_pose.keys():
                        if k != self.kinematics.ee_link:
                            # get link pose
                            relative_base_pose = link_poses[k].contiguous()
                            relative_base_pos = relative_base_pose.position
                            relative_base_quat = relative_base_pose.quaternion
                    # print("_______________current pose ___________________", ee_pos_batch, ee_quat_batch, relative_base_pos, relative_base_quat)

                    # ee_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(ee_quat_batch)
                    # base_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(relative_base_quat)
                    #
                    # base_matrix_batch_inv = base_matrix_batch.transpose(-2, -1)
                    #
                    # # Compute the inverse of the translation vector
                    # base_pos_inv = -torch.matmul(base_matrix_batch_inv, relative_base_pos.unsqueeze(-1)).squeeze(-1)
                    #
                    # relative_rot_batch = torch.matmul(base_matrix_batch_inv, ee_matrix_batch)
                    # relative_pos_batch = torch.matmul(base_matrix_batch_inv, ee_pos_batch.unsqueeze(-1)).squeeze(-1)
                    #
                    # relative_ee_pos_batch = base_pos_inv + relative_pos_batch
                    #
                    # relative_ee_quat_batch = pytorch3d.transforms.matrix_to_quaternion(relative_rot_batch)
                    #
                    # print("_______________goal buffer_________________", self._goal_buffer)
                    # print("_______________current relative pose ___________________", relative_ee_pos_batch, relative_ee_quat_batch)

                    relative_ee_pos_batch, relative_ee_quat_batch = calculate_relative_pose(relative_base_pos,
                                                                                            relative_base_quat,
                                                                                            ee_pos_batch, ee_quat_batch)

                    # print("_______________test relative pose ___________________", relative_ee_pos_batch,
                    #       relative_ee_quat_batch)
                    # print("_______________goal buffer_________________", self._goal_buffer)

                    # relative_goal_buffer = self._goal_buffer.detach().clone()
                    # (
                    #     relative_goal_buffer.goal_pose.position,
                    #     relative_goal_buffer.goal_pose.quaternion) = calculate_relative_pose(
                    #     self._goal_buffer.links_goal_pose["tool1"].position,
                    #     self._goal_buffer.links_goal_pose["tool1"].quaternion,
                    #     self._goal_buffer.goal_pose.position,
                    #     self._goal_buffer.goal_pose.quaternion)
                    # print("_______________relative goal buffer_________________", relative_goal_buffer)

                    # relative_ee_pos_batch_world = ee_pos_batch - current_pos
                    # current_quat_inv = quaternion_inverse(current_quat)
                    # current_quat_inv = normalize_quaternion(current_quat_inv)
                    # relative_ee_pos_batch = quaternion_apply(current_quat_inv, relative_ee_pos_batch_world)
                    #
                    # relative_ee_quat_batch = quaternion_multiply(current_quat_inv, ee_quat_batch)
                    # relative_ee_quat_batch = normalize_quaternion(relative_ee_quat_batch)
                    # print("_______________current relative pose ___________________", relative_ee_pos_batch_world, relative_ee_pos_batch, relative_ee_quat_batch)

                    if self._compute_g_dist:
                        # print("___________compute_g_dist_______________")
                        goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                            relative_ee_pos_batch,
                            relative_ee_quat_batch,
                            self._goal_buffer,  # self._goal_buffer
                            "relative_pose"
                        )

                        g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                    else:
                        # print("___________doesnt compute_g_dist_______________")
                        goal_cost = self.goal_cost.forward(
                            relative_ee_pos_batch, relative_ee_quat_batch, self._goal_buffer, "relative_pose"
                        )  # self._goal_buffer
                    # print(self._compute_g_dist, goal_cost.view(-1))
                    # px.insert(-1, "goal_cost", goal_cost.view(-1))
                    # px.insert(0, "goal_cost", goal_cost[0].cpu().detach().numpy())

                    # print("_______________goal cost_____________", px)
                    cost_list.append(goal_cost)
        else:
            with profiler.record_function("cost/pose"):
                if (
                        self._goal_buffer.goal_pose.position is not None
                        and self.cost_cfg.pose_cfg is not None
                        and self.goal_cost.enabled
                ):
                    if self._compute_g_dist:
                        goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                            ee_pos_batch,
                            ee_quat_batch,
                            self._goal_buffer,
                        )

                        g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                    else:
                        goal_cost = self.goal_cost.forward(
                            ee_pos_batch, ee_quat_batch, self._goal_buffer
                        )
                    # print(self._compute_g_dist, goal_cost.view(-1))
                    # px.insert(0, "goal_cost", goal_cost[0].cpu().detach().numpy())

                    cost_list.append(goal_cost)
            with profiler.record_function("cost/link_poses"):
                if self._goal_buffer.links_goal_pose is not None and self.cost_cfg.pose_cfg is not None:
                    link_poses = state.link_pose

                    for k in self._goal_buffer.links_goal_pose.keys():
                        if k != self.kinematics.ee_link:
                            current_fn = self._link_pose_costs[k]
                            if current_fn.enabled:
                                # get link pose
                                current_pose = link_poses[k].contiguous()
                                current_pos = current_pose.position
                                current_quat = current_pose.quaternion

                                c = current_fn.forward(current_pos, current_quat, self._goal_buffer, k)

                                # px.insert(0, "link_cost", c[0].cpu().detach().numpy())

                                cost_list.append(c)

        if (
                self._goal_buffer.goal_state is not None
                and self.cost_cfg.cspace_cfg is not None
                and self.dist_cost.enabled
        ):
            joint_cost = self.dist_cost.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state_batch.position,
                self._goal_buffer.batch_goal_state_idx,
            )
            cost_list.append(joint_cost)
        if self.cost_cfg.straight_line_cfg is not None and self.straight_line_cost.enabled:
            st_cost = self.straight_line_cost.forward(ee_pos_batch)
            cost_list.append(st_cost)

        if (
                self.cost_cfg.zero_acc_cfg is not None
                and self.zero_acc_cost.enabled
                # and g_dist is not None
        ):
            z_acc = self.zero_acc_cost.forward(
                state_batch.acceleration,
                g_dist,
            )

            cost_list.append(z_acc)
        if self.cost_cfg.zero_jerk_cfg is not None and self.zero_jerk_cost.enabled:
            z_jerk = self.zero_jerk_cost.forward(
                state_batch.jerk,
                g_dist,
            )
            cost_list.append(z_jerk)

        if self.cost_cfg.zero_vel_cfg is not None and self.zero_vel_cost.enabled:
            z_vel = self.zero_vel_cost.forward(
                state_batch.velocity,
                g_dist,
            )
            cost_list.append(z_vel)
        if self.cost_cfg.manipulability_cfg is not None and self.manipulability_cost.enabled:
            jac_batch = torch.cat((state.lin_jac_seq, state.ang_jac_seq), dim=-2)
            ma_cost = self.manipulability_cost.forward(
                jac_batch,
                state_batch.position,
                state_batch.velocity,
                robot_jac=robot_jac,
            )
            cost_list.append(ma_cost)

        with profiler.record_function("cost/traj_exec"):
            if self.cost_cfg.traj_exec_cfg is not None and self.traj_exec_cost.enabled:
                traj_exec_cost = self.traj_exec_cost.forward(
                    state_batch.position,
                    q_init=q_init,
                )
                cost_list.append(traj_exec_cost)
            
                # for pandas debug
                # px.insert(0, "traj_exec_cost", traj_exec_cost[0].cpu().detach().numpy())

        # print("_______________cost list_____________", cost_list)

        with profiler.record_function("cat_sum"):
            if self.sum_horizon:
                cost = cat_sum_horizon_reacher(cost_list)
            else:
                cost = cat_sum_reacher(cost_list)
            # px.insert(0, "sum", cost[0].cpu().detach().numpy())

        # px.to_csv('cost.csv', mode='a', index=False, header=False)

        return cost

    def convergence_fn(
            self, state: KinematicModelState, out_metrics: Optional[ArmReacherMetrics] = None, q_init=None
    ) -> ArmReacherMetrics:
        if out_metrics is None:
            out_metrics = ArmReacherMetrics()
        if not isinstance(out_metrics, ArmReacherMetrics):
            out_metrics = ArmReacherMetrics(**vars(out_metrics))
        # print(self._goal_buffer.batch_retract_state_idx)
        out_metrics = super(ArmReacher, self).convergence_fn(state, out_metrics, q_init=q_init)

        # compute error with pose?
        if self.cost_cfg.relative_pose_cfg is not None:
            # TODO: change relative pose convergence cfg
            if (
                    self._goal_buffer.goal_pose.position is not None
                    and self.convergence_cfg.pose_cfg is not None
            ):
                print("calculate pose error!")
                link_poses = state.link_pose
                # TODO: only support 2 TCP relative pose and check the calculation is correct
                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        # get link pose
                        relative_base_pose = link_poses[k].contiguous()
                        relative_base_pos = relative_base_pose.position
                        relative_base_quat = relative_base_pose.quaternion
                relative_ee_pos_batch, relative_ee_quat_batch = calculate_relative_pose(relative_base_pos,
                                                                                        relative_base_quat,
                                                                                        state.ee_pos_seq, state.ee_quat_seq)

                (
                    out_metrics.pose_error,
                    out_metrics.rotation_error,
                    out_metrics.position_error,
                ) = self.pose_convergence.forward_out_distance(
                    relative_ee_pos_batch, relative_ee_quat_batch, self._goal_buffer, "relative_pose"
                )
                out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
        else:
            if (
                    self._goal_buffer.goal_pose.position is not None
                    and self.convergence_cfg.pose_cfg is not None
            ):
                (
                    out_metrics.pose_error,
                    out_metrics.rotation_error,
                    out_metrics.position_error,
                ) = self.pose_convergence.forward_out_distance(
                    state.ee_pos_seq, state.ee_quat_seq, self._goal_buffer
                )
                out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
            if (
                    self._goal_buffer.links_goal_pose is not None
                    and self.convergence_cfg.pose_cfg is not None
            ):
                pose_error = [out_metrics.pose_error]
                position_error = [out_metrics.position_error]
                quat_error = [out_metrics.rotation_error]
                link_poses = state.link_pose

                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        current_fn = self._link_pose_convergence[k]
                        if current_fn.enabled:
                            # get link pose
                            current_pos = link_poses[k].position
                            current_quat = link_poses[k].quaternion

                            pose_err, pos_err, quat_err = current_fn.forward_out_distance(
                                current_pos, current_quat, self._goal_buffer, k
                            )
                            pose_error.append(pose_err)
                            position_error.append(pos_err)
                            quat_error.append(quat_err)

                out_metrics.pose_error = cat_max(pose_error)
                out_metrics.rotation_error = cat_max(quat_error)
                out_metrics.position_error = cat_max(position_error)

        if (
                self._goal_buffer.goal_state is not None
                and self.convergence_cfg.cspace_cfg is not None
                and self.cspace_convergence.enabled
        ):
            _, out_metrics.cspace_error = self.cspace_convergence.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state.state_seq.position,
                self._goal_buffer.batch_goal_state_idx,
                True,
            )

        if (
                self.convergence_cfg.null_space_cfg is not None
                and self.null_convergence.enabled
                and self._goal_buffer.batch_retract_state_idx is not None
        ):
            out_metrics.null_space_error = self.null_convergence.forward_target_idx(
                self._goal_buffer.retract_state,
                state.state_seq.position,
                self._goal_buffer.batch_retract_state_idx,
            )

        if self.cost_cfg.traj_exec_cfg is not None and self.traj_exec_cost.enabled:
            traj_exec_cost = self.traj_exec_cost.forward(
                state.state_seq.position,
                q_init=q_init,
            )
            out_metrics.exec_time = traj_exec_cost

        return out_metrics

    def update_params(
            self,
            goal: Goal,
    ):
        """
        Update params for the cost terms and dynamics model.

        """

        super(ArmReacher, self).update_params(goal)
        if goal.batch_pose_idx is not None:
            self._goal_idx_update = False
        if goal.goal_pose.position is not None:
            self.enable_cspace_cost(False)
        return True

    def enable_pose_cost(self, enable: bool = True):
        if enable:
            self.goal_cost.enable_cost()
        else:
            self.goal_cost.disable_cost()

    def enable_cspace_cost(self, enable: bool = True):
        if enable:
            self.dist_cost.enable_cost()
            self.cspace_convergence.enable_cost()
        else:
            self.dist_cost.disable_cost()
            self.cspace_convergence.disable_cost()

    def get_pose_costs(self, include_link_pose: bool = False, include_convergence: bool = True):
        pose_costs = [self.goal_cost]
        if include_convergence:
            pose_costs += [self.pose_convergence]
        if include_link_pose:
            log_error("Not implemented yet")
        return pose_costs

    def update_pose_cost_metric(
            self,
            metric: PoseCostMetric,
    ):
        pose_costs = self.get_pose_costs()
        if metric.hold_partial_pose:
            if metric.hold_vec_weight is None:
                log_error("hold_vec_weight is required")
            [x.hold_partial_pose(metric.hold_vec_weight) for x in pose_costs]
        if metric.release_partial_pose:
            [x.release_partial_pose() for x in pose_costs]
        if metric.reach_partial_pose:
            if metric.reach_vec_weight is None:
                log_error("reach_vec_weight is required")
            [x.reach_partial_pose(metric.reach_vec_weight) for x in pose_costs]
        if metric.reach_full_pose:
            [x.reach_full_pose() for x in pose_costs]

        pose_costs = self.get_pose_costs(include_convergence=False)
        if metric.remove_offset_waypoint:
            [x.remove_offset_waypoint() for x in pose_costs]

        if metric.offset_position is not None or metric.offset_rotation is not None:
            [
                x.update_offset_waypoint(
                    offset_position=metric.offset_position,
                    offset_rotation=metric.offset_rotation,
                    offset_tstep_fraction=metric.offset_tstep_fraction,
                )
                for x in pose_costs
            ]


@get_torch_jit_decorator()
def cat_sum_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=(0, -1))
    return cat_tensor


def quaternion_inverse(q):
    q_conjugate = q.clone()
    q_conjugate[:, 1:] = -q_conjugate[:, 1:]
    return q_conjugate / (q.norm(p=2, dim=1, keepdim=True) ** 2)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, :, 0], q1[:, :, 1], q1[:, :, 2], q1[:, :, 3]
    w2, x2, y2, z2 = q2[:, :, 0], q2[:, :, 1], q2[:, :, 2], q2[:, :, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=2)


def normalize_quaternion(quat):
    return quat / quat.norm(p=2, dim=-1, keepdim=True)


def quaternion_apply(quaternion, vector):
    q_w = quaternion[:, :, 0]
    q_vec = quaternion[:, :, 1:]

    uv = torch.cross(q_vec, vector, dim=2)
    uuv = torch.cross(q_vec, uv, dim=2)

    return vector + 2 * (q_w.unsqueeze(1) * uv + uuv)


def calculate_relative_pose(base_pos_batch, base_quat_batch, ee_pos_batch, ee_quat_batch):
    ee_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(ee_quat_batch)
    base_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(base_quat_batch)

    current_matrix_batch_inv = base_matrix_batch.transpose(-2, -1)

    # Compute the inverse of the translation vector
    current_pos_inv = -torch.matmul(current_matrix_batch_inv, base_pos_batch.unsqueeze(-1)).squeeze(-1)

    relative_rot_batch = torch.matmul(current_matrix_batch_inv, ee_matrix_batch)
    relative_pos_batch = torch.matmul(current_matrix_batch_inv, ee_pos_batch.unsqueeze(-1)).squeeze(-1)

    relative_ee_pos_batch = current_pos_inv + relative_pos_batch

    relative_ee_quat_batch = pytorch3d.transforms.matrix_to_quaternion(relative_rot_batch)

    # print("relative pos", relative_ee_pos_batch, "relative quat", relative_ee_quat_batch)

    return relative_ee_pos_batch, relative_ee_quat_batch
