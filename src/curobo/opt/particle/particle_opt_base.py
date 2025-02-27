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
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.opt.opt_base import Optimizer, OptimizerConfig
from curobo.rollout.rollout_base import RolloutBase, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.tensor import T_BHDOF_float, T_HDOF_float
from curobo.util.logger import log_error, log_info

from curobo.cuda_robot_model.cuda_robot_model import TorchJacobian


class SampleMode(Enum):
    MEAN = 0
    BEST = 1
    SAMPLE = 2


@dataclass
class ParticleOptInfo:
    info: Optional[Dict] = None


@dataclass
class ParticleOptConfig(OptimizerConfig):
    gamma: float
    sample_mode: SampleMode
    seed: int
    calculate_value: bool
    store_rollouts: bool

    def __post_init__(self):
        object.__setattr__(self, "action_highs", self.tensor_args.to_device(self.action_highs))
        object.__setattr__(self, "action_lows", self.tensor_args.to_device(self.action_lows))

        if self.calculate_value and self.use_cuda_graph:
            log_error("Cannot calculate_value when cuda graph is enabled")
        return super().__post_init__()

    @staticmethod
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict = OptimizerConfig.create_data_dict(
            data_dict, rollout_fn, tensor_args, child_dict
        )

        child_dict["sample_mode"] = SampleMode[child_dict["sample_mode"]]
        if "calculate_value" not in child_dict:
            child_dict["calculate_value"] = False
        if "store_rollouts" not in child_dict:
            child_dict["store_rollouts"] = False
        return child_dict


class ParticleOptBase(Optimizer, ParticleOptConfig):
    """Base class for sampling based controllers."""

    @profiler.record_function("particle_opt/init")
    def __init__(
        self,
        config: Optional[ParticleOptConfig] = None,
    ):
        if config is not None:
            super().__init__(**vars(config))
        Optimizer.__init__(self)
        self.gamma_seq = torch.cumprod(
            torch.tensor([1.0] + [self.gamma] * (self.horizon - 1)), dim=0
        ).reshape(1, self.horizon)
        self.gamma_seq = self.tensor_args.to_device(self.gamma_seq)
        self.num_steps = 0
        self.trajectories = None
        self.cu_opt_init = False
        self.info = ParticleOptInfo()
        self.update_num_particles_per_problem(self.num_particles)

    @abstractmethod
    def _get_action_seq(self, mode=SampleMode):
        """
        Get action sequence to execute on the system based
        on current control distribution

        Args:
            mode : {'mean', 'sample'}
                how to choose action to be executed
                'mean' plays mean action and
                'sample' samples from the distribution
        """
        pass

    @abstractmethod
    def sample_actions(self, init_act: T_BHDOF_float):
        """
        Sample actions from current control distribution
        """
        raise NotImplementedError("sample_actions funtion not implemented")

    def update_seed(self, init_act):
        raise NotImplementedError

    @abstractmethod
    def _update_distribution(self, trajectories: Trajectory):
        """
        Update current control distribution using
        rollout trajectories

        Args:
            trajectories : dict
                Rollout trajectories. Contains the following fields
                observations : torch.tensor
                    observations along rollouts
                actions : torch.tensor
                    actions sampled from control distribution along rollouts
                costs : torch.tensor
                    step costs along rollouts
        """
        pass

    def reset(self):
        """
        Reset the optimizer
        """
        self.num_steps = 0

        # self.rollout_fn.reset()
        super().reset()

    @abstractmethod
    def _calc_val(self, trajectories: Trajectory):
        """
        Calculate value of state given
        rollouts from a policy
        """
        pass

    def check_convergence(self):
        """
        Checks if controller has converged
        Returns False by default
        """
        return False

    def generate_rollouts(self, init_act=None, robot_jac=None, q_init=None):
        """
        Samples a batch of actions, rolls out trajectories for each particle
        and returns the resulting observations, costs,
        actions

        Parameters
        ----------
        state : dict or np.ndarray
            Initial state to set the simulation problem to
        """
        # print("[particle_opt_base] init_act: ",init_act)

        act_seq = self.sample_actions(init_act)

        # print("[particle_opt_base] act_seq: ", act_seq.shape)
        trajectories = self.rollout_fn(act_seq, robot_jac, q_init)

        # print("[particle_opt_base] trajectories: ", trajectories.actions.shape)
        return trajectories

    def _optimize(self, init_act: torch.Tensor, shift_steps=0, n_iters=None, robot_jac: TorchJacobian=None, q_init: torch.Tensor=None):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from

        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action

        Returns
        -------
        action : torch.Tensor
            next action to execute
        value: float
            optimal value estimate (default: 0.)
        info: dict
            dictionary with side-information
        """

        log_info("using MPPI optimizer!")
        n_iters = n_iters if n_iters is not None else self.n_iters

        # create cuda graph:
        if self.use_cuda_graph and self.cu_opt_init:
            log_info("using MPPI optimizer! After warming up!")
            curr_action_seq = self._call_cuda_opt_iters(init_act)
        else:
            log_info("using MPPI optimizer! warming up!")

            curr_action_seq = self._run_opt_iters(
                init_act, n_iters=n_iters, shift_steps=shift_steps, robot_jac=robot_jac, q_init=q_init
            )
        if self.use_cuda_graph:
            if not self.cu_opt_init:
                log_info("using MPPI optimizer! create cuda graph!")
                self._initialize_cuda_graph(init_act, shift_steps=shift_steps, robot_jac=robot_jac, q_init=q_init)
        
        # print('curr_action_seq for MPPI', curr_action_seq)

        print("[particle opt base] optimization finished!")

        print(self.num_steps)
        self.num_steps += 1
        if self.calculate_value:
            trajectories = self.generate_rollouts(init_act, robot_jac=robot_jac, q_init=q_init)
            value = self._calc_val(trajectories)
            self.info["value"] = value
        # print(self.act_seq)
        return curr_action_seq

    def _initialize_cuda_graph(self, init_act: T_HDOF_float, shift_steps=0, robot_jac: TorchJacobian=None, q_init: torch.Tensor=None):
        log_info("ParticleOptBase: Creating Cuda Graph")
        self._cu_act_in = init_act.detach().clone()

        # create a new stream:
        s = torch.cuda.Stream(device=self.tensor_args.device)
        s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                print("[particle opt base] run opt iters 3 times!!!!!!!")
                self._cu_act_seq = self._run_opt_iters(self._cu_act_in, shift_steps=shift_steps, robot_jac=robot_jac)
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)

        self.reset()
        self.cu_opt_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.cu_opt_graph, stream=s):
            print("[particle opt base] getting _cu_act_seq!")
            self._cu_act_seq = self._run_opt_iters(self._cu_act_in, shift_steps=shift_steps, robot_jac=robot_jac, q_init=q_init)
        self.cu_opt_init = True

    def _call_cuda_opt_iters(self, init_act: T_HDOF_float):
        self._cu_act_in.copy_(init_act.detach())
        self.cu_opt_graph.replay()
        return self._cu_act_seq.detach().clone()  # .clone()

    def _run_opt_iters(self, init_act: T_HDOF_float, shift_steps=0, n_iters=None, robot_jac: TorchJacobian=None, q_init: torch.Tensor=None):
        n_iters = n_iters if n_iters is not None else self.n_iters

        self._shift(shift_steps)
        self.update_seed(init_act)
        if not self.use_cuda_graph and self.store_debug:
            self.debug.append(self._get_action_seq(mode=self.sample_mode).clone())

        # robot_jac = TorchJacobian()
        #
        # q_grad = torch.rand(2, 7, dtype=torch.float32, device="cuda", requires_grad=True)
        # link_pos_seq = torch.rand(2, 6, dtype=torch.float32, device="cuda", requires_grad=True)
        # link_quat_seq = torch.rand(2, 6, dtype=torch.float32, device="cuda", requires_grad=True)
        #
        # lin_jac, ang_jac = robot_jac.get_jacobian(q_grad, link_pos_seq, link_quat_seq)

        # print("_run_opt_iters: n_iters ", n_iters)
        for i in range(n_iters):
            # print("generate_rollouts ", i)
            # generate random simulated trajectories

            trajectory = self.generate_rollouts(robot_jac=robot_jac, q_init=q_init)

            trajectory.actions = trajectory.actions.view(
                self.n_problems, self.particles_per_problem, self.action_horizon, self.d_action
            )
            trajectory.costs = trajectory.costs.view(
                self.n_problems, self.particles_per_problem, self.horizon
            )
            with profiler.record_function("mppi/update_distribution"):
                self._update_distribution(trajectory)
            # print("[particle opt base] trajectory: ", trajectory.actions[0][0][0][0], trajectory.state.state_seq.position[0][0][0])
            if not self.use_cuda_graph and self.store_debug:
                self.debug.append(self._get_action_seq(mode=self.sample_mode).clone())
                self.debug_cost.append(
                    torch.min(torch.sum(trajectory.costs, dim=-1), dim=-1)[0].unsqueeze(-1).clone()
                )

        curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        # print("[particle opt base] curr_action_seq: ", curr_action_seq)
        return curr_action_seq

    def update_nproblems(self, n_problems):
        assert n_problems > 0
        self.total_num_particles = n_problems * self.num_particles
        self.cu_opt_init = False
        super().update_nproblems(n_problems)

    def update_num_particles_per_problem(self, num_particles_per_problem):
        self.null_per_problem = round(int(self.null_act_frac * num_particles_per_problem * 0.5))

        self.neg_per_problem = (
            round(int(self.null_act_frac * num_particles_per_problem)) - self.null_per_problem
        )

        self.sampled_particles_per_problem = (
            num_particles_per_problem - self.null_per_problem - self.neg_per_problem
        )
        self.particles_per_problem = num_particles_per_problem
        if self.null_per_problem > 0:
            self.null_act_seqs = torch.zeros(
                self.null_per_problem,
                self.action_horizon,
                self.d_action,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
