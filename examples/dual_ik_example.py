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
import time

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml, get_robot_path
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from curobo.util.logger import log_info, setup_logger

import pytorch3d.transforms
import pytorch_kinematics as pk

from torch.profiler import ProfilerActivity, profile, record_function

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np

from curobo.cuda_robot_model.cuda_robot_model import TorchJacobian

# for visualizing the robot
from urdfpy import URDF

import h5py

def demo_basic_ik():
    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_configs_path(), "robdekon_demo_scanning.yml"))
    # urdf_file = config_file["robot_cfg"]["kinematics"][
    #     "urdf_path"
    # ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    link_names = config_file["robot_cfg"]["kinematics"]["link_names"]

    robot = URDF.load("../src/curobo/content/assets/robot/robdekon_description/demo_robdekon_scanning.urdf")
    # robot = URDF.load("../src/curobo/content/assets/robot/franka_description/franka_panda.urdf")
    # robot = URDF.load("ur5.urdf")

    world_config = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "robdekon_collision_table.yml")))

    # robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, link_names, tensor_args)
    robot_file = "robdekon_demo_scanning.yml"

    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_config,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=1024,
        self_collision_check=True,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        # use_particle_opt=False,
        use_nullspace_opt=False,
        use_particle_opt=True,
    )
    ik_config_ruckig = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_config,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=2048,
        base_cfg_file="ruckig_base_cfg.yml",
        particle_file="ruckig_particle_ik.yml",
        gradient_file="ruckig_gradient_ik.yml",
        self_collision_check=True,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        # use_particle_opt=False,
        use_nullspace_opt=False,
        use_particle_opt=True,
    )

    print("ik_config loaded!")

    ik_solver = IKSolver(ik_config)
    ik_solver_ruckig = IKSolver(ik_config_ruckig)

    nbv_quaternions = [
        np.array([0.7552389675242316, 0.643201926210182, 0.1261165494788217, -0.0]),
        np.array([0.4212219764862103, 0.16963181543595246, -0.8909529132995488, 0.0]),
        np.array([0.8991508864382314, 0.1369808511424036, -0.4156488058898345, 0.0]),
        np.array([0.96085963750168, -0.27562243494257765, -0.027946920697361665, 0.0]),
        np.array([0.6532736697142779, 0.721038464860201, -0.23094814277253972, 0.0]),
        np.array([0.7102406226041185, -0.47334219871087074, -0.5210618206340516, 0.0])
    ]

    nbv_translations = [
        np.array([-7.619850609288319, 38.861692692846, -5.630871845365393]),
        np.array([30.023115767694645, 5.716211885830074, 25.805763722004045]),
        np.array([29.898479381026966, 9.85331629838045, -24.677785326612586]),
        np.array([2.14824544724441, -21.186757834100444, -33.860099438388794]),
        np.array([12.069787259417122, 37.68283551554983, 5.8586809966432485]),
        np.array([29.606341752189152, -26.8949486333769, -0.35533935976687026])
    ]

    # print(kin_state)
    duration = 0
    duration_list = []
    position_error = 0
    position_error_list = []
    reference_duration = 0
    reference_duration_list = []
    r_duration = 0
    r_duration_list = []
    r_position_error = 0
    r_position_error_list = []
    success = 0
    r_success = 0

    q_init = ik_solver.sample_configs(1)
    q_sample = ik_solver.sample_configs(1)
    q_init[0, 0] = 3.14
    q_init[0, 1] = 0.0
    q_init[0, 2] = 1.57
    q_init[0, 3] = 0.0
    q_init[0, 4] = -1.57
    q_init[0, 5] = 0.0
    q_init[0, 6] = 0.0
    q_init[0, 7] = 0.0
    q_init[0, 8] = 0.0
    q_init[0, 9] = -1.57
    q_init[0, 10] = 0.0
    q_init[0, 11] = 1.57
    q_init[0, 12] = 0.0

    with h5py.File('curobo_ik_solution.h5', 'w') as h5f:
        # for test_index in range(10):
        for test_index in range(len(nbv_quaternions)):
            print("index", test_index)
            # q_sample = ik_solver.sample_configs(100)
            # kin_state = ik_solver.fk(q_sample)
            # goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

            # st_time = time.time()
            # result = ik_solver.solve_batch(goal)

# q_sample = ik_solver.sample_configs(1)
# q_init = ik_solver.sample_configs(1)
            # q_sample[0, 0] = 0
            # q_sample[0, 1] = -0.5
            # q_sample[0, 2] = -1.0
            # q_sample[0, 3] = -1.383
            # q_sample[0, 4] = -2.0
            # q_sample[0, 5] = 0.0
            # q_sample[0, 6] = 1.0
            # q_sample[0, 7] = -0.0
            # q_sample[0, 8] = 0.0
            # q_sample[0, 9] = -1.9
            # q_sample[0, 10] = 0.0
            # q_sample[0, 11] = 0.6
            # q_sample[0, 12] = 0.0

            # q_init[0, 0] = 0
            # q_init[0, 1] = -0.5
            # q_init[0, 2] = -1.0
            # q_init[0, 3] = -1.383
            # q_init[0, 4] = -2.0
            # q_init[0, 5] = 0.0
            # q_init[0, 6] = 2
            # q_init[0, 7] = -0.0
            # q_init[0, 8] = 0.0
            # q_init[0, 9] = -1.9
            # q_init[0, 10] = 0.0
            # q_init[0, 11] = 0.6
            # q_init[0, 12] = 0.0

            # q_init[0, 0] = 3.14
            # q_init[0, 1] = 0.0
            # q_init[0, 2] = 1.57
            # q_init[0, 3] = 0.0
            # q_init[0, 4] = -1.57
            # q_init[0, 5] = 0.0
            # q_init[0, 6] = 0.0
            # q_init[0, 7] = 0.0
            # q_init[0, 8] = 0.0
            # q_init[0, 9] = -1.57
            # q_init[0, 10] = 0.0
            # q_init[0, 11] = 1.57
            # q_init[0, 12] = 0.0

            # for joint in robot.actuated_joints:
            #     print(joint.name)
            # robot.show(cfg={
            #     'shoulder_pan_joint': q_sample[0, 0],
            #     'shoulder_lift_joint': q_sample[0, 1],
            #     'elbow_joint': q_sample[0, 2],
            #     'wrist_1_joint': q_sample[0, 3],
            #     'wrist_2_joint': q_sample[0, 4],
            #     'wrist_3_joint': q_sample[0, 5],
            #     'shoulder_pan_joint_1': q_sample[0, 6],
            #     'shoulder_lift_joint_1': q_sample[0, 7],
            #     'elbow_joint_1': q_sample[0, 8],
            #     'wrist_1_joint_1': q_sample[0, 9],
            #     'wrist_2_joint_1': q_sample[0, 10],
            #     'wrist_3_joint_1': q_sample[0, 11]
            # })
            robot.show(cfg={
                'shoulder_pan_joint': q_init[0, 0],
                'shoulder_lift_joint': q_init[0, 1],
                'elbow_joint': q_init[0, 2],
                'wrist_1_joint': q_init[0, 3],
                'wrist_2_joint': q_init[0, 4],
                'wrist_3_joint': q_init[0, 5],
                'iiwa_joint_1_right': q_init[0, 6],
                'iiwa_joint_2_right': q_init[0, 7],
                'iiwa_joint_3_right': q_init[0, 8],
                'iiwa_joint_4_right': q_init[0, 9],
                'iiwa_joint_5_right': q_init[0, 10],
                'iiwa_joint_6_right': q_init[0, 11],
                'iiwa_joint_7_right': q_init[0, 12],
            })
            #
            # robot.show(cfg={
            #     'shoulder_pan_joint': q_sample[0, 0],
            #     'shoulder_lift_joint': q_sample[0, 1],
            #     'elbow_joint': q_sample[0, 2],
            #     'wrist_1_joint': q_sample[0, 3],
            #     'wrist_2_joint': q_sample[0, 4],
            #     'wrist_3_joint': q_sample[0, 5],
            #     'iiwa_joint_1_right': q_sample[0, 6],
            #     'iiwa_joint_2_right': q_sample[0, 7],
            #     'iiwa_joint_3_right': q_sample[0, 8],
            #     'iiwa_joint_4_right': q_sample[0, 9],
            #     'iiwa_joint_5_right': q_sample[0, 10],
            #     'iiwa_joint_6_right': q_sample[0, 11],
            #     'iiwa_joint_7_right': q_sample[0, 12],
            # })
            #
            # robot.show(cfg={
            #     'shoulder_pan_joint': -3.1415876433479535,
            #     'shoulder_lift_joint': 1.6690310898927359,
            #     'elbow_joint': -1.0182153986470228,
            #     'wrist_1_joint': -2.0978063801322615,
            #     'wrist_2_joint': -2.614628186834798,
            #     'wrist_3_joint': 1.7582968574482856,
            #     'iiwa_joint_1_right': 2.0907069754706584,
            #     'iiwa_joint_2_right': -1.1730962714948905,
            #     'iiwa_joint_3_right': -1.6430561728270858,
            #     'iiwa_joint_4_right': -1.8202711721694083,
            #     'iiwa_joint_5_right': -0.19866255064376595,
            #     'iiwa_joint_6_right': -0.3757050741638216,
            #     'iiwa_joint_7_right': 1.5548407088365186,
            # })
            #
            # robot.show(cfg={
            #     'shoulder_pan_joint': -3.1400747168771383,
            #     'shoulder_lift_joint': 0.12893720149256452,
            #     'elbow_joint': 1.6268482020016746,
            #     'wrist_1_joint': -0.33406905557686567,
            #     'wrist_2_joint': -1.7297792372047258,
            #     'wrist_3_joint': 1.3868080286937388,
            #     'iiwa_joint_1_right': 2.547849371157784,
            #     'iiwa_joint_2_right': -1.9479643445546702,
            #     'iiwa_joint_3_right': -1.458400842542874,
            #     'iiwa_joint_4_right': -0.860362436552262,
            #     'iiwa_joint_5_right': 0.23582370660422702,
            #     'iiwa_joint_6_right': 0.2491087766686544,
            #     'iiwa_joint_7_right': 1.9119831479990488,
            # })



            # kin_state = ik_solver.fk(q_sample)
            # # print("kin_state", kin_state.ee_position)
            # goal_1 = Pose(kin_state.ee_position, kin_state.ee_quaternion)
            # link_poses = {}
            # for i in range(len(link_names)):
            #     if link_names[i] != ee_link:
            #         link_poses[link_names[i]] = Pose(
            #             kin_state.links_position[:, i],
            #             kin_state.links_quaternion[:, i]
            #         )
            #         # link_poses[link_names[i]] = Pose(
            #         #     kin_state.ee_position,
            #         #     kin_state.ee_quaternion
            #         # )
            # print("goal_1", goal_1.position, goal_1.quaternion)
            # print("link_pose", link_poses["zimmer_ee_iiwa"].position, link_poses["zimmer_ee_iiwa"].quaternion)


            # FOR NBV DEMO TESTING
            def pose_to_matrix(position, quaternion):
                """Convert position and quaternion to a homogeneous transformation matrix."""
                # Convert quaternion to rotation matrix using pytorch3d
                rotation_matrix = pytorch3d.transforms.quaternion_to_matrix(quaternion)

                # Create a 4x4 identity matrix
                T = torch.eye(4)

                # Set the translation (position)
                T[:3, 3] = position

                # Set the rotation matrix in T
                T[:3, :3] = rotation_matrix

                return T

            def matrix_to_pose(T):
                """Convert a homogeneous transformation matrix to position and quaternion."""
                # Extract position from the last column
                position = T[:3, 3]

                # Extract the rotation matrix
                rotation_matrix = T[:3, :3]

                # Convert the rotation matrix back to quaternion
                quaternion = pytorch3d.transforms.matrix_to_quaternion(rotation_matrix)

                return position, quaternion

            def calculate_second_tcp_pose(ee_pose_ref, relative_pose_target):
                """Calculate the second TCP pose based on a reference EE pose and relative target pose."""
                # Convert reference EE pose and relative target pose to matrices
                T_ee_ref = pose_to_matrix(ee_pose_ref['position'], ee_pose_ref['quaternion'])
                T_relative = pose_to_matrix(relative_pose_target['position'], relative_pose_target['quaternion'])

                # Compute the second TCP pose by multiplying the matrices
                T_tcp_2 = torch.matmul(T_ee_ref, T_relative)

                # Convert the resulting transformation matrix back to position and quaternion
                tcp_2_position, tcp_2_quaternion = matrix_to_pose(T_tcp_2)

                return {
                    'position': tcp_2_position.to(device=ee_pose_ref['position'].device, dtype=ee_pose_ref['position'].dtype),
                    'quaternion': tcp_2_quaternion.to(device=ee_pose_ref['position'].device, dtype=ee_pose_ref['position'].dtype)
                }

            ee_pose_ref = {
                'position': torch.tensor([0.5, 0.0, 1.4], device=tensor_args.device, dtype=tensor_args.dtype),  # Reference EE position
                'quaternion':torch.tensor([0.707, 0.707, 0.0, 0.0], device=tensor_args.device, dtype=tensor_args.dtype)  # Reference EE quaternion
            }

            # relative_pose_target = {
            #     'position': torch.tensor([-0.07619850609288319, 0.38861692692846, -0.05630871845365393], device=tensor_args.device, dtype=tensor_args.dtype),  # Relative position to the second TCP
            #     'quaternion': torch.tensor([0.7715232547474535, 0.643201926210182, 0.1261165494788217, -0.0], device=tensor_args.device, dtype=tensor_args.dtype)  # Relative orientation to the second TCP
            # }

            relative_pose_target = {
                'position': torch.tensor(nbv_translations[test_index]/100,
                                         device=tensor_args.device, dtype=tensor_args.dtype),
                # Relative position to the second TCP
                'quaternion': torch.tensor(nbv_quaternions[test_index],
                                           device=tensor_args.device, dtype=tensor_args.dtype)
                # Relative orientation to the second TCP
            }

            second_tcp_pose = calculate_second_tcp_pose(ee_pose_ref, relative_pose_target)


            goal_1 = Pose(second_tcp_pose['position'],
                        second_tcp_pose['quaternion'])
            link_poses = {}
            for i in range(len(link_names)):
                if link_names[i] != ee_link:
                    link_poses[link_names[i]] = Pose(
                        # second_tcp_pose['position'],
                        # second_tcp_pose['quaternion']
                        ee_pose_ref['position'],
                        ee_pose_ref['quaternion']
                    )
            print("goal_1", goal_1.position, goal_1.quaternion)
            print("link_pose", link_poses["zimmer_ee_iiwa"].position, link_poses["zimmer_ee_iiwa"].quaternion)

            # link_poses[link_names[i]].position[0, 0] = kin_state.ee_position[0, 0]
            # link_poses[link_names[i]].position[0, 1] = kin_state.ee_position[0, 1]
            # print( link_poses["zimmer_ee_iiwa"].position)
            # link_poses["zimmer_ee_iiwa"].position[0, 1] += 0.4
            # link_poses[link_names[i]].rotation = kin_state.ee_position.rotation
            # print("link poses", kin_state.link_poses)

            # q_init = torch.Tensor([0.0,1.57,-1.57,0,0,3.14,0.0,1.57,-1.57,0,0,3.14]).cuda()
            # q_init = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda()
            # q_init_cpu = generate_random_joint_angles()
            # q_init = torch.Tensor(q_init_cpu).cuda()
            # print("!!!!!!!", q_init.shape)
            q_init = q_init.squeeze(0)
            # q_init = q_sample[0]
            # print(q_sample[0])

            st_time = time.time()
            # q_init = torch.randn(7, device=tensor_args.device)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            #     torch.cuda.profiler.start()
            result = ik_solver.solve(goal_1, link_poses=link_poses, q_init=q_init)
            result_ruckig = ik_solver_ruckig.solve(goal_1, link_poses=link_poses, q_init=q_init)
                # torch.cuda.profiler.stop()
            end_time = time.time()

            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # print("goal", kin_state.link_pose)
            # print("q_sample", q_sample)
            # print("result: ", result.js_solution.position[0])
            torch.cuda.synchronize()
            kin_state_solution = ik_solver.fk(result.js_solution.position[0])
            kin_state_solution_ruckig = ik_solver.fk(result_ruckig.js_solution.position[0])
            # print("caparison", kin_state.link_pose, "\n result: ", kin_state_solution.link_pose)
            print("caparison", goal_1, "\n result: ", kin_state_solution.link_pose)
            relative_pos_ee, relative_quat_ee = calculate_relative_pose(kin_state_solution.link_pose, "zimmer_ee_ur",
                                                                         "zimmer_ee_iiwa")

            relative_pos_ee_ruckig, relative_quat_ee_ruckig = calculate_relative_pose(kin_state_solution_ruckig.link_poses, "zimmer_ee_ur", "zimmer_ee_iiwa")
            print("relative_pose_gt, relative_quat_gt", relative_pose_target, relative_pos_ee, relative_quat_ee)
            print("relative_pose_gt, relative_quat_gt", relative_pose_target, relative_pos_ee_ruckig, relative_quat_ee_ruckig)

            # robot.show(cfg={
            #     'shoulder_pan_joint': result.js_solution.position[0][0, 0],
            #     'shoulder_lift_joint': result.js_solution.position[0][0, 1],
            #     'elbow_joint': result.js_solution.position[0][0, 2],
            #     'wrist_1_joint': result.js_solution.position[0][0, 3],
            #     'wrist_2_joint': result.js_solution.position[0][0, 4],
            #     'wrist_3_joint': result.js_solution.position[0][0, 5],
            #     'shoulder_pan_joint_1': result.js_solution.position[0][0, 6],
            #     'shoulder_lift_joint_1': result.js_solution.position[0][0, 7],
            #     'elbow_joint_1': result.js_solution.position[0][0, 8],
            #     'wrist_1_joint_1': result.js_solution.position[0][0, 9],
            #     'wrist_2_joint_1': result.js_solution.position[0][0, 10],
            #     'wrist_3_joint_1': result.js_solution.position[0][0, 11]
            # })

            robot.show(cfg={
                'shoulder_pan_joint': result.js_solution.position[0][0, 0],
                'shoulder_lift_joint': result.js_solution.position[0][0, 1],
                'elbow_joint': result.js_solution.position[0][0, 2],
                'wrist_1_joint': result.js_solution.position[0][0, 3],
                'wrist_2_joint': result.js_solution.position[0][0, 4],
                'wrist_3_joint': result.js_solution.position[0][0, 5],
                'iiwa_joint_1_right': result.js_solution.position[0][0, 6],
                'iiwa_joint_2_right': result.js_solution.position[0][0, 7],
                'iiwa_joint_3_right': result.js_solution.position[0][0, 8],
                'iiwa_joint_4_right': result.js_solution.position[0][0, 9],
                'iiwa_joint_5_right': result.js_solution.position[0][0, 10],
                'iiwa_joint_6_right': result.js_solution.position[0][0, 11],
                'iiwa_joint_7_right': result.js_solution.position[0][0, 12]
            })

            robot.show(cfg={
                'shoulder_pan_joint': result_ruckig.js_solution.position[0][0, 0],
                'shoulder_lift_joint': result_ruckig.js_solution.position[0][0, 1],
                'elbow_joint': result_ruckig.js_solution.position[0][0, 2],
                'wrist_1_joint': result_ruckig.js_solution.position[0][0, 3],
                'wrist_2_joint': result_ruckig.js_solution.position[0][0, 4],
                'wrist_3_joint': result_ruckig.js_solution.position[0][0, 5],
                'iiwa_joint_1_right': result_ruckig.js_solution.position[0][0, 6],
                'iiwa_joint_2_right': result_ruckig.js_solution.position[0][0, 7],
                'iiwa_joint_3_right': result_ruckig.js_solution.position[0][0, 8],
                'iiwa_joint_4_right': result_ruckig.js_solution.position[0][0, 9],
                'iiwa_joint_5_right': result_ruckig.js_solution.position[0][0, 10],
                'iiwa_joint_6_right': result_ruckig.js_solution.position[0][0, 11],
                'iiwa_joint_7_right': result_ruckig.js_solution.position[0][0, 12]
            })

            # with torch.autograd.profiler.profile(enabled=True) as prof:
            #     print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# relative_pose_gt, relative_quat_gt = calculate_relative_pose(kin_state.link_pose, "zimmer_ee_iiwa", "zimmer_ee_ur")
            # print("relative_pose_gt, relative_quat_gt", relative_pose_gt, relative_quat_gt)
# relative_pose, relative_quat = calculate_relative_pose(kin_state_solution.link_pose, "zimmer_ee_iiwa", "zimmer_ee_ur")

            # urdf_path = get_robot_path() + "/ur_description/ur10e.urdf"
            # base_link = "base_link"
            # end_effector_link = "tool0"

            # joint_angles = np.array([0.0, 1.57, -1.57, 0, 0, 3.14, 0.0])
            joint_angles = np.array(q_init.cpu())
            joint_goal_angles = result.js_solution.position[0].cpu().numpy()

            print(joint_angles, joint_goal_angles)

            ground_truth_duration = calculate_trajectory_duration(joint_angles, joint_goal_angles)

            joint_angles_ruckig = np.array(q_init.cpu())
            joint_goal_angles_ruckig = result_ruckig.js_solution.position[0].cpu().numpy()
            ruckig_duration = calculate_trajectory_duration(joint_angles_ruckig, joint_goal_angles_ruckig)

            # print(joint_goal_angles, q_sample[0].cpu().numpy())
            duration_without_opt = calculate_trajectory_duration(joint_angles, q_sample.cpu().numpy())
            # print("reference duration:", duration_without_opt)
            reference_duration += duration_without_opt
            reference_duration_list.append(duration_without_opt)

            grp = h5f.create_group(f"goal_{test_index}")

            data = {
                "starting_joint_angle": q_init.cpu().numpy(),
                "reference_joint_angle": q_sample.cpu().numpy(),
                "relative_dist_joint_angle": joint_goal_angles,
                "relative_curobo_joint_angle": joint_goal_angles_ruckig,
                # "goal_position": kin_state.ee_position.cpu().numpy(),
                # "goal_quaternion": kin_state.ee_quaternion.cpu().numpy(),
                # "goal_tool_position": kin_state.link_pose["zimmer_ee_iiwa"].position.cpu().numpy(),
                # "goal_tool_quaternion": kin_state.link_pose["zimmer_ee_iiwa"].quaternion.cpu().numpy(),
                "ground_truth_duration": np.array([duration_without_opt]),
                "curobo_duration": np.array([ruckig_duration]),
                "dist_duration": np.array([ground_truth_duration]),
            }

            grp.create_dataset("starting_joint_angle", data=data["starting_joint_angle"])
            grp.create_dataset("reference_joint_angle", data=data["reference_joint_angle"])
            grp.create_dataset("relative_dist_joint_angle", data=data["relative_dist_joint_angle"])
            grp.create_dataset("relative_curobo_joint_angle", data=data["relative_curobo_joint_angle"])
            # grp.create_dataset("goal_position", data=data["goal_position"])
            # grp.create_dataset("goal_quaternion", data=data["goal_quaternion"])
            # grp.create_dataset("goal_tool_position", data=data["goal_tool_position"])
            # grp.create_dataset("goal_tool_quaternion", data=data["goal_tool_quaternion"])
            grp.create_dataset("ground_truth_duration", data=data["ground_truth_duration"])
            grp.create_dataset("curobo_duration", data=data["curobo_duration"])
            grp.create_dataset("dist_duration", data=data["dist_duration"])


            print("time", end_time-st_time)

            if result.success:
                success += 1
                # print(ground_truth_duration)
                duration += ground_truth_duration
                duration_list.append(ground_truth_duration)

                # position_error += torch.norm(relative_pose_gt - relative_pose, p=2)
                # position_error_list.append(torch.norm(relative_pose_gt - relative_pose, p=2))
                position_error += result.position_error[0]
                position_error_list.append(result.position_error[0])

            if result_ruckig.success:
                r_success += 1
                r_duration += ruckig_duration
                r_duration_list.append(ruckig_duration)
                r_kin_state_solution = ik_solver_ruckig.fk(result_ruckig.js_solution.position[0])

                r_relative_pose, r_relative_quat = calculate_relative_pose(r_kin_state_solution.link_pose, "zimmer_ee_iiwa",
                                                                           "zimmer_ee_ur")
                # r_position_error += torch.norm(relative_pose_gt - r_relative_pose, p=2)
                # r_position_error_list.append(torch.norm(relative_pose_gt - r_relative_pose, p=2))
                r_position_error += result_ruckig.position_error[0]
                # print("result ruckig",  torch.norm(relative_pose_gt - r_relative_pose, p=2), result_ruckig.position_error)
                r_position_error_list.append(result_ruckig.position_error[0])

            q_init = q_init.unsqueeze(0)
            q_init[0] = result_ruckig.js_solution.position[0]

        # Nullspace solver
        # robot_model = TorchJacobian()  # Assume this is the robot's kinematic model
        # nullspace_solver = NullspaceIKSolver(robot_model)
        # test_relative_pos = torch.tensor([[0, 0, 0.5]], device=tensor_args.device, dtype=tensor_args.dtype)
        # # wxyz
        # test_relative_qua = torch.tensor([[0, 1, 0, 0]], device=tensor_args.device, dtype=tensor_args.dtype)
        # # q_opt, success = nullspace_solver.solve(q_init, relative_pose_gt, relative_quat_gt)
        # q_opt, success = nullspace_solver.solve(q_init, test_relative_pos, test_relative_qua)
        #
        # print("nullspace solution", q_opt, success)

    print("average reference duration", reference_duration / 1)
    print("reference duration list", reference_duration_list)
    print("average duration", duration) #  /success)
    print("duration list", duration_list)
    print("average ruckig duration", r_duration) # / r_success)
    print("ruckig duration list", r_duration_list)

    print("average position error", position_error) #  /success)
    print("position error list", position_error_list)
    print("average ruckig position error", r_position_error) # / r_success)
    print("position ruckig error list", r_position_error_list)

    print("ruckig success", r_success)
    print(" success", success)




        # current_pos = kin_state_solution.link_pose["tool1"].position
        # ee_pos_batch = kin_state_solution.link_pose["tool0"].position
        #
        # ee_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(kin_state_solution.link_pose["tool0"].quaternion)
        # current_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(kin_state_solution.link_pose["tool1"].quaternion)
        #
        # current_matrix_batch_inv = current_matrix_batch.transpose(-2, -1)
        #
        # # Compute the inverse of the translation vector
        # current_pos_inv = -torch.matmul(current_matrix_batch_inv, current_pos.unsqueeze(-1)).squeeze(-1)
        #
        # relative_rot_batch = torch.matmul(current_matrix_batch_inv, ee_matrix_batch)
        # relative_pos_batch = torch.matmul(current_matrix_batch_inv, ee_pos_batch.unsqueeze(-1)).squeeze(-1)
        #
        # relative_ee_pos_batch = current_pos_inv + relative_pos_batch
        #
        # relative_ee_quat_batch = pytorch3d.transforms.matrix_to_quaternion(relative_rot_batch)


        # print(
        #     "Success, Solve Time(s), hz ",
        #     torch.count_nonzero(result.success).item() / len(q_sample),
        #     result.solve_time,
        #     q_sample.shape[0] / (time.time() - st_time),
        #     torch.mean(result.position_error),
        #     torch.mean(result.rotation_error),
        # )

def calculate_relative_pose(link_poses, ee_link, base_link):
    current_pos = link_poses[base_link].position
    ee_pos_batch = link_poses[ee_link].position

    ee_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(link_poses[ee_link].quaternion)
    current_matrix_batch = pytorch3d.transforms.quaternion_to_matrix(link_poses[base_link].quaternion)

    current_matrix_batch_inv = current_matrix_batch.transpose(-2, -1)

    # Compute the inverse of the translation vector
    current_pos_inv = -torch.matmul(current_matrix_batch_inv, current_pos.unsqueeze(-1)).squeeze(-1)

    relative_rot_batch = torch.matmul(current_matrix_batch_inv, ee_matrix_batch)
    relative_pos_batch = torch.matmul(current_matrix_batch_inv, ee_pos_batch.unsqueeze(-1)).squeeze(-1)

    relative_ee_pos_batch = current_pos_inv + relative_pos_batch

    relative_ee_quat_batch = pytorch3d.transforms.matrix_to_quaternion(relative_rot_batch)

    # print("relative pos", relative_ee_pos_batch, "relative quat", relative_ee_quat_batch)

    return relative_ee_pos_batch, relative_ee_quat_batch

def calculate_trajectory_duration(joint_angles, joint_goal_angles):
    # way_pts = np.concatenate((joint_goal_angles[0, :6], np.array([0])))
    way_pts = np.array([joint_angles, joint_goal_angles[0]])
    # print(way_pts)

    traj = toppra_solver(way_pts)
    ground_truth_duration = traj.duration

    return ground_truth_duration

def toppra_solver(way_pts):
    ss = np.linspace(0, 1, 2)
    # vlims = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    # alims = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    vlims = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, ])
    alims = np.array([5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    path = ta.SplineInterpolator(ss, way_pts)
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)
    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()
    return jnt_traj

def generate_random_joint_angles():
    joint_limits = {
        'joint1': (-3.14159, 3.14159), # ur5
        'joint2': (-3.14159, 3.14159),
        'joint3': (-3.14159, 3.14159),
        'joint4': (-3.14159, 3.14159),
        'joint5': (-3.14159, 3.14159),
        'joint6': (-3.14159, 3.14159),
        'joint7': (-2.96705972839, 2.96705972839), # iiwa14
        'joint8': (-2.09439510239, 2.09439510239),
        'joint9': (-2.96705972839, 2.96705972839),
        'joint10': (-2.09439510239, 2.09439510239),
        'joint11': (-2.96705972839, 2.96705972839),
        'joint12': (-2.09439510239, 2.09439510239),
        'joint13': (-3.05432619099, 3.05432619099)
    }
    random_joint_angles = [
        np.random.uniform(joint_limits['joint1'][0], joint_limits['joint1'][1]),
        np.random.uniform(joint_limits['joint2'][0], joint_limits['joint2'][1]),
        np.random.uniform(joint_limits['joint3'][0], joint_limits['joint3'][1]),
        np.random.uniform(joint_limits['joint4'][0], joint_limits['joint4'][1]),
        np.random.uniform(joint_limits['joint5'][0], joint_limits['joint5'][1]),
        np.random.uniform(joint_limits['joint6'][0], joint_limits['joint6'][1]),
        np.random.uniform(joint_limits['joint7'][0], joint_limits['joint7'][1]),
        np.random.uniform(joint_limits['joint8'][0], joint_limits['joint8'][1]),
        np.random.uniform(joint_limits['joint9'][0], joint_limits['joint9'][1]),
        np.random.uniform(joint_limits['joint10'][0], joint_limits['joint10'][1]),
        np.random.uniform(joint_limits['joint11'][0], joint_limits['joint11'][1]),
        np.random.uniform(joint_limits['joint12'][0], joint_limits['joint12'][1]),
        np.random.uniform(joint_limits['joint13'][0], joint_limits['joint13'][1])
    ]
    return random_joint_angles


class NullspaceIKSolver:
    def __init__(self, robot_model, damping=1e-3, secondary_objective=None):
        """
        Initialize the nullspace-based IK solver.

        Parameters:
        - robot_model: An instance of the robot's forward kinematics and Jacobian calculator.
        - damping: Damping factor for the pseudoinverse computation.
        - secondary_objective: A function that computes a secondary objective (e.g., joint limit avoidance).
        """
        self.robot_model = robot_model
        self.damping = damping
        self.secondary_objective = secondary_objective

    def compute_jacobian_pseudoinverse(self, J):
        """
        Compute the damped pseudoinverse of the Jacobian.

        Parameters:
        - J: The Jacobian matrix.

        Returns:
        - J_pseudo_inv: The pseudoinverse of the Jacobian matrix.
        """
        JT = torch.transpose(J, 1, 2)
        print("J shape", J.shape, "JT shape", JT.shape)
        JJT = J @ torch.transpose(J, 1, 2)
        print("JJT shape", JJT.shape)
        damping_matrix = self.damping * torch.eye(J.shape[0], device=J.device, dtype=J.dtype)
        J_pseudo_inv = JT @ torch.inverse(JJT + damping_matrix)
        return J_pseudo_inv

    def solve(self, q_init, target_position, target_quaternion, max_iters=1000, tolerance=1e-4):
        """
        Solve the inverse kinematics problem with nullspace optimization.

        Parameters:
        - q_init: Initial guess for the joint angles.
        - target_pose: Desired end-effector pose (position and orientation).
        - max_iters: Maximum number of iterations.
        - tolerance: Tolerance for convergence.

        Returns:
        - q_opt: The optimized joint angles.
        - success: Boolean indicating if the solver converged.
        """
        q = q_init.unsqueeze(0).clone().detach().requires_grad_(True)
        print("q shape", q.shape)
        robot = URDF.load("../src/curobo/content/assets/robot/ur_description/dual_ur10e.urdf")

        for _ in range(max_iters):
            print("q at the beginning", q)
            # Forward kinematics
            current_pose_1 = self.robot_model.chain.forward_kinematics(q[:, :6])
            current_pose_2 = self.robot_model.chain_1.forward_kinematics(q[:, 6:])
            # print("current_pose_1", current_pose_1, "current_pose_2", current_pose_2)

            current_relative_pose = torch.matmul(current_pose_2._get_matrix_inverse(), current_pose_1.get_matrix())
            print("current_relative_pose", current_relative_pose)

            target_matrix = pk.Transform3d(pos=target_position, rot=target_quaternion, device=q.device)
            # print("target_matrix", target_matrix)

            current_relative_pose = current_relative_pose.unsqueeze(1)
            print("current relative pose", current_relative_pose)

            dx, pos_diff, rot_diff = self.delta_pose(current_relative_pose, target_position, target_quaternion)
            print("dx", dx, "pos_diff", pos_diff, "rot_diff", rot_diff)

            # Check for convergence
            if torch.norm(dx) < tolerance:
                return q.detach(), True

            # Jacobian matrix
            J_1 = self.robot_model.chain.jacobian(q[:, :6])
            J_2 = self.robot_model.chain.jacobian(q[:, 6:])

            diag_R_ra = torch.block_diag(current_pose_2.get_matrix().squeeze()[:3, :3], current_pose_2.get_matrix().squeeze()[:3, :3])
            diag_R_rt = torch.block_diag(current_relative_pose.squeeze()[:3, :3],
                                         current_relative_pose.squeeze()[:3, :3])
            diag_R_tb = torch.block_diag(current_pose_1.get_matrix().squeeze()[:3, :3],
                                         current_pose_1.get_matrix().squeeze()[:3, :3])

            J_rel_cal = torch.cat((-diag_R_ra @ J_1, diag_R_rt @ diag_R_tb @ J_2), dim=2)

            # Compute Jacobian pseudoinverse
            J_pseudo_inv = self.compute_jacobian_pseudoinverse(J_rel_cal)
            # print("J_pseudo_inv", J_pseudo_inv)

            # Compute primary task update
            delta_q_primary = J_pseudo_inv @ dx
            # print("delta_q_primary", delta_q_primary.shape, "J_pseudo_inv", J_pseudo_inv.shape, "dx", dx.shape)

            # Compute secondary objective (e.g., joint limit avoidance)
            delta_q_secondary = torch.zeros_like(q)
            if self.secondary_objective:
                delta_q_secondary = self.secondary_objective(q)

            # Nullspace projection
            nullspace_projection = torch.eye(q.shape[1], device=q.device) - J_pseudo_inv @ J_rel_cal
            # print("nullspace_projection", nullspace_projection.shape)

            calculated_q_secondary = nullspace_projection @ delta_q_secondary.unsqueeze(-1)
            # print("calculated_q_secondary", calculated_q_secondary.shape)
            delta_q = delta_q_primary + calculated_q_secondary
            print("delta_q: ", delta_q)

            # if inspect.isclass(self.optimizer_method) and issubclass(self.optimizer_method, torch.optim.Optimizer):
            # optimizer = torch.optim.LBFGS([q], lr=0.01)
            # print("before calculated q", q.grad)
            # q.grad = -delta_q.squeeze(-1)
            # print("calculated q", q.grad)
            #
            # optimizer.step()
            #
            # optimizer.zero_grad()
            # print("calculated q", q)

            # Update joint angles
            q = q + delta_q.squeeze(-1)*0.05
            print("calculated q", q)

        robot.show(cfg={
            'shoulder_pan_joint': q[0, 0],
            'shoulder_lift_joint': q[0, 1],
            'elbow_joint': q[0, 2],
            'wrist_1_joint': q[0, 3],
            'wrist_2_joint': q[0, 4],
            'wrist_3_joint': q[0, 5],
            'shoulder_pan_joint_1': q[0, 6],
            'shoulder_lift_joint_1': q[0, 7],
            'elbow_joint_1': q[0, 8],
            'wrist_1_joint_1': q[0, 9],
            'wrist_2_joint_1': q[0, 10],
            'wrist_3_joint_1': q[0, 11]
        })

        # If the loop completes without convergence
        return q.detach(), False

    def compute_pose_error(self, target_pose, current_pose):
        """
        Compute the error between the target and current end-effector poses.

        Parameters:
        - target_pose: Desired end-effector pose (position and orientation).
        - current_pose: Current end-effector pose.

        Returns:
        - error: The error vector (position and orientation).
        """
        position_error = target_pose[:3] - current_pose[:3]
        orientation_error = target_pose[3:] - current_pose[3:]  # Simplified orientation error
        return torch.cat([position_error, orientation_error])

    def delta_pose(self, m: torch.tensor, target_pos, target_wxyz):
        """
        Determine the error in position and rotation between the given poses and the target poses

        :param m: (N x M x 4 x 4) tensor of homogenous transforms
        :param target_pos:
        :param target_wxyz: target orientation represented in unit quaternion
        :return: (N*M, 6, 1) tensor of delta pose (dx, dy, dz, droll, dpitch, dyaw)
        """
        pos_diff = target_pos.unsqueeze(1) - m[:, :, :3, 3]
        pos_diff = pos_diff.view(-1, 3, 1)
        cur_wxyz = pk.rotation_conversions.matrix_to_quaternion(m[:, :, :3, :3])

        # quaternion that rotates from the current orientation to the desired orientation
        # inverse for unit quaternion is the conjugate
        diff_wxyz = pk.rotation_conversions.quaternion_multiply(target_wxyz.unsqueeze(1),
                                                             pk.rotation_conversions.quaternion_invert(cur_wxyz))
        # angular velocity vector needed to correct the orientation
        # if time is considered, should divide by \delta t, but doing it iteratively we can choose delta t to be 1
        diff_axis_angle = pk.rotation_conversions.quaternion_to_axis_angle(diff_wxyz)

        rot_diff = diff_axis_angle.view(-1, 3, 1)

        dx = torch.cat((pos_diff, rot_diff), dim=1)
        return dx, pos_diff, rot_diff




if __name__ == "__main__":
    setup_logger(level="info")

    demo_basic_ik()
    # demo_full_config_collision_free_ik()
    # demo_full_config_batch_env_collision_free_ik()
