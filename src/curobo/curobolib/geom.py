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
# Third Party
import torch

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.torch_utils import get_torch_jit_decorator

try:
    # CuRobo
    from curobo.curobolib import geom_cu

except ImportError:
    log_warn("geom_cu binary not found, jit compiling...")
    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.util_file import add_cpp_path

    geom_cu = load(
        name="geom_cu",
        sources=add_cpp_path(
            [
                "geom_cuda.cpp",
                "sphere_obb_kernel.cu",
                "pose_distance_kernel.cu",
                "self_collision_kernel.cu",
            ]
        ),
    )


def get_self_collision_distance(
    out_distance,
    out_vec,
    sparse_index,
    robot_spheres,
    collision_offset,
    weight,
    coll_matrix,
    thread_locations,
    thread_size,
    b_size,
    nspheres,
    compute_grad,
    checks_per_thread=32,
    experimental_kernel=True,
):
    r = geom_cu.self_collision_distance(
        out_distance,
        out_vec,
        sparse_index,
        robot_spheres,
        collision_offset,
        weight,
        coll_matrix,
        thread_locations,
        thread_size,
        b_size,
        nspheres,
        compute_grad,
        checks_per_thread,
        experimental_kernel,
    )

    out_distance = r[0]
    out_vec = r[1]
    return out_distance, out_vec


class SelfCollisionDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        out_distance,
        out_vec,
        sparse_idx,
        robot_spheres,
        sphere_offset,
        weight,
        coll_matrix,
        thread_locations,
        max_thread,
        checks_per_thread: int,
        experimental_kernel: bool,
        return_loss: bool = False,
    ):
        # get batch size
        b, h, n_spheres, _ = robot_spheres.shape
        out_distance, out_vec = get_self_collision_distance(
            out_distance,
            out_vec,
            sparse_idx,
            robot_spheres,  # .view(-1, 4),
            sphere_offset,
            weight,
            coll_matrix,
            thread_locations,
            max_thread,
            b * h,
            n_spheres,
            robot_spheres.requires_grad,
            checks_per_thread,
            experimental_kernel,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):
        sphere_grad = None
        if ctx.needs_input_grad[3]:
            (g_vec,) = ctx.saved_tensors
            if ctx.return_loss:
                g_vec = g_vec * grad_out_distance.unsqueeze(1)
            sphere_grad = g_vec
        return None, None, None, sphere_grad, None, None, None, None, None, None, None, None


class SelfCollisionDistanceLoss(SelfCollisionDistance):
    @staticmethod
    def backward(ctx, grad_out_distance):
        sphere_grad = None
        if ctx.needs_input_grad[3]:
            (g_vec,) = ctx.saved_tensors
            sphere_grad = g_vec * grad_out_distance.unsqueeze(1)
        return None, None, None, sphere_grad, None, None, None, None, None, None, None


def get_pose_distance(
    out_distance,
    out_position_distance,
    out_rotation_distance,
    out_p_vec,
    out_q_vec,
    out_idx,
    current_position,
    goal_position,
    current_quat,
    goal_quat,
    vec_weight,
    weight,
    vec_convergence,
    run_weight,
    run_vec_weight,
    offset_waypoint,
    offset_tstep_fraction,
    batch_pose_idx,
    batch_size,
    horizon,
    mode=1,
    num_goals=1,
    write_grad=False,
    write_distance=False,
    use_metric=False,
    project_distance=True,
):
    if batch_pose_idx.shape[0] != batch_size:
        raise ValueError("Index buffer size is different from batch size")

    # print("__________current_position______________", current_position)
    # print("__________goal_position______________", goal_position)
    # print("__________current_quat______________", current_quat)
    # print("__________goal_quat______________", goal_quat)
    r = geom_cu.pose_distance(
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_q_vec,
        out_idx,
        current_position,
        goal_position.view(-1),
        current_quat,
        goal_quat.view(-1),
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        batch_size,
        horizon,
        mode,
        num_goals,
        write_grad,
        write_distance,
        use_metric,
        project_distance,
    )

    # test_torch_dist = TestTorchDistance()
    # r = test_torch_dist.pose_distance(
    #     out_distance,
    #     out_position_distance,
    #     out_rotation_distance,
    #     out_p_vec,
    #     out_q_vec,
    #     out_idx,
    #     current_position,
    #     goal_position.view(-1),
    #     current_quat,
    #     goal_quat.view(-1),
    #     vec_weight,
    #     weight,
    #     vec_convergence,
    #     run_weight,
    #     run_vec_weight,
    #     offset_waypoint,
    #     offset_tstep_fraction,
    #     batch_pose_idx,
    #     batch_size,
    #     horizon,
    #     mode,
    #     num_goals,
    #     write_grad,
    #     write_distance,
    #     use_metric,
    #     project_distance,
    # )

    # print("__________r______________", r)

    out_distance = r[0]
    out_position_distance = r[1]
    out_rotation_distance = r[2]

    out_p_vec = r[3]
    out_q_vec = r[4]

    out_idx = r[5]
    return out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_q_vec, out_idx


def get_pose_distance_backward(
    out_grad_p,
    out_grad_q,
    grad_distance,
    grad_p_distance,
    grad_q_distance,
    pose_weight,
    grad_p_vec,
    grad_q_vec,
    batch_size,
    use_distance=False,
):
    r = geom_cu.pose_distance_backward(
        out_grad_p,
        out_grad_q,
        grad_distance,
        grad_p_distance,
        grad_q_distance,
        pose_weight,
        grad_p_vec,
        grad_q_vec,
        batch_size,
        use_distance,
    )
    return r[0], r[1]


@get_torch_jit_decorator()
def backward_PoseError_jit(grad_g_dist, grad_out_distance, weight, g_vec):
    grad_vec = grad_g_dist + (grad_out_distance * weight)
    grad = 1.0 * (grad_vec).unsqueeze(-1) * g_vec
    return grad


# full method:
@get_torch_jit_decorator()
def backward_full_PoseError_jit(
    grad_out_distance, grad_g_dist, grad_r_err, p_w, q_w, g_vec_p, g_vec_q
):
    p_grad = (grad_g_dist + (grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    q_grad = (grad_r_err + (grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q
    # p_grad = ((grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    # q_grad = ((grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q

    return p_grad, q_grad


class PoseErrorDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_position,
        goal_position,
        current_quat,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode,  # =PoseErrorType.BATCH_GOAL.value,
        num_goals,
        use_metric,  # =False,
        project_distance,  # =True,
    ):
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            offset_waypoint,
            offset_tstep_fraction,
            batch_pose_idx,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            True,
            use_metric,
            project_distance,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec, weight, out_p_grad, out_q_grad)
        return out_distance, out_position_distance, out_rotation_distance, out_idx  # .view(-1,1)

    @staticmethod
    def backward(ctx, grad_out_distance, grad_g_dist, grad_r_err, grad_out_idx):
        (g_vec_p, g_vec_q, weight, out_grad_p, out_grad_q) = ctx.saved_tensors
        pos_grad = None
        quat_grad = None
        batch_size = g_vec_p.shape[0] * g_vec_p.shape[1]
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            pos_grad, quat_grad = get_pose_distance_backward(
                out_grad_p,
                out_grad_q,
                grad_out_distance.contiguous(),
                grad_g_dist.contiguous(),
                grad_r_err.contiguous(),
                weight,
                g_vec_p,
                g_vec_q,
                batch_size,
                use_distance=True,
            )

        elif ctx.needs_input_grad[0]:
            pos_grad = backward_PoseError_jit(grad_g_dist, grad_out_distance, weight[1], g_vec_p)

        elif ctx.needs_input_grad[2]:
            quat_grad = backward_PoseError_jit(grad_r_err, grad_out_distance, weight[0], g_vec_q)

        return (
            pos_grad,
            None,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PoseError(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        goal_position: torch.Tensor,
        current_quat: torch.Tensor,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode,
        num_goals,
        use_metric,
        project_distance,
        return_loss,
    ):
        """Compute error in pose

        _extended_summary_

        Args:
            ctx: _description_
            current_position: _description_
            goal_position: _description_
            current_quat: _description_
            goal_quat: _description_
            vec_weight: _description_
            weight: _description_
            vec_convergence: _description_
            run_weight: _description_
            run_vec_weight: _description_
            offset_waypoint: _description_
            offset_tstep_fraction: _description_
            batch_pose_idx: _description_
            out_distance: _description_
            out_position_distance: _description_
            out_rotation_distance: _description_
            out_p_vec: _description_
            out_r_vec: _description_
            out_idx: _description_
            out_p_grad: _description_
            out_q_grad: _description_
            batch_size: _description_
            horizon: _description_
            mode: _description_
            num_goals: _description_
            use_metric: _description_
            project_distance: _description_
            return_loss: _description_

        Returns:
            _description_
        """
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)
        ctx.return_loss = return_loss

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            offset_waypoint,
            offset_tstep_fraction,
            batch_pose_idx,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            False,
            use_metric,
            project_distance,
        )
        # print("gradient calculated?", out_p_vec, out_r_vec)
        ctx.save_for_backward(out_p_vec, out_r_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance): # , grad_g_dist, grad_r_err, grad_out_idx):
        # print("in the pose gradient backward function: ", ctx.return_loss, grad_out_distance)
        pos_grad = None
        quat_grad = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors
            pos_grad = g_vec_p
            quat_grad = g_vec_q
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)

        elif ctx.needs_input_grad[0]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            pos_grad = g_vec_p
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
        elif ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            quat_grad = g_vec_q
            if ctx.return_loss:
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)
        # print("_____________________________pose gard and quat grad", pos_grad, quat_grad)
        return (
            pos_grad,
            None,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SdfSphereOBB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        box_accel,
        box_dims,
        box_pose,
        box_enable,
        n_env_obb,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):
        r = geom_cu.closest_point(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            box_accel,
            box_dims,
            box_pose,
            box_enable,
            n_env_obb,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
            compute_esdf,
        )
        # r[1][r[1]!=r[1]] = 0.0
        ctx.compute_esdf = compute_esdf
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            # if ctx.compute_esdf:
            #    raise NotImplementedError("Gradients not implemented for compute_esdf=True")
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SdfSweptSphereOBB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        speed_dt,
        box_accel,
        box_dims,
        box_pose,
        box_enable,
        n_env_obb,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        sweep_steps,
        enable_speed_metric,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        r = geom_cu.swept_closest_point(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            speed_dt,
            box_accel,
            box_dims,
            box_pose,
            box_enable,
            n_env_obb,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            sweep_steps,
            enable_speed_metric,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(
            r[1],
        )
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SdfSphereVoxel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        grid_features,
        grid_params,
        grid_pose,
        grid_enable,
        n_env_grid,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):

        r = geom_cu.closest_point_voxel(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            grid_features,
            grid_params,
            grid_pose,
            grid_enable,
            n_env_grid,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
            compute_esdf,
        )
        ctx.compute_esdf = compute_esdf
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            # if ctx.compute_esdf:
            #    raise NotImplementedError("Gradients not implemented for compute_esdf=True")
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SdfSweptSphereVoxel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        speed_dt,
        grid_features,
        grid_params,
        grid_pose,
        grid_enable,
        n_env_grid,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        sweep_steps,
        enable_speed_metric,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        r = geom_cu.swept_closest_point_voxel(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            speed_dt,
            grid_features,
            grid_params,
            grid_pose,
            grid_enable,
            n_env_grid,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            sweep_steps,
            enable_speed_metric,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
        )

        ctx.return_loss = return_loss
        ctx.save_for_backward(
            r[1],
        )
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class TestTorchDistance():
    def compute_pose_distance_vector(self,
                                     project_distance,
                                     goal_position,
                                     goal_quat,
                                     current_position,
                                     current_quat,
                                     vec_weight,
                                     offset_position,
                                     offset_rotation,
                                     reach_offset):

        batch_size, horizon = goal_position.size(0), goal_position.size(1)

        error_position = torch.zeros_like(goal_position)
        error_quat = torch.zeros(batch_size, horizon, 3, dtype=torch.float32, device=goal_position.device)

        if project_distance:
            projected_quat = torch.zeros(4, dtype=torch.float32, device=goal_position.device)
            error_position = self.inv_transform_point(goal_position, goal_quat, current_position)
            projected_quat = self.inv_transform_quat(goal_quat, current_quat)

            r_w = projected_quat[:, :, 3]
            r_w = torch.where(r_w < 0.0, -1.0, 1.0).unsqueeze(-1)


            error_quat[:, :, 0] = r_w[:, :, 0] * projected_quat[:, :, 0]
            error_quat[:, :, 1] = r_w[:, :, 0] * projected_quat[:, :, 1]
            error_quat[:, :, 2] = r_w[:, :, 0] * projected_quat[:, :, 2]
        else:
            error_position = current_position - goal_position

            r_w = (goal_quat[:, :, 3] * current_quat[:, :, 3] +
                   goal_quat[:, :, 0] * current_quat[:, :, 0] +
                   goal_quat[:, :, 1] * current_quat[:, :, 1] +
                   goal_quat[:, :, 2] * current_quat[:, :, 2])

            r_w = torch.where(r_w < 0.0, -1.0, 1.0).unsqueeze(-1)

            error_quat[:, :, 0] = r_w[:, :, 0] * (
                        -goal_quat[:, :, 3] * current_quat[:, :, 0] + current_quat[:, :, 3] * goal_quat[:, :, 0] -
                        goal_quat[:, :, 1] * current_quat[:, :, 2] + current_quat[:, :, 1] * goal_quat[:, :, 2])
            error_quat[:, :, 1] = r_w[:, :, 0] * (
                        -goal_quat[:, :, 3] * current_quat[:, :, 1] + current_quat[:, :, 3] * goal_quat[:, :, 1] -
                        goal_quat[:, :, 2] * current_quat[:, :, 0] + current_quat[:, :, 2] * goal_quat[:, :, 0])
            error_quat[:, :, 2] = r_w[:, :, 0] * (
                        -goal_quat[:, :, 3] * current_quat[:, :, 2] + current_quat[:, :, 3] * goal_quat[:, :, 2] -
                        goal_quat[:, :, 0] * current_quat[:, :, 1] + current_quat[:, :, 0] * goal_quat[:, :, 1])

        if reach_offset:
            error_position += offset_position.unsqueeze(1)
            error_quat += offset_rotation.unsqueeze(1)

        vec_weight = vec_weight.unsqueeze(1)

        error_position = vec_weight[:, :, :3] * error_position
        error_quat = vec_weight[:, :, 3:] * error_quat

        result_vec = torch.cat([error_position, error_quat], dim=2)

        if project_distance:
            result_vec[:,:,3:] = self.transform_error_quat(goal_quat, error_quat)
            result_vec[:, :, :3]= self.transform_error_quat(goal_quat, error_position)
        else:
            result_vec[:, :, 3:] = error_position
            result_vec[:, :, :3] = error_quat
            #
        #  = torch.cat([error_position, error_quat])
        #
        # if project_distance:
        #     result_vec[3:] = self.transform_error_quat(goal_quat, error_quat)
        #     result_vec[0:3] = self.transform_error_quat(goal_quat, error_position)
        # else:
        #     result_vec[:3] = error_position
        #     result_vec[3:] = error_quat

        return result_vec

    def compute_pose_distance(self,
                              project_distance,
                              use_metric,
                              distance_vec,
                              current_position,
                              goal_position,
                              current_quat,
                              goal_quat,
                              vec_weight,
                              vec_convergence,
                              position_weight,
                              rotation_weight,
                              p_alpha,
                              r_alpha,
                              offset_position,
                              offset_rotation,
                              reach_offset):
        distance_vec = self.compute_pose_distance_vector(project_distance,
                                                         goal_position,
                                                         goal_quat,
                                                         current_position,
                                                         current_quat,
                                                         vec_weight,
                                                         offset_position,
                                                         offset_rotation,
                                                         reach_offset)

        position_distance = torch.sum(distance_vec[:, :, :3] ** 2, dim=2)
        rotation_distance = torch.sum(distance_vec[:, :, 3:] ** 2, dim=2)

        distance = torch.zeros_like(position_distance)

        # Apply the conditions element-wise
        mask_rotation = rotation_distance > vec_convergence[0] ** 2
        mask_position = position_distance > vec_convergence[1] ** 2

        if use_metric:
            distance += rotation_weight * torch.log2(
                torch.cosh(r_alpha * torch.sqrt(rotation_distance) * mask_rotation.float())) * mask_rotation
            distance += position_weight * torch.log2(
                torch.cosh(p_alpha * torch.sqrt(position_distance) * mask_position.float())) * mask_position
        else:
            distance += rotation_weight * torch.sqrt(rotation_distance) * mask_rotation
            distance += position_weight * torch.sqrt(position_distance) * mask_position

        # Calculate distances with the mask
        rotation_distance = torch.sqrt(rotation_distance) * mask_rotation
        position_distance = torch.sqrt(position_distance) * mask_position

        return distance, position_distance, rotation_distance

        # distance = 0.0
        # if rotation_distance > vec_convergence[0] ** 2:
        #     rotation_distance = torch.sqrt(rotation_distance)
        #     if use_metric:
        #         distance += rotation_weight * torch.log2(torch.cosh(r_alpha * rotation_distance))
        #     else:
        #         distance += rotation_weight * rotation_distance
        #
        # if position_distance > vec_convergence[1] ** 2:
        #     position_distance = torch.sqrt(position_distance)
        #     if use_metric:
        #         distance += position_weight * torch.log2(torch.cosh(p_alpha * position_distance))
        #     else:
        #         distance += position_weight * position_distance
        #
        # return distance, position_distance, rotation_distance

    def goalset_pose_distance_kernel(
            self,
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_q_vec,
            out_gidx,
            current_position,
            goal_position,
            current_quat,
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            offset_waypoint,
            offset_tstep_fraction,
            batch_pose_idx,
            mode,
            num_goals,
            batch_size,
            horizon,
            write_distance=False,
            project_distance=False,
            use_metric=False,
            write_grad=False
    ):
        t_idx = torch.arange(batch_size * horizon, device=current_position.device)
        batch_idx = t_idx // horizon
        h_idx = t_idx % horizon

        valid_idx = (batch_idx < batch_size) & (h_idx < horizon)
        batch_idx = batch_idx[valid_idx]
        h_idx = h_idx[valid_idx]

        # print("________________", batch_size, horizon)


        if batch_idx.numel() == 0:
            return

        position = current_position.view(batch_size, horizon, 3)[batch_idx, h_idx]
        quat = current_quat.view(batch_size, horizon, 4)[batch_idx, h_idx]
        quat = torch.cat([quat[:, 1:], quat[:, :1]], dim=1)  # Rearrange quaternion

        print(position.shape, quat.shape)

        # print(current_position.shape, current_position.shape)
        # position = current_position.view(batch_size, horizon, 3)
        # quat = current_quat.view(batch_size, horizon, 4)
        # print(position.shape, quat.shape)

        rotation_weight = weight[0]
        position_weight = weight[1]
        r_w_alpha = weight[2]
        p_w_alpha = weight[3]
        reach_offset = False
        offset_tstep_ratio = offset_tstep_fraction[0]
        offset_tstep = int(torch.floor(offset_tstep_ratio * horizon))
        d_vec_weight = torch.cat([vec_weight[0:3], vec_weight[3:6]])
        d_vec_weight = d_vec_weight.unsqueeze(0).expand(batch_size, -1)

        # print(d_vec_weight)
        valid_h_idx_mask = (h_idx < horizon - 1) & (h_idx != horizon - offset_tstep)

        d_vec_weight[valid_h_idx_mask, 0:3] *= run_vec_weight[0:3]
        d_vec_weight[valid_h_idx_mask, 3:6] *= run_vec_weight[3:6]

        print(weight, position_weight, rotation_weight, run_weight)

        # if (h_idx < horizon - 1) & (h_idx != horizon - offset_tstep):
        #     d_vec_weight[0:3] *= run_vec_weight[0:3]
        #     d_vec_weight[3:6] *= run_vec_weight[3:6]

        if not write_distance:
            position_weight *= run_weight[0, 0]
            rotation_weight *= run_weight[0, 0]
            sum_weight = torch.sum(d_vec_weight, dim=1)

            if ((position_weight == 0.0) & (rotation_weight == 0.0)) | (sum_weight.any() == 0.0):
                return

        print(horizon, offset_tstep, h_idx.shape)

        if (horizon > 1) & (offset_tstep >= 0) & (h_idx.any() == horizon - offset_tstep):
            reach_offset = True

        best_distance = torch.full((batch_size,), float('inf'), device=current_position.device)
        best_position_distance = torch.zeros(batch_size, device=current_position.device)
        best_rotation_distance = torch.zeros(batch_size, device=current_position.device)
        best_distance_vec = torch.zeros((batch_size, 6), device=current_position.device)
        d_vec_convergence = vec_convergence[0:2]

        best_idx = -torch.ones(batch_size, dtype=torch.int32, device=current_position.device)

        offset = batch_pose_idx[batch_idx]

        if mode == "BATCH_GOALSET" or mode == "BATCH_GOAL":
            offset *= num_goals

        for k in range(num_goals):
            l_goal_position = goal_position.view(-1, 3)[offset + k]
            l_goal_quat = goal_quat.view(-1, 4)[offset + k]
            l_goal_quat = torch.cat([l_goal_quat[1:], l_goal_quat[:1]])

            distance_vec = torch.zeros(6, device=current_position.device)
            distance, position_distance, rotation_distance = self.compute_pose_distance(
                project_distance,
                use_metric,
                distance_vec,
                position,
                l_goal_position,
                quat,
                l_goal_quat,
                d_vec_weight,
                d_vec_convergence,
                position_weight,
                rotation_weight,
                p_w_alpha,
                r_w_alpha,
                offset_waypoint[3:6],
                offset_waypoint[0:3],
                reach_offset
            )

            # print("___________________________", distance, position_distance, rotation_distance)

            better_distance_mask = distance <= best_distance
            best_distance = torch.where(better_distance_mask, distance, best_distance)
            best_position_distance = torch.where(better_distance_mask, position_distance, best_position_distance)
            best_rotation_distance = torch.where(better_distance_mask, rotation_distance, best_rotation_distance)
            best_idx = torch.where(better_distance_mask, torch.full_like(best_idx, k), best_idx)

            if write_grad:
                best_distance_vec[better_distance_mask] = distance_vec

        out_distance[batch_idx * horizon + h_idx] = best_distance

        if write_distance:
            out_position_distance[batch_idx * horizon + h_idx] = torch.where(position_weight == 0.0, torch.tensor(0.0,
                                                                                                                  device=current_position.device),
                                                                             best_position_distance)
            out_rotation_distance[batch_idx * horizon + h_idx] = torch.where(rotation_weight == 0.0, torch.tensor(0.0,
                                                                                                                  device=current_position.device),
                                                                             best_rotation_distance)
        out_gidx[batch_idx * horizon + h_idx] = best_idx

        if write_grad:
            if write_distance:
                position_weight = 1
                rotation_weight = 1

    def inv_transform_point(self, frame_pos, frame_quat, points):
        """
        Transform the point using the inverse of the given frame position and quaternion.

        Args:
            frame_pos (torch.Tensor): Tensor of shape (3,) representing the frame position.
            frame_quat (torch.Tensor): Tensor of shape (4,) representing the frame quaternion.
            point (torch.Tensor): Tensor of shape (3,) representing the point to be transformed.

        Returns:
            torch.Tensor: Transformed point of shape (3,).
        """
        # Negate the quaternion vector part
        negation_mask = torch.tensor([[-1, -1, -1, 1]], dtype=frame_quat.dtype, device=frame_quat.device)
        q = frame_quat * negation_mask

        # q = torch.tensor([-1 * frame_quat[:, :, 0], -1 * frame_quat[:, :, 1], -1 * frame_quat[:, :, 2], frame_quat[:, :, 3]],
        #                  dtype=frame_quat.dtype, device=frame_quat.device)

        # Perform the transformation
        transformed_points = torch.zeros_like(points)

        p = -self.transform_point(torch.zeros_like(frame_pos), q, frame_pos)

        # if q[0] != 0 or q[1] != 0 or q[2] != 0:
        #     x, y, z = point[0], point[1], point[2]
        #     qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        #
        #     transformed_point[0] = p[
        #                                0] + qw * qw * x + 2 * qy * qw * z - 2 * qz * qw * y + qx * qx * x + 2 * qy * qx * y + 2 * qz * qx * z - qz * qz * x - qy * qy * x
        #     transformed_point[1] = p[
        #                                1] + 2 * qx * qy * x + qy * qy * y + 2 * qz * qy * z + 2 * qw * qz * x - qz * qz * y + qw * qw * y - 2 * qx * qw * z - qx * qx * y
        #     transformed_point[2] = p[
        #                                2] + 2 * qx * qz * x + 2 * qy * qz * y + qz * qz * z - 2 * qw * qy * x - qy * qy * z + 2 * qw * qx * y - qx * qx * z + qw * qw * z
        # else:
        #     transformed_point = p + point
        #
        # return transformed_point

        # Ensure points are correctly broadcasted to match the batch and horizon dimensions
        points_expanded = points.unsqueeze(1).expand(-1, frame_pos.size(1), -1)

        # Initialize the transformed points tensor
        transformed_points = torch.zeros_like(points_expanded)

        # Extract quaternion components
        qw, qx, qy, qz = q[:, :, 3], q[:, :, 0], q[:, :, 1], q[:, :, 2]

        # Extract point components
        x, y, z = points_expanded[:, :, 0], points_expanded[:, :, 1], points_expanded[:, :, 2]

        # Compute transformed points
        transformed_points[:, :, 0] = (p[:, :, 0] + qw * qw * x + 2 * qy * qw * z - 2 * qz * qw * y +
                                       qx * qx * x + 2 * qy * qx * y + 2 * qz * qx * z -
                                       qz * qz * x - qy * qy * x)
        transformed_points[:, :, 1] = (p[:, :, 1] + 2 * qx * qy * x + qy * qy * y + 2 * qz * qy * z +
                                       2 * qw * qz * x - qz * qz * y + qw * qw * y -
                                       2 * qx * qw * z - qx * qx * y)
        transformed_points[:, :, 2] = (p[:, :, 2] + 2 * qx * qz * x + 2 * qy * qz * y + qz * qz * z -
                                       2 * qw * qy * x - qy * qy * z + 2 * qw * qx * y -
                                       qx * qx * z + qw * qw * z)

        # Handle the case where qx, qy, or qz is zero
        non_zero_mask = (qx != 0) | (qy != 0) | (qz != 0)
        transformed_points[~non_zero_mask] = p[~non_zero_mask] + points_expanded[~non_zero_mask]

        return transformed_points

    def transform_point(self, frame_pos, frame_quat, points):
        """
        # Transform the point using the given frame position and quaternion.
        #
        # Args:
        #     frame_pos (torch.Tensor): Tensor of shape (3,) representing the frame position.
        #     frame_quat (torch.Tensor): Tensor of shape (4,) representing the frame quaternion.
        #     point (torch.Tensor): Tensor of shape (3,) representing the point to be transformed.
        #
        # Returns:
        #     torch.Tensor: Transformed point of shape (3,).

        Transform the points using the given frame positions and quaternions.

        Args:
            frame_pos (torch.Tensor): Tensor of shape (batch_size, 3) representing the frame positions.
            frame_quat (torch.Tensor): Tensor of shape (batch_size, 4) representing the frame quaternions.
            points (torch.Tensor): Tensor of shape (batch_size, 3) representing the points to be transformed.

        Returns:
            torch.Tensor: Transformed points of shape (batch_size, 3).
        """
        batch_size = frame_pos.size(0)

        # Normalize the quaternion
        q = frame_quat / frame_quat.norm(dim=2, keepdim=True)

        # Extract quaternion components
        qw = q[:, :, 3]
        qx = q[:, :, 0]
        qy = q[:, :, 1]
        qz = q[:, :, 2]

        # Compute the quaternion rotation matrix
        R = torch.stack([
            1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy
        ], dim=1).reshape(frame_pos.size(0), frame_pos.size(1), 3, 3)

        # Apply the rotation and translation
        rotated_points = torch.matmul(R, points.unsqueeze(-1)).squeeze(-1)

        # Create masks for points where qx, qy, or qz is zero
        no_rotation_mask = (q[:, :, 0] != 0) | (q[:, :, 1] != 0) | (q[:, :, 2] != 0)


        # Initialize the transformed points with the rotated points
        transformed_points = rotated_points.clone()

        # Apply only the translation for points where qx, qy, or qz is zero
        transformed_points[no_rotation_mask] = points[no_rotation_mask] + frame_pos[no_rotation_mask]

        # For the rest, apply the translation after rotation
        transformed_points[~no_rotation_mask] += frame_pos[~no_rotation_mask]

        # q = frame_quat
        # p = frame_pos
        #
        # if q[0] != 0 or q[1] != 0 or q[2] != 0:
        #     x, y, z = point[0], point[1], point[2]
        #     qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        #
        #     transformed_point_x = p[
        #                               0] + qw * qw * x + 2 * qy * qw * z - 2 * qz * qw * y + qx * qx * x + 2 * qy * qx * y + 2 * qz * qx * z - qz * qz * x - qy * qy * x
        #     transformed_point_y = p[
        #                               1] + 2 * qx * qy * x + qy * qy * y + 2 * qz * qy * z + 2 * qw * qz * x - qz * qz * y + qw * qw * y - 2 * qx * qw * z - qx * qx * y
        #     transformed_point_z = p[
        #                               2] + 2 * qx * qz * x + 2 * qy * qz * y + qz * qz * z - 2 * qw * qy * x - qy * qy * z + 2 * qw * qx * y - qx * qx * z + qw * qw * z
        #
        #     transformed_point = torch.tensor([transformed_point_x, transformed_point_y, transformed_point_z],
        #                                      dtype=frame_pos.dtype, device=frame_pos.device)
        # else:
        #     transformed_point = p + point
        #
        return transformed_points

    def inv_transform_quat(self, frame_quat, quat):
        # """
        # Transform the quaternion using the inverse of the given frame quaternion.
        #
        # Args:
        #     frame_quat (torch.Tensor): Tensor of shape (4,) representing the frame quaternion.
        #     quat (torch.Tensor): Tensor of shape (4,) representing the quaternion to be transformed.
        #
        # Returns:
        #     torch.Tensor: Transformed quaternion of shape (4,).
        # """
        # # Invert the frame quaternion
        # q = torch.tensor([-1 * frame_quat[0], -1 * frame_quat[1], -1 * frame_quat[2], frame_quat[3]],
        #                  dtype=frame_quat.dtype, device=frame_quat.device)
        #
        # # Initialize the transformed quaternion
        # transformed_quat = torch.zeros_like(quat)
        #
        # if q[0] != 0 or q[1] != 0 or q[2] != 0:
        #     # Multiply quaternions q and quat
        #     qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        #     pw, px, py, pz = quat[3], quat[0], quat[1], quat[2]
        #
        #     transformed_quat[3] = qw * pw - qx * px - qy * py - qz * pz
        #     transformed_quat[0] = qw * px + pw * qx + qy * pz - py * qz
        #     transformed_quat[1] = qw * py + pw * qy + qz * px - pz * qx
        #     transformed_quat[2] = qw * pz + pw * qz + qx * py - px * qy
        # else:
        #     transformed_quat = quat
        #
        # return transformed_quat

        """
            Transform the quaternions using the inverse of the given frame quaternions.

            Args:
                frame_quat (torch.Tensor): Tensor of shape (batch_size, horizon, 4) representing the frame quaternions.
                quat (torch.Tensor): Tensor of shape (batch_size, horizon, 4) representing the quaternions to be transformed.

            Returns:
                torch.Tensor: Transformed quaternions of shape (batch_size, horizon, 4).
            """
        batch_size, horizon = frame_quat.size(0), frame_quat.size(1)

        # Invert the frame quaternions
        q = torch.cat([-frame_quat[:, :, :3], frame_quat[:, :, 3:4]], dim=2)

        # Initialize the transformed quaternions
        quat_expanded = quat.unsqueeze(1).expand(-1, frame_quat.size(1), -1)

        transformed_quat = torch.zeros_like(quat_expanded)

        # Check if any quaternion components are non-zero
        non_zero_mask = (q[:, :, 0] != 0) | (q[:, :, 1] != 0) | (q[:, :, 2] != 0)

        # Extract components for multiplication
        qw, qx, qy, qz = q[:, :, 3], q[:, :, 0], q[:, :, 1], q[:, :, 2]
        pw, px, py, pz = quat_expanded[:, :, 3], quat_expanded[:, :, 0], quat_expanded[:, :, 1], quat_expanded[:, :, 2]

        # Perform quaternion multiplication for non-zero components
        transformed_quat[:, :, 3] = qw * pw - qx * px - qy * py - qz * pz
        transformed_quat[:, :, 0] = qw * px + pw * qx + qy * pz - py * qz
        transformed_quat[:, :, 1] = qw * py + pw * qy + qz * px - pz * qx
        transformed_quat[:, :, 2] = qw * pz + pw * qz + qx * py - px * qy

        # For components that are zero, retain the original quaternion
        transformed_quat[~non_zero_mask] = quat_expanded[~non_zero_mask]

        return transformed_quat

    # def transform_error_quat(self, q, error):
    #     """
    #     Transform the error vector using the given quaternion.
    #
    #     Args:
    #         q (torch.Tensor): Tensor of shape (4,) representing the quaternion (qx, qy, qz, qw).
    #         error (torch.Tensor): Tensor of shape (3,) representing the error vector.
    #
    #     Returns:
    #         torch.Tensor: Transformed error vector of shape (3,).
    #     """
    #     result = torch.zeros(3, dtype=error.dtype, device=error.device)
    #
    #     if q[0] != 0 or q[1] != 0 or q[2] != 0:
    #         qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    #         ex, ey, ez = error[0], error[1], error[2]
    #
    #         result[0] = qw * qw * ex + 2 * qy * qw * ez - 2 * qz * qw * ey + qx * qx * ex + 2 * qy * qx * ey + 2 * qz * qx * ez - qz * qz * ex - qy * qy * ex
    #         result[1] = 2 * qx * qy * ex + qy * qy * ey + 2 * qz * qy * ez + 2 * qw * qz * ex - qz * qz * ey + qw * qw * ey - 2 * qx * qw * ez - qx * qx * ey
    #         result[2] = 2 * qx * qz * ex + 2 * qy * qz * ey + qz * qz * ez - 2 * qw * qy * ex - qy * qy * ez + 2 * qw * qx * ey - qx * qx * ez + qw * qw * ez
    #     else:
    #         result = error
    #
    #     return result

    def transform_error_quat(self, q, error):
        """
        Transform the error vector using the given quaternion for batch processing.

        Args:
            q (torch.Tensor): Tensor of shape (batch_size, horizon, 4) representing the quaternions (qx, qy, qz, qw).
            error (torch.Tensor): Tensor of shape (batch_size, horizon, 3) representing the error vectors.

        Returns:
            torch.Tensor: Transformed error vectors of shape (batch_size, horizon, 3).
        """
        batch_size, horizon = q.size(0), q.size(1)

        # Initialize the result tensor
        result = torch.zeros_like(error)

        # Check if any quaternion components are non-zero
        non_zero_mask = (q[:, :, 0] != 0) | (q[:, :, 1] != 0) | (q[:, :, 2] != 0)

        # Extract quaternion components
        qw, qx, qy, qz = q[:, :, 3], q[:, :, 0], q[:, :, 1], q[:, :, 2]
        ex, ey, ez = error[:, :, 0], error[:, :, 1], error[:, :, 2]

        # Apply the transformation for non-zero components
        result[:, :, 0] = (qw * qw * ex + 2 * qy * qw * ez - 2 * qz * qw * ey +
                           qx * qx * ex + 2 * qy * qx * ey + 2 * qz * qx * ez -
                           qz * qz * ex - qy * qy * ex)
        result[:, :, 1] = (2 * qx * qy * ex + qy * qy * ey + 2 * qz * qy * ez +
                           2 * qw * qz * ex - qz * qz * ey + qw * qw * ey -
                           2 * qx * qw * ez - qx * qx * ey)
        result[:, :, 2] = (2 * qx * qz * ex + 2 * qy * qz * ey + qz * qz * ez -
                           2 * qw * qy * ex - qy * qy * ez + 2 * qw * qx * ey -
                           qx * qx * ez + qw * qw * ez)

        # For components that are zero, retain the original error
        result[~non_zero_mask] = error[~non_zero_mask]

        return result

    def pose_distance(self,
                      out_distance,
                      out_position_distance,
                      out_rotation_distance,
                      distance_p_vector,
                      distance_q_vector,
                      out_gidx,
                      current_position,
                      goal_position,
                      current_quat,
                      goal_quat,
                      vec_weight,
                      weight,
                      vec_convergence,
                      run_weight,
                      run_vec_weight,
                      offset_waypoint,
                      offset_tstep_fraction,
                      batch_pose_idx,
                      batch_size,
                      horizon,
                      mode,
                      num_goals=1,
                      compute_grad=False,
                      write_distance=True,
                      use_metric=False,
                      project_distance=True):

        assert batch_pose_idx.size(0) == batch_size, "Index buffer size is different from batch size"

        # Define kernel launch parameters
        bh = batch_size * horizon
        threads_per_block = min(bh, 128)
        blocks_per_grid = (bh + threads_per_block - 1) // threads_per_block

        # Get the current CUDA stream
        stream = torch.cuda.current_stream()

        # Define a function to dispatch the kernel
        def dispatch_kernel(write_distance, use_metric, project_distance):
            return self.goalset_pose_distance_kernel(
                out_distance, out_position_distance, out_rotation_distance,
                distance_p_vector, distance_q_vector, out_gidx,
                current_position, goal_position, current_quat, goal_quat,
                vec_weight, weight, vec_convergence, run_weight, run_vec_weight,
                offset_waypoint, offset_tstep_fraction, batch_pose_idx,
                mode, num_goals, batch_size, horizon,
                write_distance=write_distance,
                project_distance=project_distance,
                use_metric=use_metric,
                write_grad=compute_grad
            )

        # Conditional logic to call the kernel with the appropriate parameters
        if project_distance:
            if use_metric:
                if write_distance:
                    dispatch_kernel(write_distance=True, use_metric=True, project_distance=True)
                else:
                    dispatch_kernel(write_distance=False, use_metric=True, project_distance=True)
            else:
                if write_distance:
                    dispatch_kernel(write_distance=True, use_metric=False, project_distance=True)
                else:
                    dispatch_kernel(write_distance=False, use_metric=False, project_distance=True)
        else:
            if use_metric:
                if write_distance:
                    dispatch_kernel(write_distance=True, use_metric=True, project_distance=False)
                else:
                    dispatch_kernel(write_distance=False, use_metric=True, project_distance=False)
            else:
                if write_distance:
                    dispatch_kernel(write_distance=True, use_metric=False, project_distance=False)
                else:
                    dispatch_kernel(write_distance=False, use_metric=False, project_distance=False)

        # Synchronize the stream to ensure all operations are complete
        stream.synchronize()

        return [out_distance, out_position_distance, out_rotation_distance,
                distance_p_vector, distance_q_vector, out_gidx]
