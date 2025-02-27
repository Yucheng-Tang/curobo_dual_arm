
from dataclasses import dataclass
from itertools import product

# Third Party
import torch

# Local Folder
from .cost_base import CostBase, CostConfig

from curobo.cuda_robot_model.cuda_robot_model import TorchJacobian

import torch.nn as nn
from torch.nn import ModuleList

import os



@dataclass
class TrajExecCostConfig(CostConfig):
    use_trajectory_length: bool = False
    use_curobo_traj_time: bool = False

    def __post_init__(self):
        return super().__post_init__()


class TrajExecCost(CostBase, TrajExecCostConfig):
    def __init__(self, config: TrajExecCostConfig):
        TrajExecCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        # cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.FullLoader)
        trained_model_path_no_collision = os.path.dirname(__file__) + "/robdekon_ruckig_model.pt"
        trained_model_path = os.path.dirname(__file__) + "/robdekon_curobo_model.pt"

        # config = {
        #     "input_dim": 14,
        #     "hidden_dim": 256,
        #     "output_dim": 1,
        #     "num_layers": 3,
        #     "dropout": 0.0,
        #     "last_no_activ": True
        # }
        # self.model = MLP(config["input_dim"], config["hidden_dim"], config["output_dim"], config["num_layers"],
        #                  config["dropout"], config["last_no_activ"])
        # model_path = os.path.dirname(__file__) + "/traj_exec_model.pt"
        # self.model.load_state_dict(torch.load(model_path, map_location=self.tensor_args.device))
        # self.model.to(self.tensor_args.device)

        config_no_collision = {
            "input_dim": 26,
            "hidden_dim": 512,
            "output_dim": 1,
            "num_layers": 6,
            "dropout": 0.0,
            "last_no_activ": True
        }
        self.model_no_collision = MLP(input_dim=config_no_collision["input_dim"], output_dim=config_no_collision["output_dim"],
                             num_layers=config_no_collision["num_layers"], hidden_dim=config_no_collision["hidden_dim"], last_no_activ=True)

        self.model_no_collision.load_state_dict(torch.load(trained_model_path_no_collision))
        self.model_no_collision.to(self.tensor_args.device)
        self.model_no_collision.eval()

        config = {
            "input_dim": 78,
            "hidden_dim": 256,
            "output_dim": 1,
            "num_layers": 5,
            "dropout": 0.01532498486383016, #  0.09687128680207124,
            "last_no_activ": True
        }

        self.model = SkipMLP(input_dim=config["input_dim"], output_dim=config["output_dim"],
                        num_layers=config["num_layers"], hidden_dim=config["hidden_dim"], last_no_activ=True)

        self.model.load_state_dict(torch.load(trained_model_path))
        self.model.to(self.tensor_args.device)
        self.model.eval()

        self.input_mean = torch.tensor([-0.787, -0.002, 0.005, -0.007, 0.005, -0.010,
                          0.013, -0.000, 0.009, -0.002, -0.012, 0.016,
                          0.010, -0.792, -0.004, 0.003, -0.002, -0.006,
                          -0.005, -0.010, 0.008, -0.005, -0.004, -0.009,
                          0.010, 0.010]).to(self.tensor_args.device)
        self.input_std = torch.tensor([1.615, 0.831, 1.383, 1.000, 1.408, 1.816, 1.713, 1.159, 1.709, 1.282,
              1.716, 1.007, 1.765, 1.614, 0.830, 1.384, 1.000, 1.406, 1.813, 1.711,
              1.159, 1.708, 1.275, 1.713, 1.006, 1.771]).to(self.tensor_args.device)

        self.output_mean = torch.tensor([3.7121287610619467]).to(self.tensor_args.device)
        self.output_std = torch.tensor([0.509848970145711]).to(self.tensor_args.device)

        self.output_mean_curobo = torch.tensor([3.7264212487503414]).to(self.tensor_args.device)
        self.output_std_curobo = torch.tensor([0.5108514418395621]).to(self.tensor_args.device)
        # self.init_q = torch.zeros(
        #     (6, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        # )

        # self.i_mat = torch.ones(
        #     (6, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        # )
        #
        # self.delta_vector = torch.zeros(
        #     (64, 1, 1, 6, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        # )
        # x = [i for i in product(range(2), repeat=6)]
        # self.delta_vector[:, 0, 0, :, 0] = torch.as_tensor(
        #     x, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        # )
        # self.delta_vector[self.delta_vector == 0] = -1.0
        #
        # if self.cost_fn is None:
        #     if self.use_joint_limits and self.joint_limits is not None:
        #         self.cost_fn = self.joint_limited_manipulability_delta
        #     else:
        #         self.cost_fn = self.manipulability

    def forward(self, q, q_init=None):
        b, h, n = q.shape

        if q_init is None:
            q_init = torch.zeros_like(q[0, 0, :])
            # additional_zeros = torch.zeros(2, device=q_init.device)
            # q_init_extended = torch.cat((q_init, additional_zeros), dim=-1)

        # q_init_extended_batch = q_init.unsqueeze(0).unsqueeze(0).repeat(b, h, 1)
        if self.use_trajectory_length is False:
            if self.use_curobo_traj_time is True:
                # print("________________________curobo cost___________________________")
                # q_init_batch = q_init.unsqueeze(0).unsqueeze(0).repeat(b, h, 1)
                # zeros = torch.zeros(b, h, 1).to(q.device)
                # # print("!!!!!!!!!!!", q[:, :, :6].shape, zeros.shape, init_q_batch[:, :, :6].shape)
                # nn_in = torch.cat((q[:, :, :6], zeros, q_init_batch[:, :, :6], zeros), dim=-1)
                # # print(nn_in.shape)
                # nn_in = nn_in.squeeze()
                #
                # nn_in_1 = torch.cat((q[:, :, 6:], zeros, q_init_batch[:, :, 6:], zeros), dim=-1)
                # # print(nn_in_1)
                # nn_in_1 = nn_in_1.squeeze()
                #
                #
                # # nn_in = torch.cat((init_q, q), dim=-1)
                # #
                # # print(nn_in)
                #
                # score = self.model(nn_in)
                # # print("!!!!!!!traj_exec_time", score)
                #
                # score += self.model(nn_in_1)
                # # print("!!!!!!!traj_exec_cost", score)
                #
                # cost = self.weight * score
                # # print("!!!!!!!traj_exec_cost", cost)

                q_init_batch = q_init.unsqueeze(0).unsqueeze(0).repeat(b, h, 1)
                # nn_in = torch.cat((q,q_init_batch), dim=-1)

                nn_in = torch.cat((q_init_batch, q, torch.cos(q_init_batch), torch.sin(q_init_batch), torch.cos(q), torch.sin(q)), dim=-1)


                # nn_in = (nn_in - self.input_mean) / self.input_std

                nn_in = nn_in.squeeze()
                # print("!!!!!!!!!!!!!!", nn_in.shape)

                score = self.model(nn_in)
                score = score * self.output_std_curobo + self.output_mean_curobo
                # print("!!!!!!!traj_exec_time", score)

                # score += self.model(nn_in_1)
                # print("!!!!!!!traj_exec_cost", score)

                cost = self.weight * score
                # print("!!!!!!!traj_exec_cost", cost)
            else:
                q_init_batch = q_init.unsqueeze(0).unsqueeze(0).repeat(b, h, 1)
                # nn_in = torch.cat((q,q_init_batch), dim=-1)

                nn_in = torch.cat(
                    (q_init_batch, q), dim=-1)

                nn_in = (nn_in - self.input_mean) / self.input_std

                nn_in = nn_in.squeeze()
                # print("!!!!!!!!!!!!!!", nn_in.shape)

                score = self.model_no_collision(nn_in)
                score = score * self.output_std + self.output_mean
                # print("!!!!!!!traj_exec_time", score)

                # score += self.model(nn_in_1)
                # print("!!!!!!!traj_exec_cost no collision", score)

            cost = self.weight * score
        else:
            q_diff = (q-q_init) # / torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2, 10, 10, 10, 10, 10, 10, 10], device=q.device, dtype=q.dtype)
            score = torch.norm(q_diff, dim=-1)
            # print("!!!!!!!!joint_diff", score)
            cost = self.weight * score

        # # if self.use_nn:
        # #     q = q.view(b * h, n)
        # score = self.cost_fn(q, jac_batch, qdot)
        # # score = Manipulability.apply(q, jac_batch, qdot, robot_jac)
        #
        # print(score.shape)
        # # if self.use_nn:
        # #     score = score.view(b, h)
        # print(score.dtype, self.hinge_value.dtype)
        # # score[score > self.hinge_value] = self.hinge_value
        # score = torch.clamp(score, max=self.hinge_value)
        # score = (self.hinge_value / score) - 1
        # cost = self.weight * score

        return cost

    def manipulability(self, q, jac_batch, qdot=None):
        with torch.cuda.amp.autocast(enabled=False):
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))

        score[score != score] = 0.0

        return score

    def joint_limited_manipulability_delta(self, q, jac_batch, qdot=None):
        # q is [b,h,dof]
        q_low = q - self.joint_limits[:, 0]
        q_high = q - self.joint_limits[:, 1]

        d_h_1 = torch.square(self.joint_limits[:, 1] - self.joint_limits[:, 0]) * (q_low + q_high)
        d_h_2 = 4.0 * (torch.square(q_low) * torch.square(q_high))
        d_h = torch.div(d_h_1, d_h_2)

        dh_term = 1.0 / torch.sqrt(1 + torch.abs(d_h))
        f_ten = torch.tensor(1.0, **self.tensor_args)
        q_low = torch.abs(q_low)
        q_high = torch.abs(q_high)
        p_plus = torch.where(q_low > q_high, dh_term, f_ten).unsqueeze(-2)
        p_minus = torch.where(q_low > q_high, f_ten, dh_term).unsqueeze(-2)

        j_sign = torch.sign(jac_batch)
        l_delta = torch.sign(self.delta_vector) * j_sign

        L = torch.where(l_delta < 0.0, p_minus, p_plus)

        with torch.cuda.amp.autocast(enabled=False):
            w_J = L * jac_batch
            J_J_t = torch.matmul(w_J, w_J.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))

        # get actual score:
        min_score = torch.min(score, dim=0)[0]
        max_score = torch.max(score, dim=0)[0]
        score = min_score / max_score
        score[score != score] = 0.0
        return score

class TrajExecTime(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q: torch.Tensor,
            jac_batch: torch.Tensor,
            qdot: torch.Tensor,
            robot_jac: TorchJacobian,
    ):
        """Compute manipulability score and cost

        Args:
            ctx: context object for saving tensors for backward pass
            q: joint positions
            jac_batch: batch of Jacobians
            qdot: joint velocities

        Returns:
            cost: manipulability cost
        """
        b, h, n = q.shape
        with torch.cuda.amp.autocast(enabled=False):
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))

        score[score != score] = 0.0  # Handle NaNs

        hessian = robot_jac.calc_hessian(q)

        ctx.save_for_backward(score, jac_batch, q, qdot, hessian)

        return score

    @staticmethod
    def backward(ctx):
        score, jac_batch, q, qdot, hessian= ctx.saved_tensors

        b, h, n = q.shape

        grad_q = grad_jac_batch = grad_qdot = None

        if ctx.needs_input_grad[1]:
            grad_jac_batch = jac_batch + jac_batch.transpose(-2, -1)

        if ctx.needs_input_grad[0]:
            hessian_transpose = hessian.transpose(-2, -1)
            grad_q = hessian + hessian_transpose

        # # Calculate the gradient of the score with respect to the Jacobian
        # if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        #     grad_jac_batch = torch.zeros_like(jac_batch)
        #     grad_q = torch.zeros_like(q)
        #
        #     for i in range(b):
        #         for j in range(h):
        #             J = jac_batch[i, j]
        #             J_J_t = torch.matmul(J, J.transpose(-2, -1))
        #             device = J_J_t.device
        #             identity_matrix = torch.eye(J_J_t.size(-1)).to(device) * 1e-6
        #             J_J_t_inv = torch.inverse(
        #                 J_J_t + identity_matrix)  # Regularization for stability
        #             grad_score = 0.5 * torch.det(J_J_t) ** (-0.5) * torch.inverse(
        #                 J_J_t + identity_matrix)
        #             grad_jac = torch.matmul(grad_score, J)
        #             grad_jac_batch[i, j] = grad_jac
        #
        #             # Chain rule to compute gradient with respect to q
        #             for k in range(n):
        #                 grad_q[i, j, k] = torch.sum(grad_jac *
        #                                             torch.autograd.grad(J, q, grad_outputs=torch.ones_like(J),
        #                                                                 retain_graph=True)[0][i, j, k])
        #
        #     grad_jac_batch *= grad_output.unsqueeze(-1).unsqueeze(
        #         -1)  # Adjust gradient with respect to the output gradient
        #     grad_q *= grad_output.unsqueeze(-1)  # Adjust gradient with respect to the output gradient
        #
        # # Placeholder for gradients w.r.t qdot if necessary
        # if ctx.needs_input_grad[2]:
        #     grad_qdot = torch.zeros_like(qdot)

        return grad_q, grad_jac_batch, None

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, last_no_activ=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.last_no_activ = last_no_activ

        self.layers = ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Use sigmoid activation for the last layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            if i < self.num_layers - 1:
                x = torch.nn.functional.leaky_relu(x)
            elif not self.last_no_activ:
                x = torch.sigmoid(x)
        return x

class SkipMLP(MLP):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, last_no_activ=False):
        super(SkipMLP, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, last_no_activ)

    def forward(self, x):
        """
        Add skip connections to the MLP
        :param x:
        :return:
        """
        for i, layer in enumerate(self.layers):
            layer_out = layer(x)
            layer_out = nn.functional.dropout(layer_out, p=self.dropout, training=self.training)
            if i < self.num_layers - 1:
                layer_out = torch.nn.functional.leaky_relu(layer_out)
                x = layer_out if i == 0 else x + layer_out  # Fixme: should we use x.clone() here?

            elif not self.last_no_activ:
                x = torch.sigmoid(layer_out)
            else:
                x = layer_out

        return x