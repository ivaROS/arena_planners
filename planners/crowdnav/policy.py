"""SARL policy extracted from vita-epfl/CrowdNav for Arena inference."""

from __future__ import annotations

import abc
import itertools
import logging

import numpy as np
import torch
import torch.nn as nn

from state import ActionRot, ActionXY, FullState, ObservableState


def _mlp(input_dim: int, mlp_dims: list[int], last_relu: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim] + mlp_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _Policy:
    """Abstract base for all CrowdNav policies."""

    def __init__(self):
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        return

    @staticmethod
    def reach_destination(state) -> bool:
        ss = state.self_state
        return np.linalg.norm((ss.py - ss.gy, ss.px - ss.gx)) < ss.radius


class _CADRL(_Policy):
    """CADRL base class, includes rotate() and propagate()."""

    def __init__(self):
        super().__init__()
        self.name = "CADRL"
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get("cadrl", "mlp_dims").split(", ")]
        self.model = nn.Sequential(_mlp(self.joint_state_dim, mlp_dims))
        self.multiagent_training = config.getboolean("cadrl", "multiagent_training")

    def set_common_parameters(self, config):
        self.gamma = config.getfloat("rl", "gamma")
        self.kinematics = config.get("action_space", "kinematics")
        self.sampling = config.get("action_space", "sampling")
        self.speed_samples = config.getint("action_space", "speed_samples")
        self.rotation_samples = config.getint("action_space", "rotation_samples")
        self.query_env = config.getboolean("action_space", "query_env")
        self.cell_num = config.getint("om", "cell_num")
        self.cell_size = config.getfloat("om", "cell_size")
        self.om_channel_size = config.getint("om", "om_channel_size")

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref: float):
        holonomic = self.kinematics == "holonomic"
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)
        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            return ObservableState(
                state.px + action.vx * self.time_step,
                state.py + action.vy * self.time_step,
                action.vx,
                action.vy,
                state.radius,
            )
        if isinstance(state, FullState):
            if self.kinematics == "holonomic":
                return FullState(
                    state.px + action.vx * self.time_step,
                    state.py + action.vy * self.time_step,
                    action.vx,
                    action.vy,
                    state.radius,
                    state.gx,
                    state.gy,
                    state.v_pref,
                    state.theta,
                )
            next_theta = state.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            return FullState(
                state.px + next_vx * self.time_step,
                state.py + next_vy * self.time_step,
                next_vx,
                next_vy,
                state.radius,
                state.gx,
                state.gy,
                state.v_pref,
                next_theta,
            )
        raise ValueError("Type error")

    def rotate(self, state: torch.Tensor) -> torch.Tensor:
        """Transform to agent-centric coordinates. Input: (batch, state_length)."""
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))
        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == "unicycle":
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(
            torch.cat(
                [(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).reshape((batch, -1))],
                dim=1,
            ),
            2,
            dim=1,
            keepdim=True,
        )
        return torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

    def predict(self, state):
        raise NotImplementedError


class _MultiHumanRL(_CADRL):
    """Multi-human extension of CADRL with attention-based value function."""

    def __init__(self):
        super().__init__()

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError("Phase, device attributes have to be set!")
        if self.phase == "train" and self.epsilon is None:
            raise AttributeError("Epsilon attribute has to be set in training phase")

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == "holonomic" else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == "train" and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float("-inf")
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy)) for human_state in state.human_states]
                    reward = self._compute_reward(next_self_state, next_human_states)
                batch_next_states = torch.cat(
                    [torch.Tensor([next_self_state + next_human_state]).to(self.device) for next_human_state in next_human_states],
                    dim=0,
                )
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self._build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError("Value network is not well trained.")

        if self.phase == "train":
            self.last_state = self._transform(state)

        return max_action

    def _compute_reward(self, nav, humans) -> float:
        dmin = float("inf")
        collision = False
        for human in humans:
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            return -0.25
        if reaching_goal:
            return 1.0
        if dmin < 0.2:
            return (dmin - 0.2) * 0.5 * self.time_step
        return 0.0

    def _transform(self, state) -> torch.Tensor:
        state_tensor = torch.cat(
            [torch.Tensor([state.self_state + human_state]).to(self.device) for human_state in state.human_states],
            dim=0,
        )
        if self.with_om:
            occupancy_maps = self._build_occupancy_maps(state.human_states)
            return torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        return self.rotate(state_tensor)

    def input_dim(self) -> int:
        return self.joint_state_dim + (self.cell_num**2 * self.om_channel_size if self.with_om else 0)

    def _build_occupancy_maps(self, human_states) -> torch.Tensor:
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate(
                [np.array([(o.px, o.py, o.vx, o.vy)]) for o in human_states if o is not human],
                axis=0,
            )
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float("-inf")
            other_x_index[other_x_index >= self.cell_num] = float("-inf")
            other_y_index[other_y_index < 0] = float("-inf")
            other_y_index[other_y_index >= self.cell_num] = float("-inf")
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num**2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num**2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num**2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, _cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])
        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()


class _SARLValueNetwork(nn.Module):
    """Attention-based value network for SARL."""

    def __init__(  # noqa: PLR0913
        self,
        input_dim,
        self_state_dim,
        mlp1_dims,
        mlp2_dims,
        mlp3_dims,
        attention_dims,
        with_global_state,
        cell_size,
        cell_num,
    ):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = _mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = _mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = _mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = _mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = _mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Input: (batch_size, num_humans, rotated_state_length)."""
        size = state.shape
        self_state = state[:, 0, : self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        return self.mlp3(joint_state)


class SARLPolicy(_MultiHumanRL):
    """SARL: Socially Attentive Reinforcement Learning policy (inference only)."""

    def __init__(self):
        super().__init__()
        self.name = "SARL"

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get("sarl", "mlp1_dims").split(", ")]
        mlp2_dims = [int(x) for x in config.get("sarl", "mlp2_dims").split(", ")]
        mlp3_dims = [int(x) for x in config.get("sarl", "mlp3_dims").split(", ")]
        attention_dims = [int(x) for x in config.get("sarl", "attention_dims").split(", ")]
        self.with_om = config.getboolean("sarl", "with_om")
        with_global_state = config.getboolean("sarl", "with_global_state")
        self.model = _SARLValueNetwork(
            self.input_dim(),
            self.self_state_dim,
            mlp1_dims,
            mlp2_dims,
            mlp3_dims,
            attention_dims,
            with_global_state,
            self.cell_size,
            self.cell_num,
        )
        self.multiagent_training = config.getboolean("sarl", "multiagent_training")
        if self.with_om:
            self.name = "OM-SARL"
        logging.info("Policy: %s %s global state", self.name, "w/" if with_global_state else "w/o")

    def get_attention_weights(self):
        return self.model.attention_weights
