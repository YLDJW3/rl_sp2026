from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

        print(f"DQNAgent initialized:")
        print(f"  observation_shape={self.observation_shape}")
        print(f"  num_actions={self.num_actions}")
        print(f"  discount={self.discount}")
        print(f"  target_update_period={self.target_update_period}")
        print(f"  clip_grad_norm={self.clip_grad_norm}")
        print(f"  use_double_q={self.use_double_q}")

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # (Section 2.4): get the action from the critic using an epsilon-greedy strategy
        best_ac = torch.argmax(self.critic(observation), dim=-1)
        if np.random.rand() >= epsilon:
            action = best_ac
        else:
            action = np.random.randint(0, self.num_actions)
            action = torch.tensor(action)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,      # (N, ob_dim)
        action: torch.Tensor,   # (N, )
        reward: torch.Tensor,   # (N, )
        next_obs: torch.Tensor, # (N, ob_dim)
        done: torch.Tensor,     # (N, )
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # (Section 2.4): compute target values
            next_qa_values = self.target_critic(next_obs)   # (N, ac_dim)

            if self.use_double_q:
                # (Section 2.5): implement double-Q target action selection
                # choose action using critic network
                # evaluate value using target network
                next_action = torch.argmax(self.critic(next_obs), dim=-1)
            else:
                next_action = torch.argmax(next_qa_values, dim=-1)  # (N, )

            # if done[i] = 1 then next_q_values[i] will be 0
            next_q_values = (1 - done.int()) * next_qa_values[torch.arange(batch_size), next_action] # (N, )
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            assert reward.shape == next_q_values.shape
            target_values = reward + self.discount * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # (Section 2.4): train the critic with the target values
        qa_values = self.critic(obs)    # (N, ac_dim)
        q_values = qa_values[torch.arange(batch_size), action]    # (N, )
        assert q_values.shape == target_values.shape, q_values.shape
        loss = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # (Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        # ENDTODO

        return critic_stats
