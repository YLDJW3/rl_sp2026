from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        expectile: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.expectile = expectile

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation).mode  # Take the mean (mode) action
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    @staticmethod
    def iql_expectile_loss(
        adv: torch.Tensor, expectile: float,
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Implement the expectile loss
        indicator = (adv > 0).int()
        w = (expectile - indicator).abs()
        assert w.shape == adv.shape
        return w * adv.square()

    @torch.compile
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) with expectile regression
        """
        # TODO(student): Compute the value loss
        v = self.value(observations)    # (B, )
        q = torch.amin(self.target_critic(observations, actions), dim=0) # (B, )
        loss = self.iql_expectile_loss(v - q, self.expectile).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        q = self.critic(observations, actions) # (2, B)
        v = self.value(next_observations) # (B, )
        target = rewards + (1 - dones) * self.discount * v # (B, )
        loss = (q - target).square().mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor using advantage-weighted regression
        """
        # TODO(student): Compute the actor loss
        dist : torch.distributions.Distribution = self.actor(observations)
        log_prob = dist.log_prob(actions) # (B, )
        q = torch.amin(self.critic(observations, actions), dim=0)    # (B, )
        v = self.value(observations)    # (B, )
        # M = 100 clips the weights to improve stability
        weight = torch.clamp(torch.exp((q - v) * self.alpha), max=100)   # (B, ) 
        assert weight.shape == log_prob.shape, weight.shape
        loss = (-weight * log_prob).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
            "mse": torch.mean((dist.mode - actions) ** 2),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_v(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + param.data * self.target_update_rate
            )
