"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        assert len(hidden_dims) >= 2
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        for x, y in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(x, y))
            layers.append(nn.ReLU())
        # output layer: predict chunk_size actions
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim)) 
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_rl_weights)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # In MSE policy, training has the same forward process as inference
        y = self.sample_actions(state)
        d = y - action_chunk
        return d.square().mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        assert state.shape[-1] == self.state_dim    # (... state_dim)
        y = self.layers(state)  # (... chunk_size * action_dim)
        assert y.shape[-1] == self.chunk_size * self.action_dim
        return y.unflatten(dim=-1, sizes=(self.chunk_size, self.action_dim))


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        assert len(hidden_dims) >= 2
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        # input: (state, \tau)
        layers = [nn.Linear(state_dim + chunk_size * action_dim + 1, hidden_dims[0]), nn.ReLU()]
        for x, y in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(x, y))
            layers.append(nn.ReLU())
        # output layer: predict chunk_size actions
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim)) 
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_rl_weights)

    def compute_loss(
        self,
        state: torch.Tensor,    # (batch_size, state_dim)
        action_chunk: torch.Tensor, # (batch_size, chunk_size, action_dim)
    ) -> torch.Tensor:
        action_chunk = action_chunk.flatten(-2, -1)
        # `randn` normal distribution
        action_init = torch.randn_like(action_chunk)    # (batch_size, chunk_size * action_dim)
        # `rand` uniform distribution
        t = torch.rand(state.shape[:-1]).unsqueeze(-1) # (batch_size, 1)
        action_t = action_chunk * t + action_init * (1 - t)
        x = torch.cat([state, action_t, t], dim=-1)
        y = self.forward(x)
        target = action_chunk - action_init
        d = y - target
        return d.square().mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_t = torch.randn((*state.shape[:-1], self.chunk_size * self.action_dim))  # (batch_size, chunk_size*action_dim)
        for i in range(num_steps):
            t = torch.full(state.shape[:-1], i / num_steps).unsqueeze(-1)   # (batch_size, 1)
            x = torch.cat([state, action_t, t], dim=-1) # (batch_size, state_dim+chunk_size*action_dim+1)
            y = self.forward(x)
            action_t += 1 / num_steps * y
        return action_t.unflatten(-1, (self.chunk_size, self.action_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (... state_dim+chunk_size*action_dim+1)
        assert x.shape[-1] == self.state_dim + self.chunk_size * self.action_dim + 1   
        return self.layers(x)  # (... chunk_size * action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

def init_rl_weights(m):
    if isinstance(m, nn.Linear):
        # Standard Kaiming for all layers
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
