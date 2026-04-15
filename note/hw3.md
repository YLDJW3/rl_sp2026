# Learning rate scheduler
1. Implemented in `configs/schedule.py`
2. LinearSchedule: decay from `initial_p` to `final_p` linearly with current timestep, and return `final_p` after `schedule_timesteps`
3. PiecewiseSchedule
    - `endpoints` is a list of fixpoints in (time, value) format
    - `interpolation` is a function describing the scheduled value between two fixpoints
    - `outside_value`: return value when time exceeds the time range given by `endpoints`
    - enumerate `endpoints` to find [start, end] endpoints of current time, call `iterpolation` to get the scheduled value of current time

# Epislon-greedy factor exploration scheduler
1. Starts $\epislon$ at a high value, close to random sampling, and decays it to a small value during training
2. Use `PiecewiseSchedule`
    ```python
    epsilon = exploration_schedule.value(step)
    ```

# Gradient clipping
```python
grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
    self.critic.parameters(), self.clip_grad_norm or float("inf")
)
```

# Replay buffer
1. `ReplayBuffer` use 5 arrays of `capacity` to store data $(s, a, r, s', d)$, observations, actions, rewards, next_observations, done
2. `MemoryEfficientReplayBuffer` share frames across consecutive transitions

# DQN
## Hyperparameter
1. Nerual network: `hidden_size` and `num_layers` 
2. Learning rate: `learning_rate`, `total_steps`, lr scheduler
3. `discount`
4. `batch_size`
5. Gradient clipping: `clip_grad_norm`
6. Architecture: `use_double_q`
7. `learning_starts`: generate enough samples before policy update
8. `target_update_period`: target network update frequency

## Epislon-greedy
```python
best_ac = torch.argmax(self.critic(observation))
if np.random.rand() >= epsilon:
    action = best_ac
else:
    action = np.random.randint(0, self.num_actions)
    action = torch.tensor(action)
```

## Target Network
```python
if step % self.target_update_period == 0:
    self.update_target_critic()

def update_target_critic(self):
    self.target_critic.load_state_dict(self.critic.state_dict())
```

## Critic Network
1. input `(s, a, s', r, done)`, online network `Q`, target network `Q'`
2. choose action: `a' = argmax_a_Q'(s', a)` where Q' is the target network
3. target value: `y = r(s, a) + (1 - done) * gamma * Q(s', a')`
4. estimated value: `Q(s, a)` where Q is the online Q network
5. `loss = MSE(Q(s, a), y)`
```python
with torch.no_grad():
    next_qa_values = self.target_critic(next_obs)   # (N, ac_dim)
    next_action = torch.argmax(next_qa_values, dim=-1)  # (N, )
    next_q_values = (1 - done.int()) * next_qa_values[torch.arange(batch_size), next_action] # (N, )
    target_values = reward + self.discount * next_q_values

qa_values = self.critic(obs)    # (N, ac_dim)
q_values = qa_values[torch.arange(batch_size), action]    # (N, )
loss = self.critic_loss(q_values, target_values)
```

## Double Q-learning
The only difference in implementation is choose a' = argmax_a_Q(s', a) where Q is the **online Q network**, instead of **target network**
```python
# choose action using critic network
# evaluate value using target network
if self.use_double_q:
    next_action = torch.argmax(self.critic(next_obs), dim=-1)
```