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

## Result
1. `uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500`
    reach a return of 500, `ndim=2`
2. `uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml`
    reach a return of 200
3. `uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml`
    took 6hr on MPS
    receive a score over 2000
4. Hyperparameters experiment on MsPacman-v0
    1. lr
        - start at 1e-4, deacy to 5e-5 at step 500k linearly
    2. network architecture
        - Conv2d + ReLU + Conv2d + ReLU + Conv2d + ReLU + Flatten + Linear + ReLU + Linear
    3. explorations schedule
        - epsilon start at 1, decay to 0.01 at step 500k linearly
    4. target network update frequency
        - 2000
# SAC
## DDPG
1. https://spinningup.openai.com/en/latest/algorithms/ddpg.html

## Algorithm
1. Input
    initial policy parameters $\theta$
    Q-function parameters $\phi_1, \phi_2$
    empty replay buffer $\mathcal{D}$
2. Set target network parameters equal to online network parameters $\phi_{targ,1} \leftarrow \phi_1, \phi_{targ,2} \leftarrow \phi_2$
3. repeat
    1. Generate samples (s,a,r,s',d), store them in replay buffer $\mathcal{D}$
    2. Randomly sample a batch $\mathcal{B}$ of transitions from $\mathcal{D}$
    3. Compute target for Q functions
        $$
        a' \sim \pi_{\theta}(\cdot | s') \\
        y(r,s',d) = r + \gamma(1 - d)(min_{i=1,2}Q_{\phi_{targ,i}}(s',a') - \alpha log\pi_{\theta}(a'|s'))
        $$
    4. update Q-functions by one step of gradient descent
        $$
        \nabla_{\phi_i}\frac{1}{|\mathcal{B}|}\sum_{\mathcal{B}}(Q_{\phi_i}(s,a) - y(r,s',d))^2
        $$
    5. update policy by one step of gradient ascent
        $$
        \nabla_{\theta}\frac{1}{\mathcal{B}}\sum_{s \in \mathcal{B}}(minQ_{\phi_i}(s,a_{\theta}(s)) - \alpha log\pi_{\theta}(a_{\theta}(s) | s))
        $$
    6. update target network with
        $$
        \phi_{targ,i} \leftarrow \rho\phi_{targ,i} + (1-\rho)\phi_i
        $$

## Implementation
1. Entropy bonus
2. Reparameterize trick
3. Temperature autotune

## Result
0. `https://wandb.ai/yangzf23-independent-developer/hw3/workspace?nw=nwuseryangzf23`
1. SAC implementation
    `uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml`
    reach a return of 1000
    `uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml`
    reach a return over 10k
2. Temperature autotune 
    `uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum_autotune.yaml`
    reach a return of 1000
    `uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml`
    the learning curve is similar to fix fine-tuned temperature version, reach a return over 10k
3. Clipped double-Q
    `uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_singleq.yaml`
    reach a return of 750
    `uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml`
    reach a retufn of 1500
    `q_values` and `target_values` curves show the clipping machanism works