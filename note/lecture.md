# Imitation Learning
1. MSE policy
2. Flow matching policy

# RL basis
## Markov decision process, MDP
1. Markov chain
    1. $\mathcal{M} = \{\mathcal{S}, \mathcal{T}\}$
    2. $\mu_{t+1} = \mathcal{T} \mu_t$, where $\mu_{t}$ is a probability vector
1. State, action, reward function and transition probabilities define MDP
    1. $\mathcal{M} = \{\mathcal{S}, \mathcal{T}, \mathcal{A}, \mathcal{r}\}$
    2. let $\mu_{t,j} = p(s_t=j)$, $\xi_{t,k} = p(a_t = k)$, $\mathcal{T}_{j,k} = p(s_{t+1}=j|s_t=j, a_t=k)$, then $$\mu_{t+1, i} = \sum_{j=1}^{|\mathcal{S}|} \sum_{k=1}^{|\mathcal{A}|} \mathcal{T}_{i, j,k} \mu_{t,j} \xi_{t,k}$$
    3. reward function $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$
## Objective of RL
1. Propability of trajectory $\tau = (s_1, a_1, \cdots)$ is $$p_{\theta}(\tau) = p(s_1) \prod_{t=1}^{\infty} \pi_{\theta}(a_t|s_t) p(s_{t+1}|s_t, a_t)$$
2. Return of trajectory $\tau$ is $$R(\tau) = \sum_{t=1}^{T} r(s_t, a_t)$$
3. When $T = \infty$, $p(s_t, a_t)$ converge to a stationary distribution, then $\mu = \mathcal{T}\mu$, $(\mathcal{T}-I)\mu = 0$, so $\mu$ is the eigenvector of $\mathcal{T}$ with eigenvalue 1
4. Objective of RL is to maximize the expected return 
    1. Finite horizon case
    $$\theta^* = \arg \max_{\theta} \sum_{t=1}^{T} \mathbb{E}_{(s_t,a_t) \sim p_{\theta}(s_t,a_t)} [r(s_t, a_t)]$$
    1. Infinite horizon case
    $$\theta^* = \arg \max_{\theta} \mathbb{E}_{(s, a)\sim p_{\theta}(s,a)} [r(s, a)]$$
5. The expectations of non-smooth non-differentiable reward functions under differentiable and smooth probability distributions are themselves **differentiable and smooth**. So we can use gradient-based optimization methods to optimize the policy parameters $\theta$
    - $r(x)$ not smooth
    - $E_{\pi_{\theta}}[r(x)]$ smooth and differentiable in $\theta$
## RL Algorithms
1. Anatomy of a RL algorithm
    ```
    loop
        1. Generate samples: run the policy
        2. Estimate the return: fit/estimate the Q-function or V-function
        3. Improve the policy: use the estimated Q-function or V-function to improve the policy
    end
    ```
2. Value functions
    1. Q-function: $Q^{\pi}(s_t, a_t) = \sum_{t'=t}^{T}\mathbb{E}_{\pi_{\theta}} [r(s_{t'}, a_{t'}) | s_t, a_t]$
    2. V-function: $V^{\pi}(s_t) = \sum_{t'=t}^{T}\mathbb{E}_{\pi_{\theta}} [r(s_{t'}, a_{t'}) | s_t]$
    3. V-function can be derived from Q-function: $$V^{\pi}(s_t) = E_{a_t \sim \pi_{\theta}(\cdot|s_t)}[Q^{\pi}(s_t, a_t)]$$
    4. Advantage function: $A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$

## Types of RL algorithms
1. **Policy gradients** methods: directly optimize the policy parameters $\theta$ by estimating the gradient of the expected return with respect to $\theta$
    - evaluate returns $J(\theta)$
    - $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
2. **Value-based** methods: estimate the value functions (Q-function or V-function)
    - fit $V(s)$ or $Q(s, a)$ using supervised learning
    - set $\pi(s) = \arg\max_{a} Q(s, a)$
3. **Actor-critic** methods: combine policy gradients and value-based methods, where the actor is the policy and the critic is the value function estimator
    - fit $V(s)$ or $Q(s, a)$ using supervised learning
    - $\theta \leftarrow \theta + \alpha \nabla_\theta E[Q(s, a)]$
4. **Model-based** methods: learn a model of the environment (transition probabilities and reward function) and use it to plan and optimize the policy
    - learn $p(s_{t+1}|s_t, a_t)$
    - use the learned model to plan(no policy): Monte Carlo tree search, optimal control
    - backpropagate gradients into the policy
    - use the learned model to learn a value function
## Trade-off of RL algorithms
1. **Sample efficiency**
    - How many samples are needed to learn a good policy
    - on-policy vs **off-policy** 
        - on-policy means even the policy changes a little bit, we need to generate new samples
        - off-policy means we can reuse old samples
2. Stability and convergence
    - Policy gradients methods are more stable but less sample efficient
    - Value-based methods are more sample efficient but less stable
    - Actor-critic methods are a compromise between the two
3. Assumptions
    1. Full observability
    2. Episodic learning
    3. Continuity or smoothness

# Policy gradients
## REINFORCE algorithm
1. Probability of trajectory $\tau = (s_1, a_1, \cdots, s_T, a_T)$ is $$p_{\theta}(\tau) = p(s_1) \prod_{t=1}^{T} \pi_{\theta}(a_t|s_t) p(s_{t+1}|s_t, a_t)$$
2. Objective of policy gradients is to maximize the expected return 
    $$\theta^* = \arg \max_{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)} [\sum_{t} r(s_t, a_t)]$$
3. Estimate the return of a trajectory $\tau$ by Monte Carlo sampling
    $$J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)} [R(\tau)] \simeq \frac{1}{N} \sum_{i=1}^{N} R(\tau_i)$$ 
4. Direct policy differentiation
    $$J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)} [R(\tau)] = \int p_{\theta}(\tau) R(\tau) d\tau$$
5. REINFORCE algorithm
    1. sample trajectories $\tau_i$ from policy $\pi_{\theta}$
    2. evaluate the policy gradient
    $$
    \begin{aligned}
    \nabla_\theta J(\theta) &= \int \nabla_\theta p_{\theta}(\tau) R(\tau) d\tau \\
    &= \int p_{\theta}(\tau) \nabla_\theta \log p_{\theta}(\tau) R(\tau) d\tau \\ 
    &= E_{\tau \sim p_{\theta}} [\nabla_\theta \log p_{\theta}(\tau) R(\tau)] \\
    &= E_{\tau \sim p_{\theta}} [(\sum_{t} \nabla_\theta \log \pi_{\theta}(a_t|s_t)) (\sum_{t} r(s_t, a_t))]\\
    &= \frac{1}{N} \sum_{i=1}^{N} (\sum_{t} \nabla_\theta \log \pi_{\theta}(a_{i,t}|s_{i,t})) (\sum_{t} r(s_{i,t}, a_{i,t}))\\
    \end{aligned}
    $$
    3. update the policy parameters $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
## Understand policy gradients
1. Increase the probability of actions that lead to high returns and decrease the probability of actions that lead to low returns
2. Policy gradient can be used in **partially observed MDPs** without modification
3. **High variance** of the policy gradient estimator makes it difficult to learn a good policy
## Variance reduction
1. **Causality**: policy at time t' cannot affect rewards at time t < t'. Replace reward with **reward-to-go**
    $$
    \begin{aligned}
    \nabla_\theta J(\theta) &= \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(a_{i,t}|s_{i,t}) (\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})) \\
    &= \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(a_{i,t}|s_{i,t}) \hat{Q}_{i,t}
    \end{aligned}
    $$
2. **Baselines**
    $$
    \begin{aligned}
    \nabla_\theta J(\theta) &= \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_{\theta}(\tau)[r(\tau) - b] \\
    b &= \frac{1}{N} \sum_{i=1}^{N} r(\tau_i)
    \end{aligned}
    $$
    substract a baseline is **unbiased in expectation**, but can reduce the variance of the policy gradient estimator
    $$
    \begin{aligned}
    E[\nabla_\theta \log p_{\theta}(\tau) b] &= \int p_{\theta}(\tau) \nabla_\theta \log p_{\theta}(\tau) b d\tau \\
    &= \int \nabla_\theta p_{\theta}(\tau)b d\tau \\
    &= b \nabla_\theta \int p_{\theta}(\tau) d\tau \\
    &= b \nabla_\theta 1 \\
    &= 0
    \end{aligned}
    $$
## Off-policy policy gradients
1. Policy gradient is on-policy
    - Neural networks change only a little bit with each gradient step
    - But we need to **generate new samples** after each gradient step
2. Importance sampling
    $$
    \begin{aligned}
    E_{x \sim p(x)} [f(x)] &= \int p(x) f(x) dx \\
    &= \int q(x) \frac{p(x)}{q(x)} f(x) dx \\
    &= E_{x \sim q(x)} [\frac{p(x)}{q(x)} f(x)] \\
    J(\theta) &= E_{\tau \sim p_{\theta}(\tau)} [r(\tau)] \\
    &= E_{\tau \sim p_{\theta_{old}}(\tau)} [\frac{p_{\theta}(\tau)}{p_{\theta_{old}}(\tau)} r(\tau)] \\
    \frac{p_{\theta}(\tau)}{p_{\theta_{old}}(\tau)} &= \prod_{t=1}^{T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
    \end{aligned}
    $$
3. Off-policy policy gradient estimator
    1. given $$J(\theta)=E_{\tau \sim p_{\theta}(\tau)} [r(\tau)]$$
    2. estimate $$J(\theta') = E_{\tau \sim p_{\theta(\tau)}} [\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)} r(\tau)]$$
    3. estimate the policy gradient 
    $$
    \begin{aligned}
    \nabla_\theta' J(\theta') &= E_{\tau \sim p_{\theta}(\tau)} [\frac{\nabla_{\theta'}p_{\theta'}(\tau)}{p_{\theta}(\tau)} r(\tau)] \\
    &= E_{\tau \sim p_{\theta}(\tau)} [\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)} \nabla_{\theta'} \log p_{\theta'}(\tau) r(\tau)] \\
    &= E_{\tau \sim p_{\theta}(\tau)} [(\prod_{t=1}^{T} \frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}) (\sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)) (\sum_{t=1}^{T} r(s_{t}, a_{t}))] \\
    \end{aligned}
    $$
## Implement policy gradients
1. Policy gradient with automatic differentiation
    - Build a graph s.t. its gradient is the policy gradient
    - Implement "pseudo-loss"
    $$L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log \pi_{\theta}(a_{i,t}|s_{i,t}) \hat{Q}_{i,t}$$
2. Using much **larger batch sizes** to reduce the variance of the policy gradient estimator
3. Tweaking the learning rate is very hard. Using `Adam` is a good starting point

# Off-policy Policy Gradient


# Value-based RL
1. Actor-critic architecture
2. DQN
    replay buffer
    target network
    double Q learning
3. SAC
    continuous action space
    policy network
        output $\pi(a|s) \sim \mathcal{N}(\mu, \sigma)$
        reparameterization
    critic network
        clipped double Q
        target network
        soft target network update
        entropy bonus
    automatic temeperature tuning

# Variational inference

# LLM RL

# Model-based RL
## Overview
1. Learn the transistion dynamics
2. Choose actions using the system dynamics
3. Optimal control, trajectory optimization, planning
4. Learned policies + learned dynamics

## Open-loop planning
1. Stochastic optimization
    $$
    a_1, ..., a_T = \arg \max_{a_1, ..., a_T} J(a_1, ..., a_T) \\
    A = arg \max_{A} J(A)
    $$
    1. Guess and check
        1. pick $A_1, ..., A_N$ from some distribution
        2. choose $A_i$ based on $arg\max_{A_i} J(A_i)$
    2. Cross-entropy method, CEM
        1. pick $A_1, ..., A_N$ from some distribution
        2. choose $A_i$ based on $arg\max_{A_i} J(A_i)$
        3. **refit the distribution** to the best $A_i$
        4. repeat until convergence
2. Monte Carlo tree search (MCTS)
    1. MCTS sketch
    ```
    Loop
        1. find a leaf sl using TreePolicy(s1)
        2. evaluate the leaf using DefaultPolicy(sl)
        3. update all values in tree between s1 and sl
    End
    take best action from s1
    ```
    2. UCT TreePolicy
        1. if st not fully expanded, choose new $a_t$
        2. else choose child with best score
        $$
        Score(s_t) = \frac{Q(s_t)}{N(s_t)} + 2C \sqrt{\frac{\log N(s_{t-1})}{N(s_t)}}
        $$
## Trajectory optimization
1. Object
$$
\min_{u_1,...u_t} c(x_1, u_1) + c(f(x_1, u_1), u_2) + ... + c(f(f(...)...), u_T) \\
$$
where $x_{t+1} = f(x_t, u_t)$
2. Linear dynamics
$$
f(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{F}_t 
\begin{bmatrix}
\mathbf{x}_t \\
\mathbf{u}_t
\end{bmatrix} + \mathbf{f}_t
$$
3. Quadratic cost
$$
c(\mathbf{x}_t, \mathbf{u}_t) = \frac{1}{2} \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{C}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{c}_t
$$
4. LQR
    1. $V(x_t)$ is cost-to-go from state $x_t$ until end 
    2. $Q(x_t, u_t)$ is the cost-to-go from state $x_t$ and take action $u_t$
    3. $$V(x_t) = \min_{u_t} Q(x_t, u_t)$$
    4. $Q(x_t, u_t)$ and $V(x_t)$ are all **quadratic functions** of $x_t$ and $u_t$, so we can solve for the optimal action $u_t$ in closed form
5. Stochastic dynamics
$$
f(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{F}_t 
\begin{bmatrix}
\mathbf{x}_t \\
\mathbf{u}_t
\end{bmatrix} + \mathbf{f}_t \\
p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t) = \mathcal{N}(\mathbf{F}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \mathbf{f}_t, \boldsymbol{\Sigma}_t)
$$
LQR still works
6. **Nonlinear dynamics**: DDP/iterative LQR (iLQR)
    1. Guess a trajectory $\tau = (x_1, u_1, ..., x_T, u_T)$
    2. Linearize the dynamics around the trajectory
    $$
    \mathbf{F}_t = \nabla_{x_t, u_t} f(\hat{x}_t, \hat{u}_t) \\
    $$
    3. Quadraticize the cost around the trajectory
    $$
    c_t = \nabla_{x_t, u_t} c(\hat{x}_t, \hat{u}_t) \\
    C_t = \nabla^2_{x_t, u_t} c(\hat{x}_t, \hat{u}_t) \\
    $$
    4. Run LQR backward pass on 
        state $\delta x_t=x_t - \hat{x}_t$
        action $\delta u_t = u_t - \hat{u}_t$
    5. Run forward pass with real nonlinear dynamics 
    $$
    u_t = K_t(x_t-\hat{x}_t) + \alpha k_t + \hat{u}_t \\
    x_{t+1} = f(x_t, u_t)
    $$
    $\alpha$ is the line search parameter
    6. Update $\hat{x}_t$ and $\hat{u}_t$ based on states and actions in forward pass
    7. Repeat until convergence

## Learned model
1. Distribution shift
    1. learn probability model
    2. trust region policy update
2. Uncertainty-aware nerual networks 
    1. Environment uncertainty (aleatoric)
    2. Model uncertainty (epistemic)
3. Bayesian neural networks: high-dimensional, hard to scale up 
4. Boostrap ensembles
    1. Resample with replacment from $\mathcal{D}$
    2. Train $N$ neural networks, $\theta_i$ is trained on $\mathcal{D}_i$
    $$
    p(\theta \mid D) \approx \frac{1}{N}\sum_i \delta(\theta_i) \\
    \int p(s_{t+1}\mid s_t, a_t, \theta)\, p(\theta\mid D)\, d\theta
    \approx
    \frac{1}{N}\sum_i p(s_{t+1}\mid s_t, a_t, \theta_i)
    $$
## Different ways to use model
### Planning with models
1. Stochastic optimization
    1. pick $A_1$, ..., $A_N$ from some districutions
    2. choose $\argmax_i J(A_i)$
2. Cross-entropy method
    1. pick $A_1$, ..., $A_N$ from some districutions
    2. evaluate $J(A_i)$
    3. pick the top M samples $A_1, ..., A_M$
    4. refit $p(A)$ to $A_1, ..., A_M$
3. Monte Carlo tree search
4. Continuous trajectory optimization, LQR
5. Plan with uncertainty
    1. plan under an esemble of models
    2. posterior estimation with BNNs
### Policy learning with models
1. Tradeoff
    1. Long rollouts see further into the future but accumulate **large error**
    2. Short roolouts are more accurate but **not expose later consequences**
2. Model based RL with **short rollouts**
    1. Run base policy $\pi_0(a_t \mid s_t)$, collect $\mathcal{D} = {(s,a,s')_i}$
    2. Learn dynamic model $f(s, a)$ to minimize MSE $\sum_i ||f(s_i, a_i) - s_i'||^2$
    3. Repeat K times
      1. Pick states $s_i$ from $\mathcal{D}$, use $f(s, a)$ to make short rollouts
      2. Use both real and model data to imporve $\pi_{\theta}(a|s)$ with off-policy RL
    4. Run $\pi_{\theta}(a|s)$ and append visited data to $\mathcal{D}$
3. Dyna style model-based RL
    1. Run base policy $\pi_0(a_t \mid s_t)$, collect $\mathcal{B} = {(s,a,s')_i}$
    2. Learn model $\hat{p}(s' | s, a)$ and $\hat{r}(s, a)$
    3. Repeat k times
      1. Sample $s\sim\mathcal{B}$ from buffer
      2. Choose action a from $\mathcal{B}$ or policy or random
      3. Simulate $s'\sim\hat{p}$ and $r = \hat{r}$
      4. Train on (s, a, s', r) with model-free RL
      5. Take N more mode-based steps
4. Model-accelerated off-policy RL, MBA
    1. Data collections: stored in **real transition buffer**
    2. Learn transition model: train $\hat{p}$ (and reward model $\hat{r}$) on real transition data
    3. Generate model-based transitions: pick state from real transition buffer, use learned model to generate imaginary transitions, stored in **model-based transition buffer**
    4. Learn Q-function: train on real + model-based transistion data
    5. Update policy: with learned Q-function
    6. Data evict
      1. model-based transition buffer: evicted once the model changes (every iteration) 
      2. real transition buffer: evicted old data (sliding window)
5. MBA, MVE and MBPO
    1. MBA: use **model-based transitions data** to accelearte model-free learning
    2. MVE: use model to **improve target value** in a Bellman update 
      $$y = r_0 + γ r_1 + γ² r_2 + ... + γ^H V(s_H)$$
      where $r_i$ is given by the learned model, $V(s_H)$ is given by the learned V function
    3. MBPO: a specific implementation of MBA
      1. Learn an ensemble dynamics model
      2. Start short rollouts from real replay-buffer states
      3. Choose actions from current policy
      4. Train SAC on real + model data
### Latent state-space model
1. Object
    $$
    \max_{\phi,\psi} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{H}
    \mathbb{E}_{q}
    \left[
    \log p_{\phi}(s_{t+1,i} \mid s_{t,i}, a_{t,i})
    +
    \log p_{\phi}(o_{t,i} \mid s_{t,i})
    \right]
    +
    \mathcal{H}\left(
    q_{\psi}(s_t, s_{t+1} \mid o_{1:H,i}, a_{1:H,i})
    \right)
    $$
2. Latent space model
    1. Encoder
        1. full smoothing posterior: $q_{\psi}(s_t, s_{t+1}\mid o_{1:H}, a_{1:H})$
        2. single step encoder: $q_{\psi}(s_t\mid o_t)$
    3. Decoder: $p(o_t\mid s_t)$
    4. Dynamics model: $p(s_{t+1}|s_t, a_t)$
    5. Reward model: $p(r_{t}|s_t, a_t)$
3. Model-based RL with latent space models 
    1. run base policy to collect $\mathcal{D}$
    2. learn dynamics model, reward model, encoder and decoder
    3. plan through model to choose actions
    4. execute the first planned action, observe result o'(MPC)
    5. append $(o, a, o')$ to dataset $\mathcal{D}$
4. Actor-critic with learned representations
    1. Learn latent model from real data
    2. Train actor-critic on inferred latent states (still use real data)
5. Actor-critic + model-based RL
    1. Learn latent model from real data
    2. Use learned model to **generate transition data**
      real observation data $o_i$
      -> encoder gives latent state $s_i$
      -> choose action $a_i$ from $\pi(a | s)$ or $\pi(a | o)$
			-> reward model predict reward $r_i$
      -> dynamic model predict next state $s_i'$
      -> add $(s_i, a_i, s_i')$ model-based transition to buffer
    3. Train actor-critic on real data + model-based data


# Offline RL
