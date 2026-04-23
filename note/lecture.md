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

# Offline RL

