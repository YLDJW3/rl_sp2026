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