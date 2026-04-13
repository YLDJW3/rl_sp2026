# Key concepts
## States and Observations
1. A **state** $s$ is a complete description of the state of the world
2. An **observation** $o$ is a partial description of a state, which may omit information
    Fully observed: agent is able to observe the **complete** state of the environment
    Partially observed: agent can only see a **partial** observation
## Action space
1. The set of all valid actions in a given environment is often called the action space
2. Discrete action space: like Go
3. Continuous action space: robot control in real world 
## Policy
1. Policy: a rule used by an agent to decide what actions to take
2. Deterministic policy: $a_t = \mu(s_t)$
    ```python
    # a multi-layer perceptron (MLP) network with two hidden layers of size 64 and tanh activation functions
    pi_net = nn.Sequential(
              nn.Linear(obs_dim, 64),
              nn.Tanh(),
              nn.Linear(64, 64),
              nn.Tanh(),
              nn.Linear(64, act_dim)
            )
    ```
3. Stochastic policy: $a_t \sim \pi(\cdot|s_t)$
    1. **Sampling** actions from the policy
    2. Computing **log likelihoods** of particular actions, $log \pi_{\theta}(a|s)$
    3. Discrete: categorical policies
        some layers of nn: logits vector
        softmax: probs vector
    4. Continuous: diagonal Gaussian policies
        mean vector $\mu$
        diagonal covariance matrix $\Sigma$
## Trajectory
1. A trajectory $\tau$ is a sequence of states and actions in the world
    $\tau = (s_0, a_0, s_1, a_1, ...)$
2. State transitions are governed by the natural laws of the environment
    $s_{t+1}=f(s_t, a_t)$ or $s_{t+1} \sim P(\cdot|s_t,a_t)$
## Reward and return
1. Reward function R is critically important in reinforcement learning
    $r_t = R(s_t, a_t, s_{t+1})$
2. **Finite-horizon undiscounted return** is the sum of rewards obtained in a fixed window of steps
    $R(\tau) = \Sigma_{t=0}^{T} r_t$
3. **Infinite-horizon discounted return**
    $R(\tau) = \Sigma_{t=0}^{\infty}\gamma^{t}r_t$

## RL problem
1. The goal in RL is to **select a policy** which **maximizes expected return** when the agent acts according to it
2. The probability of a T-step trajectory is
    $P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t)$
3. The expected return denoted by $J(\pi)$ is then
    $J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = E_{\tau\sim \pi}[R(\tau)]$
4. The central problem of RL can be expressed by 
    $\pi^* = \arg \max_{\pi} J(\pi)$
    where $\pi^*$ be the optimal policy
## Value function
1. By value, we mean the expected return if you start in that state or state-action pair, and then act according to a particular policy forever after
2. **On-Policy Value Function** 
    start in state s and always act according to policy $\pi$
    $V^{\pi}(s) = E_{\tau \sim \pi}[R(\tau)|s_0 = s]$
3. **On-Policy Action-Value Function**
    start in state s and take an arbitary action $a$ (might not be given by policy $\pi$) and then act according to $\pi$ forever
    $Q^{\pi}(s,a) = E_{\tau \sim \pi}[R(\tau)|s_0 = s, a_0 = a]$
4. **Optimal Value Function**
    start in state s and always act according to the optimal policy in the environment:
    $V^*(s) = \max_{\pi} E_{\tau \sim \pi}[R(\tau) | s_0 = s]$
5. **Optimal Action-Value Function**
    start in state s and take an arbitary action $a$ and then act according to optimal policy in the environment
    $Q^*(s,a) = \max_{\pi}E_{\tau \sim \pi}[R(\tau)|s_0 = s, a_0 = a]$
## Bellman equations: reward + next-value
1. The value of your starting point is the **reward** you expect to get from being there, plus the **value** of wherever you land next
2. 4 Bellman equations
    $V^{\pi}(s) = E_{a\sim \pi, s'\sim P}[r(s, a) + \gamma V^{\pi}(s')]$
    $Q^{\pi}(s,a) = r(s, a) + \gamma E_{s' \sim P, a' \sim \pi}[Q^{\pi}(s', a')]$
    $V^*(s) = \max_{a}[r(s, a) + \gamma E_{s' \sim P}[V^*(s')]]$
    $Q^*(s,a) = r(s, a) + E_{s' \sim P}[\gamma \max_{a'}E_{a' \sim \pi}[Q^*(s', a')]]$
## Advantage function
1. Advantage function tells how much better an action is than others on average
2. $A^{\pi}(s, a) = Q^{\pi}(s, A) - V^{\pi}(s)$

## Markov Decision Processes
1. An MDP is a 5-tuple, $\langle S, A, R, P, \rho_0 \rangle$ where
    S is the set of all valid states
    A is the set of all valid actions
    R : $S \times A \times S \to \mathbb{R}$ is the reward function, with $r_t = R(s_t, a_t, s_{t+1})$
    P : $S \times A \to \mathcal{P}(S)$ is the transition probability function, with $P(s'|s,a)$ being the probability of transitioning into state s' if you start in state s and take action a
    $\rho_0$ is the starting state distribution

# RL Algorithm
1. Model based vs Model free
    whether the agent has access to (or learns) a model of the environment
2. What to learn 
    Policy, Q-function, V-function, environemnt model
3. Model-free RL
    **Policy optimization**: optimize $\theta$ in $\pi_{\theta}(a|s)$ by gradient descent on performance object $J(\pi_{\theta})$, such as `A2C, PPO`
    **Q-learning**: learn an approximator $Q_{\theta}(s,a)$ for the optimal action-value function $Q^*(s,a)$, such as `DQN`. The actions taken by the Q-learning agent are given by
        $a(s) = argmax_{a}Q_{\theta}(s, a)$
    **Interpolating** between policy optimization and Q-learning: there exist a range of algorithms that live in between the two extremes, such as `DDPG, SAC`
4. Model-based RL
    1. Pure planning: not explicitly represents the policy, use pure planning techniques like **model-predictive control, MPC** to choose actions
    2. Expert iteration: learn an explicit representation of the policy $\pi_{\theta}(a|s)$
        Uses a **planning algorithm** like Monte Carlo tree search to produce an action which is better than the policy would have produced
        The policy is afterwards updated to produce an action more like the planning algorithm’s output
    3. Data augmentation for model-free methods: MBVE, world model
        1. Use a **model-free RL algorithm** to train a policy or Q-function
        2. Augment real experiences with fictitious ones, or use only **fictitous experience** for updating the agent
    4. Embedding Planning Loops into Policies: I2A
        1. Embeds the planning procedure directly into a policy as a subroutine
        2. The policy can learn to choose how and when to use the plans

# Policy optimization
## Deriving the simplest policy gradient
1. Object: maximize the expected return $J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}[R(\tau)]$
2. Optimize the policy by gradient descent
    $\theta_{k+1} = \theta_k + \alpha \nabla_{\theta}J(\pi_{\theta})|_{\theta_k}$
3. Deduct
    1. Probability of a trajectory $\tau=(s_0,a_0,s_1,...,a_T,s_{T+1})$
        $P(\tau|\theta)=\rho_o(s_0)\prod_{t=0}^T P(s_{t+1}|s_t,a_t)\pi(a_t|s_t)$
    2. Log derivative
        $\nabla_{\theta}P(\tau|\theta)=P(\tau|\theta)\nabla_{\theta}logP(\tau|\theta)$
    3. Log probability of a trajectory
        $logP(\tau|\theta)=log\rho_o(s_0) + \Sigma_{t=0}^T (logP(s_{t+1}|s_t,a_t) + log\pi(a_t|s_t))$
    4. Gradient of log probability of a trajectory
        $\nabla_{\theta}logP(\tau|\theta)= \Sigma_{t=0}^T \nabla_{\theta}log\pi(a_t|s_t)$
    5. $\nabla_{\theta}J(\pi_{\theta})=\nabla_{\theta}E_{\tau\sim\pi}[R(\tau)]=\int_{\tau}\nabla_{\theta}P(\tau|\theta)R(\tau)=\int_{\tau}P(\tau|\theta)\nabla_{\theta}logP(\tau|\theta)R(\tau)=E_{\tau\sim\pi_{\theta}}[\nabla_{\theta}logP(\tau|\theta)R(\tau)]=E_{\tau\sim\pi_{\theta}}[\Sigma_{t=0}^T\nabla_{\theta}log\pi_{\theta}(a_t|s_t)R(\tau)]$
4. If we collect a set of trajectories $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$ where each trajectory is obtained by letting the agent act in the environment using the policy $\pi_{\theta}$, the policy gradient can be estimated with
    $\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)$


# Key papers in Deep RL
1. https://spinningup.openai.com/en/latest/spinningup/keypapers.html

# Vanilla policy gradient, VPG
1. Idea: push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return
2. Take the gradient with $\theta$ of $(\pi_{\theta})$, update the policy parameter with gradient descent method
    $$\theta_{t+1} = \theta_{t} + \alpha\nabla_{\theta}J(\pi_{\theta_{t}})$$

# Trust Region Policy Optimization, TRPO
