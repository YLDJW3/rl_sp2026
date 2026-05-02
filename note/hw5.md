# Theory
## SAC+BC
1. Train two Q functions $Q_1, Q_2$ and a Gaussian policy network $\pi$, and a learnable entropy coefficient $\beta\gt0$
2. Q functions are trained by minimizing the Bellman error:
$$
y = r + \frac{\gamma}{2}\sum_{j=1}^2\bar{Q}_j(s',a') \\
\mathcal{L}(Q) = \sum_{i=1}^2\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}, a'\sim\pi(\cdot|s')} \left[ (Q_i(s,a) - y)^2 \right]
$$
3. Policy is trained to minimize the following loss
    $$
    \mathcal{L}(\pi) = \mathbb{E}_{(s,a)\sim\mathcal{D}, a^{\pi}\sim\pi(\cdot|s)} \left[ -\frac{1}{2}\sum_{i=1}^2Q_i(s,a^{\pi}) + \beta \log \pi(a^{\pi}|s) + \alpha\frac{1}{||\mathcal{A}||}||a-a^{\pi}||^2 \right]
    $$
    1. where $\alpha$ is a hyperparameter that controls the strength of the behavioral cloning term
    2. The first term maximize the Q values
    3. The second term maximize the entropy of the policy
    4. The third term regularizes the policy to stay close to the behavior policy
4. The entropy coefficient $\beta$ is updated by minimizing the following loss:
    $$\mathcal{L}(\beta) = \mathbb{E}_{s\sim\mathcal{D},a^{\pi}\sim\pi(\cdot|s)} \left[ \beta\cdot(-\log \pi(a^{\pi}|s) - \bar{\mathcal{H}}) \right]$$
    In practice target entropy is set to $\bar{\mathcal{H}} = -\dim(\mathcal{A})/2$
## IQL
1. Train two Q functions $Q_1, Q_2$ and a value function $V$, and a Gaussian policy network $\pi$
2. The Q and V functions are trained by minimizing the following losses:
$$
\mathcal{L}(Q) = \sum_{i=1}^2\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}} \left[ (Q_i(s,a) - r - \gamma V(s'))^2 \right] \\
\mathcal{L}(V) = \mathbb{E}_{s,a\sim\mathcal{D}} \left[ \mathcal{l}_2^{\tau}\left(V(s) - \min_{i=1,2} \bar{Q}_i(s,a)\right) \right]
$$
3. The IOL policy is trained to minimize the following loss:
$$
\mathcal{L}(\pi) = \mathbb{E}_{s,a\sim\mathcal{D}} \left[-\min(e^{\alpha A(s,a)},M)log\pi(a\mid s)\right] \\
A(s,a) = \min_{i=1,2} Q_i(s,a) - V(s)
$$
    - $\alpha\gt0$ is an inverse temperature parameter that controls the strength of advantage weighting
    - $M\gt0$ is a constant that clips the weights to improve stability
4. Hyperparameter $\alpha$ and $\tau$
    1. expectile $\tau$ interpolates between SARSA operator ($\tau=0.5$) and Bellman optimality operator ($\tau\to1$)
    2. $\alpha$ interpolates between behavior cloning ($\alpha=0$) and greedy maximiazation of Q function ($\alpha\to\infty$)
## FQL
