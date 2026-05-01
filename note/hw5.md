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
## FQL
