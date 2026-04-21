# Off-policy Policy Gradient
1. An approximation of the KL divergence with a **2nd order Taylor expansion**
$$
\begin{aligned}
D_{KL}(p_{\theta}||p_{\theta+\delta}) &= E_{x\sim p_{\theta}} [log p_{\theta}(x) − log p_{\theta+\delta}(x)]  \\
&= -\frac{1}{2} \delta^{T} F(\theta)\delta
\end{aligned}
$$
where $F (\theta) = E_{x∼p_θ}[\nabla_θ log p_θ(x)\nabla_θ log p_θ(x)^T]$ is called the **fisher information matrix** 
2. Sample inefficiency of REINFORCE algorithm
    1. Policy gradient estimate itself is high variance, needs lots of samples to get a low-variance estimate
    2. REINFORCE is on-policy
3. Off-policy policy gradient by Importance Sampling
    $$
    \begin{aligned}
    \nabla_\theta J(\theta) 
    &=
    \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ r(\tau) \right] \\
    &=
    \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[
    \sum_{t=0}^{H-1}
    \nabla_\theta \log \pi_\theta(a_t \mid s_t)
    \left(
    \prod_{t'=0}^{t} \frac{\pi_\theta(a_{t'} \mid s_{t'})}{\pi_\beta(a_{t'} \mid s_{t'})}
    \right)
    \left(
    \sum_{t'=t}^{H-1} r(s_{t'}, a_{t'})
    \left(
    \prod_{t''=t+1}^{t'} \frac{\pi_\theta(a_{t''} \mid s_{t''})}{\pi_\beta(a_{t''} \mid
    s_{t''})}
    \right)
    \right)
    \right]
    \end{aligned}
    $$
4. Off-policy policy iteration
    $$
    \begin{aligned}
    J(\theta') - J(\theta) &= \mathbb{E}_{\tau\sim p_{\theta'}(\tau)} \left[\sum_{t=0}^{\infty}\gamma^t A^{\pi_\theta}(s_t, a_t)\right] \\
    &= \sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t)} \left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] \right] \\
    &\geq \sum_t \mathbb{E}_{s_t \sim p_\theta(s_t)} \left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] \right] - \sum_t 2\epsilon t C
    \end{aligned}
    $$
    1. $J(\theta') - J(\theta)$ is the advantage of the old policy in expectation on trajectories sampled from the new
    2. Object
    $$
    \theta' \leftarrow \argmax_{\theta'}\sum_t \mathbb{E}_{s_t \sim p_\theta(s_t)} \left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] \right]
    $$
    3. Subject to constraint $D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t | s_t)) \leq \epsilon$
    4. Convert to a unconstrained optimization
        $$\mathcal{L}(\theta',\lambda)=\sum_t \mathbb{E}_{s_t \sim p_\theta(s_t)} \left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] \right] - \lambda(D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t | s_t)) - \epsilon)$$
    5. Dual gradient descent
        $$
        \begin{aligned}
        \theta' &\leftarrow \argmax_{\theta'} \mathcal{L}(\theta', \lambda) \\
        \lambda &\leftarrow \lambda + \alpha(D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t | s_t)) - \epsilon)
        \end{aligned}
        $$
    
# TRPO
1. Algorithm
    $$
    \begin{aligned}
    \theta_{k+1} = \arg \max_{\theta} \; & {\mathcal L}(\theta_k, \theta) \\
    \text{s.t.} \; & \bar{D}_{KL}(\theta || \theta_k) \leq \delta
    \end{aligned}
    $$
2. Surrogate advantage
    $$
    {\mathcal L}(\theta_k, \theta) = \mathbb{E}_{s,a \sim \pi_{\theta_k}}\left[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)\right]
    $$
3. KL-divergence
    $$
    \bar{D}_{KL}(\theta || \theta_k) = \mathbb{E}_{s \sim \pi_{\theta_k}}\left[D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)\right]
    $$
4. Taylor exapnd the objective and constraint
    $$
    \begin{aligned}
    {\mathcal L}(\theta_k, \theta) &\approx g^T (\theta - \theta_k) \\
    \bar{D}_{KL}(\theta || \theta_k) & \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)
    \end{aligned}
    $$
    and get an approximate optimization problem
    $$
    \begin{aligned}
    \theta_{k+1} = \arg \max_{\theta} \; & g^T (\theta - \theta_k) \\
    \text{s.t.} \; & \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta.
    \end{aligned}
    $$
5. Update with a backtracking line search
    $$
    \theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g
    $$
    1. where $\alpha\in (0, 1)$ is the backtracking coefficient and $j$ is the smallest nonnegative integer such that $\pi_{\theta_{k+1}}$ 
        1. satisifies the **KL constraint**
        2. produces a **postive surrogate advantage**
    2. instead of calculating $H^{-1}$, using conjugate gradient algorithm to solve $Hx = g$ and get $x = H^{-1}g$
# PPO
## PPO-KL
1. Penalizes the KL-divergence in the objective function instead of making it a hard constraint
2. Automatically adjusts the penalty coefficient
## PPO-Clip
1. Update policy via
$$
\begin{aligned}
\theta_{k+1} &= \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[L(s,a,\theta_k, \theta)\right] \\
L(s,a,\theta_k,\theta) &= \min\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;g(\epsilon, A^{\pi_{\theta_k}}(s,a))\right) \\
g(\epsilon, A) &= \left\{
    \begin{array}{ll}
    (1 + \epsilon) A & A \geq 0 \\
    (1 - \epsilon) A & A < 0.
    \end{array}
    \right.
\end{aligned}
$$
2. $\epsilon$ is a hyperparameter which roughly says how far away the new policy is allowed to go from the old
3. It is still possible to end up with a new policy which is too far from the old policy
    **Early stopping**: if the mean KL-divergence of the new policy from the old grows beyond a threshold, stop taking gradient steps

# GRPO
1. Group Relative Advantage: for one prompt $x_i$, let the rewards of its $G$ sampled completions be $r_{i,1}, ... , r_{i,G}$. GRPO-style group normalization defines
    $$
    A_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i + \varepsilon}, \quad \mu_i = \frac{1}{G} \sum_{j=1}^G r_{i,j}, \quad \sigma_i = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_{i,j} - \mu_i)^2}.
    $$
2. For token $t$ of completion $i$, define the importance ratio
    $$
    \rho_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} \mid x_i, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x_i, y_{i,<t})}
    $$
3. Loss function
    $$
    L_{\text{pg}}^{\text{grpo}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{t=1}^{T_i} \min\left(\rho_{i,t}(\theta)A_i, \ \text{clip}(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon)A_i\right)
    $$
4. full GRPO loss
    $$
    L^{\text{grpo}}(\theta) = L_{\text{pg}}^{\text{grpo}}(\theta) + \beta \widehat{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}).
    $$
# Impl
## LLM as policy
1. `compute_per_token_logprobs`
    1. calculate $log\pi_{\theta}(y_{i,t}|x_i,y_{i,<t})$
    2. `out = model(input_ids, attention_mask)` where `model` is `Qwen/Qwen2.5-Math-1.5B-Instruct`, `out` returns logits of shape `(B, L, V)`
    3. trick: call `-F.cross_entropy` to calculate log softmax efficiently
2. `build_completion_mask`
    1. input_id is organized as `[prompt, completion, padding]`
    2. all inputs share the same prompt with `prompt_input_len` 
    3. `attention_mask` exclude the paddings
    4. return a `completion_mask` of size `(B, L-1)` to indicate the completion
3. `approx_kl_from_logprobs`
    1. exact KL divergence is
    $$
    \mathrm{KL}\left(\pi_\theta(\cdot \mid s)||\pi_{\mathrm{ref}}(\cdot \mid s)\right)
    = \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}
    \left[\log \pi_\theta(a \mid s)-\log \pi_{\mathrm{ref}}(a \mid s)\right]
    $$
    2. use sampled-token estimator
    $$
    \hat{k}(a) = e^{\Delta(a)}-\Delta(a)-1 \\
    \Delta(a) = \log \pi_{\mathrm{ref}}(a \mid s) - \log \pi_\theta(a \mid s)
    $$
    3. return masked mean of `(B, L-1)` shape
## Minibatch
1. `iter_minibatches`
    1. Split `RolloutBatch` into minibatches of `minibatch_size`
    2. GR-REINFORCE is strictly on-policy. Setting `batch_size=8`, `group_size=8` `minibatch_size=8` and `grad_accum_steps=8`, then **number of minibatches equals to grad_accum_step**, and gradients are accumulated across 8 minibatches before taking one optimizer step
    3. GRPO reuses old policy samples for `ppo_epochs > 1`
## Group relative advantages
1. `compute_group_advantages`
    1. convert rewards from shape `(batch_size * group_size, )` to `(batch_size, group_size)`
    2. calculate group relative advantages by
    $$
    A_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i + \varepsilon}, \quad \mu_i = \frac{1}{G} \sum_{j=1}^G r_{i,j}, \quad \sigma_i = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_{i,j} - \mu_i)^2}.
    $$
    3. convert rewards back to shape `(batch_size * group_size, )`
2. `maybe_normalize_advantages`: normalize advantages across all samples (the same normalize advantage trick as hw1-3)

## Policy update
1. `Reinforce.update`
    1. use `completion_mask` for calculating to exclude prompt and padding tokens
    $$
    \bar{\ell}_i(\theta)=\frac{1}{T_i}\sum_{t=1}^{T_i}\log \pi_\theta\left(y_{i,t}\mid x_i,y_{i,<t}\right)
    $$
    2. use `completion_mask` for calculating KL
2. `GRPO.update`
    1. calculate `log_ratio` with `new_logprobs` and `old_logprobs`, get ratio by `log_ratio.exp()`
    $$
    \rho_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} \mid x_i, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x_i, y_{i,<t})}
    $$
    2. use `torch.clip` and `torch.min` to get clipped token objective
    $$
    \min\left(\rho_{i,t}(\theta)A_i, \ \text{clip}(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon)A_i\right)
    $$
    3. use `completion_mask` to calculate seq objective 
    $$
    \frac{1}{T_i} \sum_{t=1}^{T_i} \min\left(\rho_{i,t}(\theta)A_i, \ \text{clip}(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon)A_i\right)
    $$

# Result
1. REINFORCE on format_copy
    1. run on local CPU take 2.5hr
    2. reward starts near 0, jumps sharply by about step 10, and reaches about 1.3 by the end 
    3. eval exact match usually approaches 1.0
2. GRPO on format_copy
    1. run on local CPU take 3hr
    2. reward starts near 0, jumps sharply by about step 10, and reaches about 1.3 by the end 
    3. eval exact match usually approaches 1.0
3. REINFORCE on hard math
    1. run on Modal H100 take 2hr
    2. final eval reach 0.28+
4. GRPO on hard math
    1. run on Modal H100 take 5.5hr
    2. final eval reach 0.39+
5. Gr-reinforce vs GRPO on math
    1. In the first 200 iterations, the `eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser` of GRPO climbs up to 0.34+ from 0.22+, whose growth is approximately double as Gr-reinforce. Since GRPO set `ppo_epochs=2` and share the same `batch_size, group_size, minibatch_size, grad_accum_steps` with Gr-refinforce, the optimizer steps also double, and they share the same lr schedule, the eval result is expected