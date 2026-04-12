# Training loop
1. Actor: Policy gradient
    - `forward` returns `Categorical` distribution in discrete action space, and returns `Normal` distribution in continuous action space
    - `get_action` should **sample from the distribution** returned by `forward`, NOT modify it with `argmax` since it is used in training
    - "loss" compute
        ```python
        dist : D.Distribution = self.forward(obs) 
        log_prob = dist.log_prob(actions)
        loss = (-log_prob * advantages).mean()
        ```
2. PG agent
    - implement `_calculate_q_vals` by reward or reward-to-go
    - advantage estimator
        - use `Q` as advantage (no baseline version)
        - use `Q - V` as advantage, `V` given by value fucntion estimator implemented with neural network
        - use `GAE`
3. Critic: Value function estimator
    - use MLP as value function estimator
    - calculate MSE loss with predicted `V` and given `Q` (acutally this is biased since we use Q(at, st) on behalf of V(st)) 
4. Sample: get action sample from the predicted $\pi_{\theta}(a_t|s_t)$ distribution
    ```python
    # use the most recent ob to decide what to do
    ac = policy.get_action(ob)
    # take that action and get reward and next ob
    next_ob, rew, done, info = env.step(ac)
    ```

# Baseline

# Result
1. https://wandb.ai/yangzf23-independent-developer/cs285_hw2/workspace?nw=nwuseryangzf23
