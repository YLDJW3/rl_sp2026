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
1. Baseline can reach average eval return 300, but it's not stable in each sample.
2. As the `blr` or `bgs` decrase, the converge speed becomes slower, the average eval return is still negative after 100 iterations.
3. Increase the `-n` to 200 do increase the return, but return is still unstable in each sample (e.g. average eval return reach 393 at 196-th iteration but fall down to negative in 197-th iteration)
4. `na` obviously improves the learning curve
    no-baseline with `na` reach an average eval return over 200, simliar to baseline without `na`
    baseline with `na` reach an average eval return over 600 in 97th iteration (unstability still exists), slightly higher than other learning curves

# Advantage estimator
1. GAE is not necessarily better or worse than n-step returns in general. GAE and n-step returns are just different ways to control the bias-variance tradeoff in policy gradient estimation, and they both have a hyperparameter (`λ` for GAE and `n` for n-step returns) that needs to be tuned in practice
2. In policy gradient methods, GAE is often more popular than n-step returns
3. In off-policy RL algorithms, n-step returns are more popular than GAE 
4. Result
    1. Find a fatal bug in critic's `update`, https://github.com/YLDJW3/rl_sp2026/commit/d577278ad07d272a01deaac39f7419cd08eee972
        ```python
        # q_values is (N, ) while values is (N, 1), squeeze values to (N, )
        values = self.forward(obs).squeeze(-1)
        assert values.shape == q_values.shape
        loss = (q_values - values).square().mean()
        ```
    2. when `lambda=0`, GAE is the 1-step Monte Carlo return (low variance, high bias)
    3. when `lambda=1`, GAE is the full Monte Carlo return (high variance, unbiased)
    4. eval average return has high variance, but does reach 150 in some steps, meeting the hw requirement

# Hyperparameter experiment
1. InvertedPendulum give 1 point for 1 environment step
2. Log
    1. Version 0: reach 1000 at step 74 (envstep 380k), but fall down to < 10 quickly
    2. Version 1: reach 1000 at step 70, final return 1000, never fall down to < 100
    ```makefile
    uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
		--exp_name pendulum \
		-rtg \
		-na \
		--use_baseline
    ```
    3. Version 2: reach 1000 at step 35 (env step 181k)
    ```makefile
    uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
		--exp_name pendulum \
		-rtg \
		-na \
		--use_baseline \
		--gae_lambda 0.99 \
		--discount 0.99 \
		-l 3 \
		-s 128 \
		-lr 0.01 \
		-blr 0.01 \
		-bgs 10
    ```
    4. Version 3: reach 1000 at step 42 (env step 89k), batch_size is decreased to 2000, meets the `env step < 100k` requirement
    ```
    	uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 2000 -eb 1000 \
		--exp_name pendulum \
		-rtg \
		-na \
		--use_baseline \
		--gae_lambda 0.98 \
		--discount 0.995 \
		-l 2 \
		-s 64 \
		-lr 0.01 \
		-blr 0.01 \
		-bgs 5
    ```
3. Conclusion
    1. incrase the `batch_size` actually decrease the policy update frequency (measured by env step), so we should choose a proper `batch_size` to ensuere sample efficiency. `batch_size=2000` is a good choice.
    2. **size of neural network** matters, larger network requires more samples to train, so we should choose proper `l` and `s`. `l=2` and `s=64` is large enouth to learn a good policy with eval average return over 900, and small enough to train with 100k env steps.
    3. `lr` and `blr` do matter, we want to adjust them to ensure the learning curve is increasing, but not too high to cause instability
    4. `na`, `rtg` and `use_baseline` do improve the learning curve

# Result
1. https://wandb.ai/yangzf23-independent-developer/cs285_hw2/workspace?nw=nwuseryangzf23
