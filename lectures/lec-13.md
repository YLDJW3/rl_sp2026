# Lecture 13: Control as Variational Inference

Source: `lectures/lec-13.pdf`

## Big picture

This lecture takes the "control as inference" story from lecture 12 and pushes it further in two directions:

1. derive **maximum-entropy RL** from a variational-inference objective that avoids changing the environment dynamics;
2. use the same framework to motivate **inverse reinforcement learning** and then connect IRL to **adversarial imitation learning**.

The key ideas are:

- plain control-as-inference produces a soft policy, but it also implicitly changes transition probabilities in an "optimistic" way;
- variational inference fixes this by introducing an approximate trajectory distribution `q` that keeps the real dynamics and only changes the policy;
- the resulting objective is reward plus policy entropy;
- this gives the standard max-ent RL view, which then leads naturally into IRL and GAIL.

## Part 1: Control as inference recap

### 1. Title

The lecture topic is **control as variational inference**. The goal is to reinterpret RL and imitation learning through probabilistic inference.

### 2. Part 1: Control as inference recap

This section briefly recalls the previous lecture's probabilistic graphical model for decision making.

### 3. A probabilistic graphical model of decision making

The slide revisits the control-as-inference setup:

- state variables `s_t`
- action variables `a_t`
- optimality variables `O_t`

with:

```math
p(O_t=1 \mid s_t, a_t) \propto \exp(r(s_t,a_t)).
```

This means high-reward state-action pairs are more likely to be marked "optimal." The model does **not** assume exact optimality; instead it says good behavior is more probable.

### 4. Policy computation

From backward message passing we get:

```math
Q_t(s_t,a_t)=r(s_t,a_t)+\log \mathbb E[\exp(V_{t+1}(s_{t+1}))]
```

and

```math
V_t(s_t)=\log \int \exp(Q_t(s_t,a_t))\,da_t.
```

The induced policy is:

```math
\pi(a_t|s_t)=\exp(Q_t(s_t,a_t)-V_t(s_t)).
```

So the policy is a softmax over soft Q-values / advantages.

### 5. Summary

The recap slide emphasizes:

1. optimal control can be written as a probabilistic graphical model;
2. control can be solved by inference in that model;
3. the resulting algorithm looks like dynamic programming, except with a **soft** max instead of a hard max.

### 6. Part 2: Control as variational inference

Now the lecture asks: what is wrong with the previous derivation, and how can we fix it?

## Part 2: Control as variational inference

### 7. The optimism problem

The issue is subtle but important.

What we want is a policy:

```math
p(a_t \mid s_t, O_{1:T}),
```

meaning: given that high reward happened, what action was likely?

But if we condition naively on optimality, we also change:

```math
p(s_{t+1}\mid s_t,a_t,O_{1:T}) \neq p(s_{t+1}\mid s_t,a_t).
```

That means the inference problem is also changing the transition dynamics. Intuitively, the agent starts to behave as if the environment is more favorable than it really is. This is the **optimism problem**.

### 8. Addressing the optimism problem

The slide states the fix:

- find another distribution `q(s_{1:T}, a_{1:T})`
- keep the real dynamics inside `q`
- make `q` close to the posterior over trajectories conditioned on optimality

This is now exactly a variational-inference problem:

- observed variable `x = O_{1:T}`
- latent variable `z = (s_{1:T}, a_{1:T})`
- find `q(z)` to approximate `p(z|x)`

but with the important structural constraint that `q` should preserve the real dynamics.

### 9. Control via variational inference

The lecture defines a variational trajectory distribution of the form:

```math
q(s_{1:T}, a_{1:T})
=
p(s_1)\prod_t p(s_{t+1}\mid s_t,a_t)\,q(a_t\mid s_t).
```

Interpretation:

- initial state distribution is real
- transitions are real
- only the policy `q(a_t|s_t)` is free to change

So optimization is now only over the policy, not over the environment.

### 10. The variational lower bound

Apply the ELBO:

```math
\log p(x) \ge \mathbb E_{z\sim q(z)}[\log p(x,z)-\log q(z)].
```

Substituting:

- `x = O_{1:T}`
- `z = (s_{1:T}, a_{1:T})`
- and the constrained form of `q`

causes the initial-state and transition terms to cancel, leaving:

```math
\log p(O_{1:T})

\ge
\mathbb E_q\left[\sum_t r(s_t,a_t)-\log q(a_t|s_t)\right].
```

Equivalently:

```math
\sum_t \mathbb E_q\left[r(s_t,a_t)+\mathcal H(q(a_t|s_t))\right].
```

This is the key result:

- maximize reward
- and maximize action entropy

So **maximum-entropy RL** falls directly out of the variational lower bound.

### 11. Summary

This slide summarizes the new perspective:

- the variational distribution preserves the true dynamics;
- the free part is just the policy;
- the resulting objective is max-ent RL;
- the Bellman backup becomes the soft Bellman update.

The slide also notes common variants:

- discounted formulations
- temperature-scaled entropy terms

## Part 3: Maximum entropy RL algorithms

### 12. Part 3: Maximum entropy RL algorithms

This section turns the variational objective into familiar RL algorithms.

### 13. Policy gradient with soft optimality

From the variational objective:

```math
J(q)=\sum_t \mathbb E_q[r(s_t,a_t)+\mathcal H(q(a_t|s_t))].
```

So policy gradient becomes ordinary policy gradient plus an entropy bonus.

That is why entropy regularization is so common in practical policy-gradient methods: it is not just a heuristic, it has a probabilistic derivation here.

### 14. Q-learning with soft optimality

This slide compares standard and soft Q-learning.

Standard target:

```math
V(s')=\max_{a'} Q(s',a').
```

Soft target:

```math
V(s')=\text{softmax}_{a'} Q(s',a')
=
\log \int \exp(Q(s',a'))\,da'.
```

And the induced policy is:

```math
\pi(a|s)=\exp(Q(s,a)-V(s)).
```

So soft Q-learning is the control-as-variational-inference version of Q-learning.

### 15. Soft actor-critic

The slide gives the actor-critic version:

- critic learns soft Q-values
- value network / target network estimates the soft value
- actor maximizes expected Q plus entropy

The replay buffer diagram emphasizes that this is an **off-policy** algorithm. This is essentially the structure behind **Soft Actor-Critic (SAC)**.

The lecture notes that a temperature parameter is often added in front of the entropy term.

### 16. Inference = planning

This slide explicitly reconnects the RL algorithms to the earlier message-passing picture:

- backward messages produce the soft value recursion;
- the policy is derived from the resulting posterior over actions.

So the algorithmic view and the inference view are two faces of the same object.

## Part 4: Inverse reinforcement learning

### 17. Part 4: Inverse reinforcement learning

Now the lecture flips the problem:

- before, assume reward and solve for policy;
- now, observe demonstrations and solve for reward.

### 18. Why should we worry about learning rewards? (imitation perspective)

Pure imitation learning copies the expert's **actions**.

But human imitation often copies **intent**, not necessarily the exact low-level action. Different low-level actions can realize the same objective.

So if we want robust transfer or generalization, it may be better to infer the underlying reward rather than only cloning actions.

### 19. Why should we worry about learning rewards? (RL perspective)

This slide emphasizes that for many tasks, the reward is not obvious from observations alone. We may see successful behavior, but the underlying objective is ambiguous.

IRL tries to recover:

```math
r(s,a)
```

from demonstrations.

### 20. Inverse reinforcement learning

The problem statement:

- infer reward functions from demonstrations

But the slide immediately points out the ambiguity:

- many reward functions can explain the same observed behavior

So IRL is under-specified unless we add additional structure, such as the max-ent trajectory model.

### 21. Learning the optimality variable

Under the max-ent trajectory model:

```math
p(\tau \mid \psi) \propto p(\tau)\exp(r_\psi(\tau))
```

where `\psi` are reward parameters and:

```math
r_\psi(\tau)=\sum_t r_\psi(s_t,a_t).
```

Maximum likelihood of demonstrations becomes:

```math
\max_\psi \frac{1}{N}\sum_i \log p(\tau_i \mid \psi)
=
\max_\psi \frac{1}{N}\sum_i r_\psi(\tau_i) - \log Z.
```

The hard part is the partition function `Z`.

### 22. The IRL partition function

The partition function is:

```math
Z=\int p(\tau)\exp(r_\psi(\tau))\,d\tau.
```

Differentiating the log-likelihood gives a classic moment-matching form:

```math
\nabla_\psi \mathcal L

=
\mathbb E_{\tau \sim \text{expert}}[\nabla_\psi r_\psi(\tau)]
-
\mathbb E_{\tau \sim p(\tau|O_{1:T},\psi)}[\nabla_\psi r_\psi(\tau)].
```

So the reward is updated to:

- increase scores on expert trajectories
- decrease scores on trajectories sampled from the current soft-optimal policy

### 23. Estimating the gradient

To estimate the second expectation, run the current max-ent RL policy induced by the current reward.

So IRL becomes an alternating procedure:

1. optimize policy for current reward
2. compare policy trajectories to demonstrations
3. update reward

This is the direct max-ent IRL loop.

### 24. More efficient sample-based updates

The slide points out that fully re-solving RL after every reward update is expensive.

To reduce cost, one can:

- update the reward several times for a slightly stale policy
- and correct the mismatch using importance sampling

This is the motivation behind **Guided Cost Learning**.

### 25. Importance sampling

Importance weights correct for the fact that policy samples come from the old proposal distribution rather than exactly the current one.

So the policy update and reward update can be partially decoupled, improving efficiency while introducing some bias-variance tradeoff.

### 26. Guided cost learning algorithm

This slide summarizes the full guided cost learning loop:

1. start with a policy
2. collect policy samples
3. combine policy samples with demonstrations
4. update reward using weighted samples
5. update policy with respect to the learned reward
6. repeat

The slide emphasizes the game-like flavor of the resulting algorithm.

## Part 2: Adversarial imitation learning

### 27. Part 2: Adversarial imitation learning

This section makes the "game" interpretation explicit by connecting IRL to GANs.

### 28. Generative Adversarial Networks

The lecture quickly reviews GANs:

- generator produces samples
- discriminator distinguishes real vs generated samples
- the two are trained adversarially

This is just background for the imitation-learning analogy.

### 29. Inverse RL as a GAN

The key observation is that the optimal discriminator in the IRL setting has a form involving:

```math
\exp(r_\psi(\tau))
```

and the policy distribution `\pi(\tau)`.

So the discriminator and reward are tightly connected. With the right discriminator parameterization, optimizing the discriminator corresponds to learning the reward.

This also explains why importance weights disappear into the partition function in the derivation.

### 30. Inverse RL as a GAN

This slide gives the alternating optimization view:

- policy / generator produces trajectories
- demonstrations play the role of real data
- discriminator / reward tries to separate demos from policy samples
- policy tries to fool the discriminator

So adversarial imitation learning is not just vaguely similar to GANs; it is a direct consequence of the IRL derivation.

### 31. Can we just use a regular discriminator?

This slide discusses the practical simplification used in **GAIL**:

- instead of explicitly recovering a reward that corresponds exactly to the max-ent IRL derivation,
- use a standard binary classifier as the discriminator

Pros:

- simpler optimization
- fewer moving parts

Cons:

- the discriminator may no longer correspond exactly to the true IRL reward;
- it may not recover a reward with the right semantics.

### 32. IRL as adversarial optimization

The slide makes the equivalence explicit:

- reward function is minimized / policy is optimized against it
- classifier is maximized to distinguish demos from policy rollouts
- in the right parameterization, the discriminator's logit plays the role of the reward

So the reward-learning view and the classifier view are often the same object written differently.

### 33. Applications / examples

The final content slide shows applications such as humanoid motion imitation and robotic imitation. The point is that adversarial imitation learning can produce complex behavior without manually designing rewards.

### 34. End / wrap-up

The lecture closes on the idea that:

- max-ent RL comes from variational inference;
- IRL compares expert trajectories to trajectories from the current soft-optimal policy;
- adversarial imitation learning is a practical game-theoretic approximation of this loop.

## Main takeaways

- Naive control-as-inference has an **optimism problem** because conditioning on optimality changes both the policy and the dynamics.
- Variational inference fixes this by optimizing over a trajectory distribution `q` that keeps the real dynamics and only changes the policy.
- The resulting ELBO is exactly the **maximum-entropy RL objective**:

```math
\mathbb E_q\left[\sum_t r(s_t,a_t) - \log q(a_t|s_t)\right].
```

- Soft policy gradient, soft Q-learning, and SAC all fit naturally into this framework.
- In max-ent IRL, the reward gradient is:

```math
\mathbb E_{\tau\sim \text{expert}}[\nabla r_\psi(\tau)]
-
\mathbb E_{\tau\sim \pi_\psi}[\nabla r_\psi(\tau)].
```

- Guided cost learning improves efficiency with sample reuse and importance weighting.
- Adversarial imitation learning / GAIL can be understood as a practical adversarial approximation to max-ent IRL.

## One-sentence summary

This lecture shows that variational inference turns control into maximum-entropy RL once we constrain the policy to preserve real dynamics, and that the same max-ent trajectory model leads directly to inverse RL and then to adversarial imitation learning.
