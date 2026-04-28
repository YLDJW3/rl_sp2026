# Lecture 18: Offline RL Algorithms

Source: `lectures/lec-18.pdf`

## Big picture

This lecture turns the previous lecture's offline RL motivation into concrete algorithmic families.

The main question is:

> How can we do RL from a fixed dataset without letting the policy or Bellman backup query unsupported actions?

The main methods are:

1. **policy constraints**, which keep the learned policy close to the data;
2. **implicit Q-learning (IQL)**, which avoids evaluating out-of-dataset actions in the Q update;
3. **conservative Q-learning (CQL)**, which intentionally lowers Q-values for unsupported actions;
4. **offline-to-online RL**, which uses offline data to accelerate later online fine-tuning;
5. **model-based offline RL**, which uses learned models but must handle model exploitation and OOD states.

The main message is that offline RL algorithms differ in implementation, but most are trying to solve the same distributional shift problem: avoid trusting value estimates or model predictions on actions and states that the data does not support.

## Slide-by-slide note

### 1. Title

The lecture topic is **offline RL algorithms**.

The previous lecture explained why offline RL is hard. This lecture studies several practical algorithm templates.

### 2. Some principles for offline RL

The lecture recalls the key principles:

- use value-based methods such as Q-learning or actor-critic with a Q-function;
- fix the distributional shift problem somehow;
- use pessimism, as in conservative Q-learning;
- constrain the policy so it stays close to the dataset;
- avoid out-of-distribution actions inside updates, as in IQL.

The three recurring design goals are:

1. train the Q-function so it does not overestimate unsupported actions;
2. train the policy so it stays near the data distribution;
3. set up the update so it does not need to evaluate actions outside the dataset.

## Part 1: Policy constraints continued

### 3. Part 1: Policy constraints continued

This section continues the policy-constraint idea from Lecture 17.

The goal is to keep policy improvement from moving into action regions where the Q-function is unreliable.

### 4. Policy constraints review

The key picture is:

- Q-values near dataset actions are relatively reliable;
- Q-values far from dataset actions are unreliable;
- unconstrained policy improvement may choose actions with high but false Q-values;
- constrained policy improvement chooses the best action within the data-supported region.

The lecture also recalls forward KL and reverse KL:

```text
forward KL:  D_KL(beta || pi)   mode covering
reverse KL:  D_KL(pi || beta)   mode seeking
```

Reverse KL is often attractive in offline RL because it penalizes the learned policy for leaving the behavior-policy support, but does not force it to imitate every bad mode in the dataset.

### 5. Explicit policy constraint methods

One way to implement a policy constraint is to modify the actor objective directly.

Instead of only maximizing Q-values:

```text
max_pi E_s E_{a ~ pi(.|s)} [Q(s,a)]
```

we add a penalty that keeps the policy close to the behavior policy `beta`:

```text
max_pi E_s E_{a ~ pi(.|s)} [Q(s,a)]
       - alpha D(pi(.|s), beta(.|s))
```

For Gaussian or categorical policies, KL-style penalties are easy to compute and differentiate.

A very practical version is to add a behavior cloning term:

```text
actor objective = Q-improvement + BC regularization
```

This produces methods often named:

```text
DDPG+BC, TD3+BC, SAC+BC
```

These methods are simple and can work well, but they are not always the most principled way to handle offline distribution shift.

### 6. Reward-regularized policy constraints

Another way to implement the constraint is to modify the reward:

```text
r_bar(s,a) = r(s,a) - alpha * divergence_from_behavior(s,a)
```

This can be preferable because it accounts not only for immediate action divergence, but also for future divergence along the trajectory.

The downside is that it typically requires estimating the behavior policy:

```text
beta(a | s)
```

Estimating `beta` accurately can be hard, especially in high-dimensional or multimodal action spaces.

### 7. BRAC-like algorithm

BRAC stands for **Behavior Regularized Actor Critic**.

Its high-level recipe is:

1. train or estimate a behavior model from the dataset;
2. train a Q-function with Bellman backups;
3. update the actor to maximize Q while staying close to the behavior policy.

The actor objective has the form:

```text
maximize Q(s,a) - alpha * D(pi(.|s), beta(.|s))
```

This directly implements the "do not go too far from the data" principle.

### 8. AC+BC-like algorithm

Actor-critic plus behavior cloning is a simpler policy-constraint method.

The actor update mixes:

- a Q-maximization term, which pushes toward high-value actions;
- a BC term, which keeps actions near the dataset.

For deterministic actor-critic, this often looks like:

```text
maximize_pi   lambda Q(s, pi(s)) - ||pi(s) - a_data||^2
```

or, for stochastic policies:

```text
maximize_pi   lambda Q(s,a) + log pi(a_data | s)
```

The BC term is not just imitation. It is a distributional-shift control mechanism.

### 9. Implicit policy constraint methods

Implicit policy constraints avoid writing an explicit KL penalty in the actor objective.

A common derivation starts from a constrained policy-improvement problem and uses duality to show that the solution can be approximated by **advantage-weighted maximum likelihood**:

```text
maximize_pi E_{(s,a) ~ D}
[
    w(s,a) log pi(a | s)
]
```

where the weight is larger for high-advantage dataset actions:

```text
w(s,a) = exp(A(s,a) / lambda)
```

This is the idea behind AWR/AWAC-style methods.

The key point is that the policy is trained only on dataset actions, but it emphasizes the better ones.

### 10. AWAC-like algorithm

AWAC stands for **Advantage-Weighted Actor-Critic**.

The basic actor update is:

```text
maximize_pi E_{(s,a) ~ D}
[
    exp(A(s,a) / lambda) log pi(a | s)
]
```

where:

```text
A(s,a) = Q(s,a) - V(s)
```

Advantages:

- simple;
- stable;
- policy stays close to data because it is trained by weighted supervised learning;
- good dataset actions get more weight than bad dataset actions.

Possible problems:

- it can only directly imitate or reweight actions present in the dataset;
- performance depends on the quality of the critic;
- if the dataset lacks good actions, weighting cannot invent them.

## Part 2: Implicit Q-learning

### 11. Part 2: Implicit Q-learning

This section introduces **implicit Q-learning (IQL)**.

IQL is designed around a sharp offline RL principle:

> Do not evaluate actions that are not in the dataset.

### 12. Can we avoid all OOD actions in the Q update?

Standard Q-learning uses:

```text
y = r + gamma max_a' Q(s', a')
```

The `max_a'` is dangerous offline because it may choose an out-of-distribution action.

IQL asks whether we can get the benefit of policy improvement without ever explicitly evaluating OOD actions.

The key idea is to estimate the value of the **best action supported by the dataset**, rather than the best action over all possible actions.

### 13. Implicit Q-learning

IQL introduces a separate value network:

```text
V_psi(s)
```

This value is not simply the expectation under the dataset policy. Instead, it is trained to approximate an upper expectile of the dataset-action Q-values:

```text
Q(s,a),  a ~ dataset actions at state s
```

Intuitively:

```text
V(s) ≈ value of a good action that is still supported by the data
```

The distribution is induced by dataset actions only, so no OOD action has to be evaluated.

### 14. IQL Q update

Once `V(s)` is learned, the Q update can use:

```text
y = r + gamma V(s')
```

instead of:

```text
y = r + gamma max_a' Q(s', a')
```

This avoids the dangerous maximization over arbitrary actions.

The value network is fit by asymmetric squared loss, also called expectile regression:

```text
L_V = |tau - 1[Q(s,a) - V(s) < 0]| * (Q(s,a) - V(s))^2
```

For `tau > 0.5`, this fits an upper expectile, which behaves like a soft high-value statistic over dataset actions.

### 15. IQL policy extraction

After fitting `Q` and `V`, IQL extracts a policy with advantage-weighted behavioral cloning:

```text
maximize_pi E_{(s,a) ~ D}
[
    exp(beta (Q(s,a) - V(s))) log pi(a | s)
]
```

So IQL has three pieces:

1. fit `V` to upper expectiles of dataset-action Q-values;
2. fit `Q` using targets `r + gamma V(s')`;
3. train the actor with advantage-weighted regression on dataset actions.

The critical property is that the Q update does not require evaluating unseen actions.

### 16. Intermission

This slide separates IQL from the next major family: conservative value learning.

## Part 3: Conservative Q-learning

### 17. Part 3: Conservative Q-learning

This section introduces **conservative Q-learning (CQL)**.

Unlike IQL, CQL still uses value learning but modifies the Q objective so that unsupported actions receive lower values.

### 18. Conservative Q-learning

The motivating failure is overestimation:

```text
actual performance is low,
but learned Q-values are very high.
```

CQL adds a regularization term that pushes down large Q-values, especially for actions not supported by the dataset.

The goal is to learn a lower-bound or conservative estimate of value, so the policy cannot easily exploit overestimated OOD actions.

### 19. CQL objective intuition

CQL's regularizer has two opposing effects:

1. push Q-values down broadly over actions;
2. push Q-values up on dataset actions.

A common form is:

```text
alpha * (
    E_s [logsumexp_a Q(s,a)]
    - E_{(s,a) ~ D} [Q(s,a)]
)
```

plus the usual Bellman error.

The `logsumexp` term penalizes high Q-values over candidate actions. The dataset term prevents the method from simply pushing every value down equally.

### 20. CQL and unsupported actions

CQL is pessimistic where the data is weak.

If an action is not in the dataset but the current Q-function assigns it a high value, the conservative penalty pushes it downward.

This directly combats the main offline RL failure mode:

```text
policy chooses OOD action because Q mistakenly thinks it is good
```

### 21. CQL and maximum entropy regularization

The slide connects the CQL regularizer to maximum-entropy-style expressions.

The important practical point is that CQL often samples actions from several sources:

- random actions;
- actions from the current policy;
- dataset actions.

It then penalizes high values for non-dataset actions while preserving values on dataset actions.

Compared with policy constraints, CQL acts primarily on the **critic**:

```text
make Q conservative,
then policy improvement is safer.
```

## Part 4: Offline-to-online RL

### 22. Part 4: Offline-to-online RL

This section studies what happens when offline training is followed by online fine-tuning.

The goal is to use offline data for a strong starting point, then continue improving with real interaction.

### 23. The big picture

The desired pipeline is:

```text
offline RL pretraining -> fast, safe initial policy -> online RL fine-tuning
```

This is attractive because offline data can give the agent a good initial policy, while online RL can correct mistakes and explore beyond the dataset.

### 24. Example: offline-to-online RL with CQL

Offline-to-online training can behave unexpectedly.

For example, CQL may learn values that are too pessimistic during offline training. When online fine-tuning starts, the algorithm may spend many steps recalibrating the Q-function rather than improving the policy.

This can create a dip or plateau:

```text
offline policy is good,
online training starts,
performance temporarily stagnates or drops,
then eventually recovers.
```

Different offline RL algorithms have different versions of this problem.

### 25. The quest for one ultimate algorithm

The ideal algorithm would behave differently in the two phases:

- offline phase: be conservative, avoid unsupported actions, and prevent overestimation;
- online phase: explore and adapt aggressively using fresh environment feedback.

These objectives conflict. Strong pessimism is useful offline but can slow online exploration and value recalibration.

So it is hard to design one algorithm that is optimal for both offline pretraining and online fine-tuning.

### 26. An embarrassingly effective method

One simple offline-to-online method is:

1. keep two buffers:
   - an offline dataset buffer;
   - an online replay buffer;
2. initialize the actor and value function from scratch;
3. run online RL;
4. train each batch using half offline data and half online replay data.

This is unsatisfying because it does not do sophisticated offline pretraining, but it can work surprisingly well.

The offline data acts as a stabilizing source of useful experience while online RL learns and corrects itself.

### 27. An unlikely hero: diffusion and flow policies

The lecture notes that actor representations based on diffusion models or flow matching often work well in offline-to-online RL.

One possible reason is that these models can represent complex, multimodal behavior policies. This helps keep generated actions close to the offline data distribution while still allowing improvement during online fine-tuning.

Surprisingly, some methods train the diffusion or flow model mostly with supervised learning rather than direct RL gradients through the entire generative process.

### 28. Why is it hard to use diffusion as the actor in RL?

A diffusion or flow policy samples an action through many denoising or transformation steps.

Using it directly as an RL actor can be difficult because:

- action likelihoods and gradients can be expensive;
- policy-gradient updates through many sampling steps can be awkward;
- Q-gradient actor updates may require differentiating through the generative process;
- training can be much slower than with a simple Gaussian actor.

Some work does train diffusion policies with RL, but the lecture emphasizes that avoiding full RL training of the diffusion model may be part of why these methods work well offline-to-online.

### 29. IDQL: offline-to-online RL with diffusion

IDQL combines IQL-style value learning with diffusion policies.

The policy is trained to imitate high-value dataset actions, similar to advantage-weighted regression, but the actor is represented by a diffusion model.

Why this can be good:

- diffusion can represent multimodal action distributions;
- policy samples remain close to the data distribution;
- it can use offline data safely.

Why it can be bad:

- diffusion sampling is more expensive than a simple actor;
- direct online policy improvement can be more complicated;
- the method may depend heavily on the quality of the learned value weights.

### 30. Flow Q-learning

Flow Q-learning uses a flow or diffusion-like model to provide an action distribution, but the actor used for RL can be a simpler neural network conditioned on latent noise.

The idea is to keep the actor close to a learned generative policy while allowing efficient RL updates.

This can be useful because:

- the generative model captures the offline action distribution;
- the RL actor can still be optimized efficiently;
- the latent variable can provide diverse in-distribution actions.

### 31. Diffusion steering

Diffusion steering uses a diffusion or flow model as an action generator and then performs efficient online RL in its latent space.

The key idea:

```text
sample latent noise z
diffusion/flow model maps z to an in-distribution action
run RL over z rather than directly over raw actions
```

Because every latent sample maps through the learned generative model, the resulting actions tend to stay in distribution.

This directly addresses the offline RL problem:

```text
avoid OOD actions while still allowing online policy improvement
```

## Part 5: Model-based offline RL

### 32. Part 5: Model-based offline RL

The final section connects offline RL with model-based RL.

Model-based offline RL seems natural because a learned model can answer "what if" questions without real environment interaction. But this is exactly where offline distribution shift becomes dangerous.

### 33. How does model-based RL work offline?

A learned model predicts:

```text
p_hat(s' | s,a),   r_hat(s,a)
```

and lets the agent simulate actions that are not in the dataset.

The problem is that if the policy drives the model into OOD states or actions, the model's predictions can become invalid. Since offline RL cannot collect new real data, those model errors cannot be corrected by interaction.

So model-based offline RL has the same core issue as model-free offline RL:

```text
the optimizer may exploit errors in parts of the learned object that are not supported by data
```

For model-free offline RL, the learned object is usually `Q`.
For model-based offline RL, the learned object is the dynamics model.

### 34. MOPO: Model-Based Offline Policy Optimization

MOPO handles model exploitation with an uncertainty penalty.

The learned model is used to generate synthetic rollouts, but rewards are penalized when the model is uncertain:

```text
r_penalized(s,a) = r_hat(s,a) - lambda * uncertainty(s,a)
```

This makes uncertain model predictions less attractive.

The policy is therefore encouraged to use model rollouts only where the model is confident, which usually means near the dataset distribution.

Related idea: MOReL constructs a pessimistic MDP where uncertain transitions are treated as bad or terminal, so the policy avoids poorly supported regions.

### 35. COMBO: Conservative model-based RL

COMBO combines model-based rollouts with conservative value learning.

The model generates additional transitions, but the Q-function is trained conservatively so that out-of-distribution or model-generated actions do not receive overly optimistic values.

At a high level:

1. train a dynamics model on the offline dataset;
2. generate model rollouts, often short and starting from dataset states;
3. train a conservative Q-function on real and model data;
4. learn a policy from the conservative value estimate.

The key idea is that model-based data can improve coverage and sample efficiency, but conservative value learning is still needed to prevent model-error exploitation.

## Main takeaways

- Offline RL algorithms mainly differ in how they prevent distributional shift from corrupting policy improvement.
- Policy-constraint methods keep the actor close to the behavior data, either explicitly with KL/BC penalties or implicitly with advantage-weighted regression.
- IQL avoids OOD actions in the Q update by replacing `max_a Q(s,a)` with an expectile value over dataset actions.
- CQL makes the critic pessimistic by pushing down Q-values for non-dataset actions while preserving values on data actions.
- Offline-to-online RL is useful but tricky because offline pessimism can conflict with online exploration and value recalibration.
- Diffusion and flow policies can help represent multimodal data-supported action distributions for offline-to-online learning.
- Model-based offline RL must control model exploitation, often with uncertainty penalties or conservative value learning.

## One-sentence summary

Lecture 18 explains the main families of offline RL algorithms: constrain the policy, avoid OOD actions in value backups, pessimistically regularize Q-values, and use offline data or learned models in ways that keep policy improvement inside the support of the dataset.
