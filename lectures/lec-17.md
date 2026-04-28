# Lecture 17: Offline Reinforcement Learning

Source: `lectures/lec-17.pdf`

## Big picture

This lecture introduces **offline reinforcement learning**: training an RL policy from a fixed dataset, without further environment interaction during training.

The main message is:

- offline RL tries to combine the strengths of data-driven learning and RL optimization;
- the fixed dataset makes the problem useful, but also creates a severe distributional shift problem;
- standard off-policy RL can fail badly because it asks counterfactual questions about actions that were not tried in the data;
- practical offline RL methods usually constrain the policy, make the value function pessimistic, or avoid out-of-distribution actions in the update.

## Slide-by-slide note

### 1. Title

The lecture topic is **offline reinforcement learning**.

Offline RL is different from ordinary off-policy RL because the agent cannot keep collecting new data while learning. It must learn entirely from a fixed dataset.

## Part 1: What is offline reinforcement learning?

### 2. Part 1: What is offline reinforcement learning?

This section motivates the problem and explains why it is not just ordinary off-policy RL.

### 3. Offline reinforcement learning

The slide contrasts:

- **on-policy RL**: data is collected by the current policy;
- **off-policy RL**: data can come from older policies, but the learner may still interact with the environment;
- **offline RL**: the learner receives a fixed dataset and cannot gather additional online experience.

The key question is why this harder setting is useful.

### 4. Why offline RL?

The lecture uses generative AI as an analogy for data-driven learning. Modern models can learn impressive behavior from large datasets because the data already contains structure, patterns, and examples that can be recombined.

Offline RL asks whether we can do something similar for decision making: use prior data to learn behavior that is not merely copied from one trajectory, but optimized toward a goal.

### 5. Data-driven AI vs reinforcement learning

The slide frames the tradeoff:

- data-driven AI learns about the real world from data, but does not directly optimize a reward objective;
- reinforcement learning optimizes a goal and can produce emergent behavior, but often requires online interaction or simulation;
- offline RL tries to get both: use real-world data and still perform RL-style optimization.

This is why offline RL is appealing for robotics, healthcare, recommendation systems, operations, and other domains where online trial-and-error is expensive or unsafe.

### 6. The big picture

Offline RL is one way to use prior data for RL training.

The pipeline is:

1. collect or reuse a dataset of behavior;
2. train offline, without interacting with the environment;
3. optionally deploy or fine-tune online later.

Offline RL can make later online learning faster and safer, but it is not the only pretraining approach. For example, policies can also be pretrained by imitation learning.

### 7. How is offline RL possible?

The lecture gives three intuitions:

1. **Find the good parts of the dataset.** Even if the dataset contains mixed-quality behavior, the algorithm can identify which actions and trajectories were better.
2. **Generalize.** Good behavior in one state may imply good behavior in similar states.
3. **Stitching.** Parts of good trajectories can sometimes be recombined into a better policy than any single full trajectory in the data.

This last point is one of the main reasons offline RL can be more powerful than pure imitation learning.

### 8. What do we expect offline RL methods to do?

The slide distinguishes large-scale and small-scale forms of stitching.

At a large scale, offline RL may combine segments from different trajectories to produce a new route or strategy. At a smaller scale, the same idea appears inside value-function learning: a Bellman backup can connect a good action from one transition with a good future state from another transition.

So offline RL is not just copying the dataset. It tries to use the dataset as raw material for policy improvement.

## Part 2: Distributional shift

### 9. Part 2: Distributional shift

This section introduces the central technical obstacle in offline RL.

The fixed dataset is useful because it avoids online interaction, but it also means the learner cannot test uncertain actions. That makes distributional shift much more dangerous than in online RL.

### 10. The basic challenge with offline RL

Standard RL repeatedly does something like:

1. fit a value function or model to estimate return;
2. improve the policy using that estimate;
3. generate new samples by running the improved policy.

Offline RL breaks the third step. The improved policy may choose actions that are not represented in the dataset, but the learner cannot run the policy to check whether those actions are actually good.

### 11. Counterfactual queries

The fundamental problem is that offline RL asks counterfactual questions:

> What would have happened if we took this different action?

If the dataset did not contain that action in that state, the answer is not directly known.

Online RL can try the action and observe the result. Offline RL cannot. Therefore it must either:

- avoid such actions;
- estimate them conservatively;
- or use generalization carefully.

The difficulty is that we still want generalization. If the method never generalizes beyond exact dataset actions, it cannot improve much over the behavior policy.

### 12. Revisiting distributional shift

The slide uses an exam analogy: training on one distribution and being evaluated on a very different distribution can fail badly.

In offline RL, the training distribution is the dataset distribution, but the learned policy may induce a different state-action distribution.

This is the core mismatch:

```text
data distribution:       (s, a) from behavior policy
learned policy queries:  (s, a) from improved policy
```

The more aggressively the policy improves, the more likely it is to query unsupported actions.

### 13. Distribution shift in ERM

The lecture briefly compares offline RL to ordinary supervised learning.

In empirical risk minimization, we usually train by maximum likelihood or supervised loss on the dataset, and we often rely on neural networks to generalize to nearby test examples.

In offline RL, this ordinary generalization problem becomes more severe because the learned policy actively seeks actions with high estimated value. If the value estimate is wrong out of distribution, the policy will exploit that error.

### 14. Where does it appear for each off-policy RL method?

The slide begins connecting distributional shift to standard off-policy RL algorithms.

The main idea is that every off-policy method has a place where it must reason about actions or states that may not have been well covered by the data.

### 15. Distribution shift in value-based methods

For Q-learning-style methods, the risky operation is the Bellman backup:

```text
y = r + gamma max_a' Q(s', a')
```

The maximization over `a'` may select an action that is not in the dataset. If `Q(s', a')` is overestimated for that out-of-distribution action, the backup propagates an erroneous high value.

This is one reason offline Q-learning can produce severe overestimation.

### 16. Distribution shift in importance sampling and policy constraints

For importance-sampling-style off-policy evaluation or policy gradients, the problematic term is the action probability ratio:

```text
pi(a | s) / beta(a | s)
```

where `beta` is the behavior policy that generated the data.

If this ratio is far from 1, the estimator has high variance. If `beta(a | s)` is tiny or zero, the offline data provides very little information about that action.

This motivates the policy-constraint idea:

```text
just do not go too far from the data-generating policy
```

### 17. Example with soft actor-critic

The slide shows that an off-policy actor-critic such as SAC can become overconfident offline.

As training proceeds, the learned Q-values can become very large even when actual performance does not improve. The policy learns to exploit value-function errors instead of learning genuinely better behavior.

This is a characteristic offline RL failure mode:

```text
the policy does what the Q-function thinks is good,
but the Q-function is wrong on unsupported actions.
```

### 18. Issues with generalization are not corrected

In online RL, sampling error and function approximation error can sometimes be corrected because the agent eventually interacts with the environment and gets new feedback.

In offline RL, those errors are not automatically corrected. If the algorithm makes a bad extrapolation, it may keep reinforcing it through Bellman backups and policy improvement.

So offline RL magnifies ordinary RL problems:

- bootstrapping error;
- function approximation error;
- overestimation bias;
- distributional shift.

## Part 3: Policy constraints

### 19. Part 3: Policy constraints

This section studies one major family of offline RL solutions: constrain the learned policy so it stays close to the data.

### 20. Some principles for offline RL

Many offline RL methods follow similar high-level principles:

1. use value-based methods such as Q-learning or Q-function actor-critic;
2. fix distributional shift somehow;
3. use pessimism, as in conservative Q-learning;
4. constrain the policy so it stays close to the data;
5. avoid out-of-distribution actions in backups and policy updates.

The three broad strategies are:

- train the Q-function so it does not overestimate unsupported actions;
- train the policy so it remains close to the behavior distribution;
- design the update so it does not query out-of-sample actions.

### 21. Policy constraints

The basic policy-constraint idea is:

```text
maximize expected return
subject to pi staying close to the behavior policy beta
```

For example:

```text
maximize_pi   J(pi)
subject to    D(pi(. | s), beta(. | s)) <= epsilon
```

This directly addresses the offline RL problem: if the learned policy only chooses actions similar to the dataset actions, the value function is less likely to be queried far out of distribution.

### 22. Why does a policy constraint fix the problem?

The slide illustrates that Q-values are more reliable near the data and less reliable far away from it.

Without a constraint, policy improvement may choose an action with a high but unreliable value estimate. With a constraint, the policy is forced to choose among actions that are supported by the dataset.

This may prevent the policy from finding the unconstrained optimum, but it can improve actual performance because it avoids exploiting Q-function errors.

### 23. What kind of constraint do we use?

The lecture compares forward KL and reverse KL.

Forward KL:

```text
D_KL(beta || pi)
```

is often called **mode covering**. It penalizes the learned policy if it fails to assign probability to actions that appeared in the behavior data.

Reverse KL:

```text
D_KL(pi || beta)
```

is often called **mode seeking**. It penalizes the learned policy for putting probability mass on actions that are unlikely under the behavior data.

### 24. Forward KL vs reverse KL

The difference matters because the behavior data can contain both good and bad actions.

A forward-KL constraint can force the learned policy to cover all modes of the behavior distribution, including bad actions. That is useful when coverage is important, but it can prevent improvement.

A reverse-KL constraint allows the policy to select one high-value behavior mode and ignore lower-value modes, while still avoiding actions outside the data support.

### 25. Why prefer the ideal constraint?

The ideal offline RL constraint would allow any action that could plausibly occur under the data distribution, but penalize actions that are truly outside the behavior support.

This is attractive because:

- we do not want to keep bad behavior just because it appears in the dataset;
- we do want to allow high-value behavior that is supported by the data;
- we only want to penalize actions that the data gives us no reason to trust.

The problem is that this ideal support constraint is hard to implement exactly, especially with continuous actions and unknown behavior policies. Practical algorithms use approximations such as learned behavior models, KL penalties, conservative Q penalties, or expectile-style objectives.

## Main takeaways

- Offline RL trains from a fixed dataset with no environment interaction during training.
- It is useful because it can use real-world data while still optimizing a reward objective.
- Its main difficulty is distributional shift: the learned policy may choose actions that are not covered by the dataset.
- Standard off-policy RL methods can fail offline because Bellman backups and policy improvement query unsupported actions.
- Policy constraints reduce this problem by keeping the learned policy near the behavior distribution.
- Reverse-KL-style or support-style constraints are attractive because they can discard bad behavior while avoiding truly out-of-distribution actions.
- Offline RL methods often combine policy constraints with pessimistic value learning or update rules that avoid out-of-sample actions.

## One-sentence summary

Lecture 17 explains offline RL as reward-driven learning from a fixed dataset, where the central challenge is distributional shift from counterfactual action queries, and the main remedy introduced here is constraining policy improvement to stay within the support of the data.
