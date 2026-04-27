# Lecture 16: Model-Based RL Algorithms

Source: `lectures/lec-16.pdf`

## Big picture

This lecture turns the previous lecture’s ideas into concrete **algorithms**. It covers three connected topics:

1. how to **plan with a learned model**;
2. how to **train policies using a model** without letting long-horizon model errors dominate;
3. how to build and use **latent state-space models** so planning can happen in a learned compact state rather than raw observations.

The main message is:

- open-loop planning is simple but limited;
- policy learning through a model is attractive, but naive long rollouts are brittle;
- short model rollouts plus model-free updates often work better;
- in realistic high-dimensional settings, the model is usually built in a learned latent space.

## Slide-by-slide note

### 1. Title

The lecture topic is **model-based RL algorithms**. The focus is no longer just “should we trust the model?” but “how do we actually use it?”

## Part 1: Planning with models

### 2. Part 1: Planning with models

This section studies how to choose actions directly from a learned model, even before introducing a separately trained policy.

### 3. Last time…

The lecture recalls the previous point:

- uncertainty helps with distributional shift;
- but instead of immediately running a standard RL algorithm, we can directly **plan** with the model.

Planning here means using the model to select actions online, without needing a permanently stored policy.

### 4. The deterministic case

This is the easiest setting:

- deterministic dynamics;
- deterministic rewards;
- optimize an action sequence directly.

Conceptually, planning reduces to trajectory optimization.

### 5. The stochastic open-loop case

Now the environment is stochastic, but the plan is still an open-loop action sequence chosen at the beginning.

The slide asks why this is suboptimal. The answer is that an open-loop plan cannot react to randomness after execution starts. Once the world deviates from the nominal prediction, the action sequence stays fixed.

### 6. Aside: terminology

The lecture clarifies:

- **open-loop**: choose all actions up front;
- **closed-loop**: actions depend on the current observed state.

Closed-loop control is generally better in stochastic settings because it can adapt online.

### 7. The stochastic closed-loop case

This is the ideal case conceptually: use the model to reason about a policy or feedback controller, not just one fixed action sequence.

But it is also more computationally difficult.

### 8. But for now, open-loop planning

The lecture deliberately restricts attention to open-loop planning first because it is much simpler and often a useful baseline.

### 9. Stochastic optimization: random shooting

The simplest planning algorithm is **guess and check**:

1. sample many candidate action sequences;
2. roll them out in the model;
3. keep the one with the highest predicted return.

This is the **random shooting** method.

It is crude but often surprisingly competitive when parallel computation is available.

### 10. Cross-entropy method (CEM)

The next improvement is **CEM**:

1. maintain a distribution over action sequences, usually Gaussian;
2. sample candidate sequences;
3. keep the top-performing samples;
4. refit the Gaussian to those elites;
5. repeat.

So CEM is still sampling-based planning, but it iteratively concentrates search around better regions instead of resampling blindly each time.

### 11. Pros and cons of these planning methods

The lecture emphasizes two limitations:

1. harsh dimensionality limits, because long horizons create huge action-search spaces;
2. open-loop planning only.

But the upsides are also clear:

1. very fast when parallelized;
2. extremely simple.

The slide also situates them among alternatives like MCTS, LQR-style trajectory optimization, and motion-planning methods such as RRT.

### 12. Planning with uncertainty

Now uncertainty-aware models enter the planning procedure.

Instead of planning under one deterministic predictor, we can plan under:

- an ensemble of models;
- or a posterior / approximate posterior over models.

The planner then has to decide whether to optimize mean predicted reward, pessimistic reward, optimistic reward, or something else that accounts for uncertainty.

### 13. Real-world example

This slide gives a dexterous manipulation example to show that the previous planning ideas are not merely theoretical. The point is that uncertainty-aware model-based control can work on challenging robotic tasks.

## Part 2: Policy learning with models

### 14. Part 2: Policy learning with models

Now the lecture switches from online planning to learning reusable policies with help from the model.

### 15. Notional “learned simulator” algorithm

This slide recalls the simple recipe:

1. learn a model;
2. optimize a policy using that model.

The rest of this section explains why the naive version is fragile.

### 16. The stochastic closed-loop case

Policy learning is naturally a closed-loop problem, because a policy maps current states to actions.

That is better than open-loop planning in stochastic environments, but it introduces another difficulty: differentiating or estimating returns through long model rollouts.

### 17. Model-free optimization with a model

The slide contrasts two ways to optimize a policy using the model:

- **policy gradient / score-function estimator**
- **backpropagation through the model / pathwise gradient**

Backpropagation through the model can be sample efficient, but it requires multiplying many Jacobians across time. This can become unstable or chaotic.

Policy-gradient style updates may be more stable, though often noisier.

### 18. Potential problem with this approach

The lecture asks the key question: what happens when model errors accumulate over many imagined time steps?

That leads directly to the next slide.

### 19. The curse of long model-based rollouts

Prediction errors compound with horizon length.

Even small one-step errors can become large multi-step trajectory errors, which means gradients or policy updates are based on unrealistic states. So long-horizon optimization inside the model can optimize the wrong objective.

### 20. How to get away with short rollouts?

This slide explains the tradeoff:

- long rollouts see farther into the future but accumulate large error;
- very short rollouts are much more accurate but may not expose later consequences directly;
- short rollouts started from real states let us cover many parts of the state distribution while keeping model usage local.

This is the core modern compromise in practical model-based RL.

### 21. Model-based RL with short rollouts

The main idea is:

1. collect real transitions;
2. fit a model;
3. start from real states from the replay buffer;
4. generate only short imagined rollouts;
5. use those synthetic transitions to help train value functions or policies.

This uses the model where it is most trustworthy: locally around real data.

### 22. Model-free optimization with a model: Dyna

The lecture connects the idea to **Dyna**.

Dyna uses a learned model to generate extra transitions, then feeds those transitions into ordinary model-free RL updates. So the model accelerates learning, but the underlying optimization remains model-free.

### 23. General Dyna-style recipe

The slide highlights why this is attractive:

- it only needs short rollouts, sometimes even one step;
- it still exposes learning algorithms to diverse states;
- it avoids relying on the model for full long-horizon optimization.

This is one of the most practically important model-based RL templates.

### 24. Model-accelerated off-policy RL

Here the architecture becomes concrete:

- keep a replay buffer of real transitions;
- learn model, value, and policy from that data;
- generate additional synthetic transitions from the model;
- mix them into off-policy learning.

This is the general pattern behind many modern algorithms.

### 25. MBA, MVE, MBPO

The lecture names several representative examples:

- Model-Based Acceleration (MBA)
- Model-Based Value Expansion (MVE)
- Model-Based Policy Optimization (MBPO)

The purpose of this slide is not the derivation of each method, but the common idea:

- use the model to improve sample efficiency;
- limit how much the learning algorithm depends on long imagined trajectories.

### 26. Intermission

This is a section break before the lecture changes focus from algorithmic usage to model representation.

## Part 3: Representing the model

### 27. Part 3: Representing the model

The lecture now asks what kind of model we should learn in high-dimensional environments, especially when observations are images.

### 28. Notional “learned simulator” algorithm

The same recipe reappears, but now the emphasis is on the structure of the model itself.

### 29. Latent state-space models

The lecture introduces **latent state-space models**.

Instead of predicting directly in observation space, we learn:

- a latent state;
- latent dynamics;
- a decoder back to observations.

The slide asks:

- what is the prior?
- what is the decoder?
- what is the encoder?

This is intentionally connecting model-based RL back to latent-variable models and VAEs from earlier lectures.

### 30. Latent state-space models

This slide suggests that the latent-state dynamics can be represented by a sequence model such as an LSTM or Transformer.

So the model is no longer just a hand-designed transition function. It is a learned recurrent or autoregressive system in latent space.

### 31. State-space models

The lecture separates three pieces:

- observation model;
- dynamics model;
- reward model.

In a fully observed standard model, these are written directly in observed state space.
In a latent model, they are written in latent space, and observations are connected through the encoder/decoder.

This is the right abstraction for high-dimensional environments where raw observations are not convenient states for control.

### 32. How to infer latent states

The slide compares several encoders / inference schemes:

- full smoothing posterior;
- single-step encoder.

The tradeoff is standard:

- full smoothing is more accurate because it uses more temporal context;
- single-step inference is simpler and cheaper.

The lecture says it will mostly talk about the simpler version for now.

### 33. Deterministic encoder: why might this be bad?

The warning is that a deterministic encoder may throw away uncertainty.

If multiple latent states could explain the same observation history, a deterministic mapping cannot represent that ambiguity. This matters especially under partial observability, but even in fully observed visual domains it can still be limiting.

### 34. Stochastic encoder and richer posteriors

The slide therefore notes that many practical methods use:

- stochastic encoders;
- or more sophisticated filtering / smoothing posteriors.

This brings the discussion back to variational inference: the encoder is an approximate posterior over latent states.

### 35. Periodic training / rollout procedure

The slide indicates a practical loop where model learning, imagined rollouts, and policy updates happen repeatedly every `N` steps.

The exact diagram is compact, but the main point is that representation learning and control are interleaved rather than fully separated.

### 36. Actor-critic with learned representations

Now the lecture combines actor-critic RL with latent-state learning from replay-buffer data.

The slide again mentions that a full filtering or smoothing posterior can be even better. The reason is that a better posterior gives better latent states, which improves both prediction and control.

### 37. Actor-critic + model-based RL

This slide shows the integrated architecture:

- replay buffer for real experience;
- learned latent representation and model;
- actor-critic updates;
- model-generated or model-informed targets / rollouts.

The message is that modern systems often blur the boundary between “representation learning,” “model learning,” and “policy learning.”

### 38. Example applications

The final slide points to applications where learned representations and model-based rollouts work together.

The takeaway is that latent-space modeling is not just a compression trick. It is what makes model-based RL feasible in complex observation spaces.

## Main takeaways

- Planning with a model can be done by simple sampling methods like random shooting and CEM.
- Open-loop planning is easy to implement but fundamentally limited in stochastic settings.
- Direct policy optimization through long model rollouts suffers from compounding model error.
- Short rollouts from real states are a practical compromise and motivate Dyna-style methods.
- Modern model-based RL often accelerates off-policy RL rather than replacing it.
- In high-dimensional domains, the model is usually learned in a latent state space.
- The encoder in a latent model is an approximate inference mechanism, so variational ideas remain central.

## One-sentence summary

Lecture 16 explains how model-based RL is actually implemented: use simple planners or short-horizon synthetic rollouts, and learn compact latent dynamics models so control is possible in high-dimensional environments.
