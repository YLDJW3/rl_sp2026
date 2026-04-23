# Lecture 12: Variational Inference in RL

Source: `lectures/lec-12.pdf`

## Big picture

This lecture extends the previous one in three directions:

1. **amortized variational inference**: replace one variational distribution `q_i(z)` per data point with a shared inference network `q_\phi(z|x)`;
2. **the reparameterization trick**: get lower-variance gradients for the inference network than plain score-function / policy-gradient estimators;
3. **control as inference**: reinterpret planning and RL as probabilistic inference in a graphical model with optimality variables.

The main message is:

- lecture 11's variational inference works in principle but scales poorly;
- amortization fixes the parameter-count problem by learning an encoder;
- VAEs are the canonical generative-model example of this idea;
- the same inference viewpoint also gives a clean derivation of soft value functions and soft policies in RL.

## Part 1: Amortized variational inference

### 1. Title

The lecture topic is **variational inference in RL**. In practice this means both using VI inside latent-variable generative models and using inference ideas to derive RL algorithms.

### 2. Part 1: Amortized variational inference

This section starts from the end of lecture 11 and asks how to make variational inference practical for large datasets.

### 3. Variational inference (recap)

This slide reviews the previous lecture's setup:

```math
\log p(x_i) \ge \mathcal L_i(p,q_i)
```

with a separate approximate posterior `q_i(z)` for each data point. The recap reminds us that:

- optimizing the ELBO improves the model;
- `q_i` should approximate `p(z|x_i)`;
- but storing separate variational parameters per example is expensive.

### 4. What's the problem?

The slide makes the scaling issue explicit.

If for every `x_i` we use:

```math
q_i(z)=\mathcal N(\mu_i,\sigma_i),
```

then we have to optimize:

- model parameters `\theta`;
- plus `\mu_i, \sigma_i` for every training example.

So the parameter count becomes:

```math
|\theta| + (|\mu_i| + |\sigma_i|)\times N.
```

The proposed fix is to learn a **network** that predicts the variational distribution:

```math
q_\phi(z|x)=\mathcal N(\mu_\phi(x), \sigma_\phi(x)).
```

This is the key idea of amortization: inference becomes a learned function of `x`, not a separately optimized object for each example.

### 5. Amortized variational inference

The ELBO is now written with the encoder network:

```math
\log p(x_i) \ge
\mathbb E_{z\sim q_\phi(z|x_i)}
[\log p_\theta(x_i|z) + \log p(z)] + \mathcal H(q_\phi(z|x_i)).
```

The training loop becomes:

- sample `z ~ q_\phi(z|x_i)`;
- update decoder / generative parameters `\theta`;
- update encoder / inference parameters `\phi`.

The decoder `p_\theta(x|z)` and encoder `q_\phi(z|x)` are now both neural networks.

### 6. Amortized variational inference

This slide points at the hard part: updating `\phi`.

For `\theta`, gradients are straightforward because once `z` is sampled, the objective looks like a standard likelihood term.

For `\phi`, the parameter affects:

- the sampled latent `z`;
- the entropy term of the Gaussian;
- and therefore the expectation itself.

The slide phrases the ELBO as:

```math
J(\phi)=\mathbb E_{z\sim q_\phi(z|x_i)}[r(x_i,z)]
```

and says we could use a **policy-gradient / score-function** estimator. This is correct but noisy.

### 7. The reparameterization trick

Instead of sampling directly from:

```math
z \sim q_\phi(z|x)=\mathcal N(\mu_\phi(x), \sigma_\phi(x)),
```

rewrite sampling as:

```math
\epsilon \sim \mathcal N(0, I), \qquad
z = \mu_\phi(x) + \epsilon \sigma_\phi(x).
```

Now the randomness is in `\epsilon`, which is independent of `\phi`, and `z` is a differentiable function of `\phi`.

This lets autodiff compute:

```math
\nabla_\phi \, r(x_i, \mu_\phi(x)+\epsilon \sigma_\phi(x))
```

directly.

### 8. Another way to look at it

This slide rewrites the ELBO after reparameterization:

```math
\mathcal L_i
=
\mathbb E_{\epsilon\sim \mathcal N(0,I)}
[\log p_\theta(x_i|f_\phi(x_i,\epsilon))] - D_{\mathrm{KL}}(q_\phi(z|x_i)\|p(z))
```

where `f_\phi(x,\epsilon)=\mu_\phi(x)+\epsilon\sigma_\phi(x)`.

Important point:

- the KL term often has a closed form for Gaussians;
- the reconstruction term is now differentiable through `f_\phi`.

So the whole objective can be trained with ordinary backprop.

### 9. Reparameterization trick vs. policy gradient

This slide compares the two estimators.

Policy gradient / score-function:

- works for both discrete and continuous latent variables;
- but has high variance and usually needs many samples or small learning rates.

Reparameterization trick:

- works cleanly for continuous latents;
- is simple to implement;
- gives much lower-variance gradients.

So for Gaussian latent VAEs, reparameterization is the standard choice.

## Part 2: Generative models

### 10. Part 2: Generative models

This section shows how amortized VI appears in practice through the **variational autoencoder**.

### 11. The variational autoencoder

The slide introduces the VAE architecture:

- encoder `q_\phi(z|x)`;
- decoder `p_\theta(x|z)`;
- prior `p(z)`.

Sampling story:

```math
z \sim p(z), \qquad x \sim p_\theta(x|z).
```

Training objective:

```math
\max_{\theta,\phi}
\frac{1}{N}\sum_i
\Big(
\mathbb E_{z\sim q_\phi(z|x_i)}[\log p_\theta(x_i|z)]
-
D_{\mathrm{KL}}(q_\phi(z|x_i)\|p(z))
\Big).
```

This is exactly the amortized ELBO.

### 12. Using the variational autoencoder

The slide asks why the VAE works as a generative model.

Answer:

- the encoder is only used during training;
- the decoder and prior define the generative model;
- after training, generate by sampling `z ~ p(z)` and decoding with `p_\theta(x|z)`.

So the encoder trains the latent representation, but generation only needs `p(z)` and `p_\theta(x|z)`.

### 13. Example applications: representation learning

Here `z` is treated as a learned representation of observations. The slide suggests:

1. train the VAE on replay-buffer states;
2. run RL using `z` instead of raw state `s`.

The motivation is that latent space can provide:

- compact state features;
- smoother geometry;
- reuse of prior / offline data.

So VAEs are not only generators; they can also be representation-learning tools for RL.

### 14. Conditional models

Now the VAE is conditioned on an input `x_i` and predicts another variable `y_i`.

The objective becomes a conditional ELBO:

```math
\mathbb E_{z\sim q_\phi(z|x_i,y_i)}
[\log p_\theta(y_i|x_i,z)]
-
D_{\mathrm{KL}}(q_\phi(z|x_i,y_i)\|p(z|x_i)).
```

At test time:

- sample `z ~ p(z|x_i)` or from a simple prior;
- generate `y_i` from `p_\theta(y_i|x_i,z)`.

This is how latent-variable models handle **multi-modal predictions** such as future trajectories from a road image.

### 15. Example applications: multimodal imitation learning

The slide shows latent variables for behavior cloning / imitation learning. The point is:

- one observation may support multiple good actions or plans;
- latent `z` picks a mode;
- decoder produces the action conditioned on both state and `z`.

This avoids averaging incompatible behaviors into a bad unimodal prediction.

The slide also references "Learning Latent Plans from Play", which uses latent-variable ideas to learn reusable plan representations from broad, unlabeled behavior data.

### 16. Example applications: multimodal imitation learning

This slide gives another robotics example, emphasizing that latent variables are useful for fine-grained manipulation where behavior is highly multi-modal and temporally structured.

The key message is practical:

- latent variables let the policy represent different strategies;
- amortized inference lets us train those latent structures efficiently.

### 17. Relationship to diffusion and flow matching

This slide connects conditional VAEs to modern generative modeling.

The idea is not that diffusion equals a VAE, but that:

- both introduce latent / hidden variables and optimize objectives involving tractable surrogates;
- diffusion can be seen as a kind of **hierarchical** latent-variable model with many latent steps;
- denoising / forward-noising structure makes the model easier to optimize.

So the lecture is placing VAEs and diffusion in the same broad probabilistic-modeling family.

### 18. State space models

Here the latent variable is sequential:

- latent states `z_1, z_2, ...`;
- observations `o_1, o_2, ...`.

The slide highlights:

- a prior / dynamics model over latent states;
- a decoder from latent state to observation;
- and an inference model from observations back to latent states.

This is the sequential analogue of the VAE setup, and it previews model-based RL and latent-dynamics models.

### 19. Intermission

Section break.

## Part 3: Control as inference

### 20. Part 3: Control as inference

This section switches from latent-variable generative models to a probabilistic view of planning and RL.

### 21. Optimal Control as a Model of Human Behavior

The slide starts from the observation that inverse RL often models human behavior as approximately reward-seeking.

The classical assumption is:

- actions maximize cumulative reward;
- state transitions follow dynamics;
- behavior data is explained by some reward function.

This motivates a probabilistic model of decision-making instead of a hard argmax view of optimal control.

### 22. What if the data is not optimal?

Real behavior is:

- noisy;
- stochastic;
- not perfectly optimal.

Still, better behavior should be **more likely** than worse behavior.

So instead of assuming exact optimality, we assign probability to trajectories in proportion to how good they are.

### 23. A probabilistic graphical model of decision making

The lecture introduces binary **optimality variables** `O_t`.

The crucial definition is:

```math
p(O_t = 1 \mid s_t, a_t) \propto \exp(r(s_t,a_t)).
```

Interpretation:

- high-reward state-action pairs are more likely to be labeled "optimal";
- low-reward ones are less likely.

Now planning can be posed as inference:

- condition on `O_{1:T}=1`;
- infer which actions and states are likely under that condition.

### 24. Why is this interesting?

This inference view is useful because it:

- handles suboptimal / stochastic behavior naturally;
- connects control to general inference algorithms;
- gives an explanation for why stochastic policies can be desirable.

It also links inverse RL, planning, and probabilistic modeling under one framework.

### 25. Inference = planning

The slide states the inference problem:

compute distributions like:

```math
p(a_t \mid s_t, O_{1:T})
```

by using message passing in the graphical model. The outline is:

1. compute backward messages;
2. compute the policy from those messages;
3. compute forward messages for state marginals.

### 26. Inference = planning

This is a repeat / continuation slide emphasizing the same message-passing plan.

The conceptual point is that planning is being reduced to ordinary inference in a chain-structured probabilistic model.

### 27. Backward messages

Backward messages summarize:

- how likely future optimality is if we start from a particular state or action now.

Define:

```math
\beta_t(s_t, a_t) = p(O_{t:T} \mid s_t, a_t),
```

and similarly for `\beta_t(s_t)`.

These are analogous to backward messages in HMMs.

### 28. A closer look at the backward pass

The key recursions are:

```math
\beta_t(s_t,a_t)
=
p(O_t|s_t,a_t)\;
\mathbb E_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[\beta_{t+1}(s_{t+1})]
```

and

```math
\beta_t(s_t)=\mathbb E_{a_t\sim p(a_t|s_t)}[\beta_t(s_t,a_t)].
```

Then define soft values:

```math
V_t(s_t)=\log \beta_t(s_t), \qquad
Q_t(s_t,a_t)=\log \beta_t(s_t,a_t).
```

This yields:

```math
Q_t(s_t,a_t)=r(s_t,a_t)+\log \mathbb E[\exp(V_{t+1}(s_{t+1}))]
```

and

```math
V_t(s_t)=\log \int \exp(Q_t(s_t,a_t))\,da_t.
```

This is the **soft Bellman backup**.

### 29. Backward pass summary

This slide summarizes the preceding derivation:

- `\beta_t(s_t,a_t)` is like a soft Q-value in probability space;
- `\beta_t(s_t)` is the corresponding soft value;
- the recursion is computed from `T` back to `1`.

So dynamic programming appears as backward message passing.

### 30. The action prior

If the action prior `p(a_t|s_t)` is not uniform, it enters the soft value:

```math
V(s_t)=\log \int \exp(Q(s_t,a_t)+\log p(a_t|s_t))\,da_t.
```

Equivalently, the prior can be folded into the reward:

```math
\tilde Q(s_t,a_t)=r(s_t,a_t)+\log p(a_t|s_t)+\log \mathbb E[\exp(V(s_{t+1}))]
```

So assuming a uniform action prior is mostly without loss of generality.

### 31. Policy computation

The policy is obtained from the ratio of backward messages:

```math
\pi(a_t|s_t)
=
\frac{\beta_t(s_t,a_t)}{\beta_t(s_t)}.
```

This is exactly "choose actions in proportion to how much they support future optimality."

### 32. Policy computation with value functions

Using the log-domain definitions:

```math
\pi(a_t|s_t)
=
\exp(Q_t(s_t,a_t)-V_t(s_t))
=
\exp(A_t(s_t,a_t)).
```

So the policy is a **softmax over advantages**.

This is the control-as-inference version of soft / entropy-regularized RL.

### 33. Policy computation summary

The slide emphasizes the interpretation:

- better actions get higher probability;
- ties are broken smoothly and randomly;
- this resembles Boltzmann exploration;
- as temperature decreases, the policy approaches greedy choice.

So stochasticity is not just noise; it comes out of the probabilistic formulation.

### 34. Forward messages

Forward messages compute:

```math
\alpha_t(s_t)=p(s_t|O_{1:t-1}),
```

the probability of reaching a state under the current optimality-conditioned dynamics.

Combining forward and backward messages gives:

```math
p(s_t|O_{1:T}) \propto \alpha_t(s_t)\beta_t(s_t).
```

This is the same forward-backward structure as in HMMs.

### 35. Forward/backward message intersection

This slide gives the geometric intuition:

- forward messages say which states are reachable from the start;
- backward messages say which states can still lead to high reward;
- their intersection identifies the states that matter.

This is a clean probabilistic interpretation of planning.

### 36. Forward/backward message intersection

The second version ties the same picture back to human behavior modeling:

- reachable states from the start;
- reward-seeking states from the end / future objective;
- observed stochastic behavior can be understood as living where those two pressures overlap.

### 37. Summary

The lecture summary has three main points:

1. build a **probabilistic graphical model** for optimal control;
2. solve control by **inference** in that model;
3. the resulting algorithm looks very similar to dynamic programming and value iteration, except it is **soft** rather than hard-max.

## Main takeaways

- Amortized variational inference replaces per-example variational parameters with a shared encoder `q_\phi(z|x)`.
- The reparameterization trick makes encoder gradients low-variance for continuous latents like Gaussians.
- A VAE is the standard generative-model instance of amortized variational inference.
- Latent-variable models are useful in RL for representation learning, multimodal prediction, imitation learning, and state-space modeling.
- In control as inference, introduce optimality variables with:

```math
p(O_t=1|s_t,a_t)\propto \exp(r(s_t,a_t)).
```

- Backward message passing yields soft Bellman backups:

```math
Q(s,a)=r(s,a)+\log \mathbb E[\exp(V(s'))], \qquad
V(s)=\log \int \exp(Q(s,a))\,da.
```

- The induced policy is:

```math
\pi(a|s)=\exp(Q(s,a)-V(s)),
```

which is a softmax over advantage.

## One-sentence summary

This lecture shows that amortized inference and the reparameterization trick make latent-variable models trainable at scale, and that a closely related probabilistic inference view turns optimal control into soft dynamic programming.
