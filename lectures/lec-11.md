# Lecture 11: Variational Inference

Source: `lectures/lec-11.pdf`

## Big picture

This lecture has two connected goals:

1. explain **latent variable models**, where we introduce an unobserved variable `z` to make complicated data distributions easier to model;
2. explain **variational inference**, where we replace the intractable posterior `p(z|x)` with a tractable approximation `q(z)` or `q_i(z)`.

The main message is:

- latent variables make modeling more expressive;
- exact maximum likelihood becomes hard because it requires integrating over `z`;
- variational inference solves this by optimizing a **lower bound** on `log p(x)`;
- but the naive version is still expensive because it introduces separate variational parameters for every training example.

## Slide-by-slide note

### 1. Title

The lecture topic is **variational inference**. The rest of the slides explain why it appears naturally when we try to train latent variable models.

### 2. Part 1: Latent variable models

This is a section divider. First the lecture motivates latent variable models before deriving variational inference.

### 3. Probabilistic models

The slide contrasts two familiar settings:

- `p(x)`: modeling data directly, like density estimation or clustering;
- `p(y|x)`: modeling outputs conditioned on inputs, like regression, classification, or policy learning.

The point is that probabilistic modeling already covers many ML problems. Latent variable models extend both views by inserting hidden structure.

### 4. Latent variable models

The core equations are:

```math
p(x) = \sum_z p(x|z)p(z), \qquad
p(y|x) = \sum_z p(y|x,z)p(z)
```

What the slide is saying:

- `z` is an unobserved variable that selects a hidden mode, component, plan, style, or intent;
- after conditioning on `z`, the distribution becomes simpler;
- marginalizing over `z` gives a richer final distribution.

The clustering picture illustrates `z` as a mixture component. The driving example suggests that a hidden variable could represent different future behaviors like turning left, going straight, or turning right. This is why latent variables are useful for **multi-modal predictions**.

### 5. Latent variable models in general

Now the lecture moves from discrete sums to continuous latent variables:

```math
p(x) = \int p(x|z)p(z)\,dz
```

The important idea is:

- choose an easy prior `p(z)` such as a Gaussian;
- choose an easy conditional `p(x|z)`, often something like a Gaussian whose mean and variance come from a neural network;
- after integrating out `z`, the resulting `p(x)` can still be very complicated and multi-modal.

So the latent variable gives us a trick for building a complicated data model out of simpler pieces.

### 6. Latent variable models in RL

The lecture gives two RL motivations.

For policies:

- sample a latent code `z ~ N(0, I)`;
- condition the policy on both state and `z`;
- different `z` values can produce different coherent behaviors.

This is a way to represent **multi-modal policies**.

For model-based RL:

- latent variable models can describe hidden structure in trajectories;
- the diagram emphasizes modeling transitions `p(s_{t+1}|s_t, a_t)` and initial states, not just observations;
- the latent space can capture structure useful for planning.

So latent variables are not only for static density models. They also help represent ambiguity and structure in sequential decision-making.

### 7. How do we train latent variable models?

If we use maximum likelihood, we would like to solve:

```math
\theta \leftarrow \arg\max_\theta \frac{1}{N}\sum_i \log p_\theta(x_i)
```

But for a latent variable model,

```math
p_\theta(x_i)=\int p_\theta(x_i|z)p(z)\,dz
```

so the objective becomes:

```math
\theta \leftarrow \arg\max_\theta \frac{1}{N}\sum_i
\log \left(\int p_\theta(x_i|z)p(z)\,dz\right)
```

The slide’s main point is that this is **intractable** in general. The integral itself is hard, and it sits inside a logarithm, which makes optimization even harder.

### 8. Estimating the log-likelihood

A natural idea is to use the posterior over latent variables:

```math
p(z|x_i)
```

and optimize an expected complete-data log-likelihood. Intuitively:

- guess which hidden `z` probably explains `x_i`;
- average over plausible `z` values rather than committing to one.

But then the obvious problem appears: how do we compute `p(z|x_i)`? The slide labels this problem **probabilistic inference**.

This is the transition point of the lecture:

- training the model requires inference;
- exact inference is the hard part.

If we know $p(z|x_i)$, then we can optmize $\mathbb E_{z \sim p(z|x_i)}[\log p_\theta(x_i, z)]$

### 9. Other places we will see probabilistic inference

This slide broadens the scope. Probabilistic or variational inference shows up in:

- RL / control with hidden human intent or behavior models;
- exploration with generative models;
- learning from human feedback.

The message is that inference is a recurring theme across modern ML, but this lecture will focus on the fundamentals rather than those applications.

### 10. Part 2: Variational inference

This is the second section divider. The lecture now introduces a practical workaround for intractable posterior inference.

### 11. The variational approximation

We introduce a tractable approximation:

```math
q_i(z) = \mathcal{N}(\mu_i, \sigma_i)
```

The slide explicitly says this is **incorrect but convenient**. That is the philosophy of variational inference:

- do not insist on computing the true posterior exactly;
- instead choose a simple family and find the member that best approximates the posterior.

The derivation rewrites:

```math
\log p(x_i)
= \log \int p(x_i|z)p(z)\frac{q_i(z)}{q_i(z)}\,dz
= \log \mathbb{E}_{z\sim q_i(z)}
\left[\frac{p(x_i|z)p(z)}{q_i(z)}\right]
```

This sets up Jensen’s inequality.

### 12. Jensen gives the lower bound

Using

```math
\log \mathbb{E}[y] \ge \mathbb{E}[\log y],
```

we get:

```math
\log p(x_i)
\ge
\mathbb{E}_{z\sim q_i(z)}
\left[
\log p(x_i|z) + \log p(z) - \log q_i(z)
\right]
```

or equivalently

```math
\log p(x_i)
\ge
\mathbb{E}_{z\sim q_i(z)}[\log p(x_i|z) + \log p(z)]
 + \mathcal{H}(q_i)
```

This is the **variational lower bound**, later usually called the **ELBO**.

The meaning of the three terms:

- `log p(x_i|z)`: fit the data well after choosing a latent;
- `log p(z)`: keep the latent plausible under the prior;
- `H(q_i)`: do not collapse the approximate posterior too aggressively.

### 13. Brief aside: entropy

The slide defines entropy:

```math
\mathcal{H}(p) = -\mathbb{E}_{x\sim p(x)}[\log p(x)]
```

Two intuitions are emphasized:

- entropy measures how random or spread out a distribution is;
- larger entropy means the distribution is broader, less concentrated, and less confident.

In the ELBO, maximizing `H(q_i)` pushes `q_i` to stay wide unless the reconstruction and prior terms justify narrowing it.

### 14. Brief aside: KL divergence

The KL divergence is defined as:

```math
D_{\mathrm{KL}}(q\|p)
= \mathbb{E}_{x\sim q}\left[\log \frac{q(x)}{p(x)}\right]
```

The slide stresses two views:

- KL measures how different two distributions are;
- algebraically, it is expected negative log-probability under `p`, minus entropy of `q`.

This matters because the entropy term in the ELBO is not arbitrary. It appears naturally when we express the approximation quality through KL divergence.

### 15. Variational approximation and KL

The lecture now shows the key identity:

```math
\log p(x_i)
= D_{\mathrm{KL}}(q_i(z)\|p(z|x_i)) + \mathcal{L}_i(p, q_i)
```

where

```math
\mathcal{L}_i(p, q_i)
=
\mathbb{E}_{z\sim q_i(z)}[\log p(x_i|z)+\log p(z)] + \mathcal{H}(q_i)
```

This explains what the ELBO is doing:

- it is a lower bound on `log p(x_i)`;
- it becomes tight exactly when `q_i(z) = p(z|x_i)`.

So a “good” variational approximation is one that minimizes KL to the true posterior.

### 16. Why maximizing the ELBO minimizes KL

This slide isolates the same identity and emphasizes one crucial fact:

- `log p(x_i)` does **not** depend on `q_i`.

Therefore, with respect to `q_i`, maximizing `L_i(p, q_i)` is exactly the same as minimizing:

```math
D_{\mathrm{KL}}(q_i(z)\|p(z|x_i))
```

That is the central justification for variational inference.

### 17. How do we use this?

Instead of directly maximizing the intractable log-likelihood, we maximize the average ELBO:

```math
\theta \leftarrow \arg\max_\theta \frac{1}{N}\sum_i \mathcal{L}_i(p, q_i)
```

The slide suggests an alternating view:

For each data point `x_i`:

- sample `z ~ q_i(z)`;
- use that sample to estimate gradients for model parameters `\theta`;
- update `q_i` so it better maximizes the bound.

If `q_i(z)` is Gaussian with parameters `\mu_i, \sigma_i`, then we can also optimize those variational parameters by gradient ascent.

So the recipe is:

1. optimize the generative model;
2. optimize the approximate posterior;
3. do both through the same lower bound.

### 18. What is the problem?

The naive method is not scalable.

If every data point `x_i` gets its own variational parameters `\mu_i, \sigma_i`, then the total parameter count is:

```math
|\theta| + (|\mu_i| + |\sigma_i|)\times N
```

That means the number of inference parameters grows linearly with dataset size. This is too expensive and awkward for large datasets, minibatching, or generalization to unseen examples.

The slide ends by saying the next lecture will explain how deep learning makes this tractable. That is a preview of **amortized inference**, where a neural network predicts `q(z|x)` instead of storing separate parameters for each example.

## Main takeaways

- Latent variable models express complicated distributions by integrating over hidden variables.
- Exact maximum likelihood is hard because `p(x)=\int p(x|z)p(z)dz` is usually intractable.
- Variational inference introduces a simple approximate posterior `q_i(z)`.
- Jensen’s inequality gives a lower bound:

```math
\mathcal{L}_i
=
\mathbb{E}_{q_i}[\log p(x_i|z)+\log p(z)] + \mathcal{H}(q_i)
\le \log p(x_i)
```

- Maximizing the ELBO is equivalent to minimizing `D_{\mathrm{KL}}(q_i(z)\|p(z|x_i))`.
- The naive version is still expensive because it uses separate variational parameters for every training example.

## One-sentence summary

The lecture shows that once latent variables make maximum-likelihood training intractable, variational inference rescues the problem by replacing exact posterior inference with optimization of a tractable lower bound, but the non-amortized version does not scale.
