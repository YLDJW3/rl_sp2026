# Lecture 15: Model-Based RL

Source: `lectures/lec-15.pdf`

## Big picture

This lecture introduces the basic motivation for **model-based reinforcement learning**:

1. learn a predictive model of the environment, like a learned simulator;
2. use that model for planning or policy learning;
3. deal with the main failure mode, which is **distributional shift**;
4. make model usage safer by estimating **uncertainty**.

The main message is:

- learning a model is not enough by itself;
- planning inside a learned model can exploit model errors;
- the hardest problem is often not fitting the model, but deciding how much to trust it;
- uncertainty estimation is a key tool for making model-based RL practical.

## Slide-by-slide note

### 1. Title

The lecture topic is **model-based RL**. The rest of the slides explain what it means to learn a model and why uncertainty matters.

### 2. Part 1: Can we learn a “simulator”?

This section asks the most direct model-based RL question: can we learn a world model that is good enough to plan in?

### 3. Can we learn a “simulator”?

The slide uses modern video-generation systems as motivation. The point is not that these systems are RL models, but that large generative models make the idea of a learned simulator feel plausible.

### 4. Notional “learned simulator” algorithm

The abstract recipe is:

1. learn a model of the environment;
2. then either run model-free RL inside that model, or use planning / trajectory optimization in the model.

The lecture immediately asks the right question: why might this fail?

### 5. Why might it fail?

The first issue is **distributional shift**.

The policy we optimize inside the learned model determines what states the model will be queried on. If those states are different from the training data, the model can make large errors exactly where the policy most wants to exploit them.

So model-based RL has the same familiar pattern:

- train on one distribution;
- deploy on another;
- fail because the model is wrong where it matters.

### 6. Why might it fail?

This slide adds a second point: model design is **domain dependent**.

Some domains are much easier to simulate than others. Predicting low-dimensional physics is easier than predicting raw pixels, contact-rich interaction, or long-horizon visual futures. So whether a learned simulator works depends heavily on what the environment looks like and what output space the model must predict.

### 7. More reasons the notional algorithm can fail

Even if the model is accurate, planning might still be hard:

- large neural models can be expensive;
- planning or RL over image observations is difficult;
- running model-free RL inside the learned simulator can still be sample- and compute-intensive;
- sometimes a simpler but less accurate model is preferable because it makes planning easier.

This is an important engineering point: the best predictive model is not always the best model for control.

### 8. Three kinds of problems

The slide separates the model-based RL challenge into three pieces:

- statistics / algorithms: can we estimate the model well?
- deep learning / models: can we build a sufficiently expressive predictor?
- controls / RL: can we actually make decisions using it?

The lecture’s framing is that the control part is often the least glamorous but most difficult.

### 9. Part 2: Distributional shift and uncertainty

Now the lecture turns from “can we fit a model?” to “how do we avoid using the model in the wrong places?”

### 10. Distributional shift in model-based RL

The slide shows the classic temptation: if the model predicts slightly higher reward in an unfamiliar region, the planner wants to go there.

This can partially improve behavior, but it can also just exploit model error.

### 11. How far should we go?

If the goal is fast learning, it is tempting to push the policy as far as possible toward seemingly high-reward regions.

But this creates a tension:

- aggressive policy improvement explores promising states quickly;
- aggressive policy improvement also pushes the policy into areas where the model is least reliable.

### 12. The problem is pretty bad

This slide emphasizes that the exploitation issue is not a small correction. A planner can aggressively seek out unrealistic transitions because the learned model accidentally makes them look excellent.

### 13. Two ways to address distributional shift

The slide gives two broad remedies:

1. **don’t change the policy too much**
2. **stay where the model is confident**

Concretely, that means:

- trust-region style updates, so the policy does not move too far from the data distribution;
- uncertainty-aware models plus pessimistic penalties, so uncertain predictions are treated as less valuable.

This is the central practical principle of model-based RL.

### 14. How can uncertainty estimation help?

The slide’s intuition is that high predictive variance should reduce how attractive a state-action sequence looks.

Even if two plans have the same predicted mean reward, the one with much larger uncertainty should often be treated as worse if we want conservative decision making.

So uncertainty is not only about confidence intervals. It directly changes the optimization objective.

### 15. Need to explore to get better

The lecture then makes an important distinction:

- pessimistic value is not the same as expected value;
- optimistic value is not the same as expected value.

Expected value is often a useful starting point, but exploration and safe model use require reasoning about uncertainty more explicitly.

This slide is setting up the next question: what kind of uncertainty do we need?

## Part 3: Uncertainty-aware neural networks

### 16. Part 3: Uncertainty-aware neural networks

This section asks how to make neural network models output something more informative than a point prediction.

### 17. Why is output entropy not enough?

The slide distinguishes two types of uncertainty:

- **aleatoric uncertainty**: randomness in the data itself;
- **epistemic uncertainty**: uncertainty about the model parameters because data is limited.

A single probabilistic output distribution only captures aleatoric uncertainty well. It does not tell us whether the model itself is unsure due to lack of data.

That is exactly why output entropy alone is insufficient for model-based RL.

### 18. Estimate model uncertainty

The desired object is a **distribution over models**, not just a distribution over outputs from one model.

If many plausible models disagree, then the prediction is epistemically uncertain. The entropy or spread induced by that disagreement is what helps detect out-of-distribution regions.

### 19. Bayesian neural networks

The slide gives a quick overview of Bayesian neural networks.

The idea is to put a distribution over weights and reason about uncertainty in the parameters. This is principled, but often hard to implement and optimize well in large-scale deep learning.

The lecture mentions this mostly as conceptual background rather than the default practical tool.

### 20. Bootstrap ensembles

A simpler practical idea is to train multiple models and compare their predictions.

If the models agree, the system is more confident.
If they disagree, the system is uncertain.

The statistical motivation is bootstrap resampling: different data subsets should produce slightly different fitted models.

### 21. Bootstrap ensembles in deep learning

The lecture’s practical take is:

- ensembles work reasonably well;
- they are a crude posterior approximation because we usually use only a few models;
- exact bootstrap resampling is often unnecessary in deep learning because random initialization and SGD already create diversity.

This makes ensembles one of the most common practical tools for uncertainty-aware model-based RL.

### 22. Next time

The closing slide previews the next step: once we have uncertainty-aware models, how should we actually plan or learn policies with them?

## Main takeaways

- Model-based RL is appealing because a learned model can be reused for planning and policy improvement.
- The biggest practical problem is distributional shift: the controller can exploit model mistakes.
- A good predictive model is not automatically a good control model.
- Uncertainty estimation helps identify where the model should not be trusted.
- The most important uncertainty for this purpose is often epistemic uncertainty.
- Bayesian neural networks are principled, but ensembles are the simpler and more common practical approximation.

## One-sentence summary

Lecture 15 says that model-based RL is only useful if we can control **where** the learned model is trusted, and uncertainty estimation is the main tool for doing that.
