---
title: "Energy Landscapes and Attention: The Statistical Physics Lens"
date: 2025-10-31
excerpt: "Transformers as dynamical systems in the thermodynamic limit."
series: "transformer-spin"      # Must match across all posts in series
series_order: 2 
permalink: /posts/2025/energy-landscape/
tags:
  - transformers
  - statistical-physics
published: true
draft: true
show_ile_notice: true
---

<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-bottom: 20px;">
<strong>⚠️ Work in Progress</strong> - This post is still being developed and may contain incomplete sections or change substantially.
</div>

### Statistical Physics

 These exponentially weighted factors call to mind the formalism of statistical physics, in which the probability that a system be found in a particular "state" with "energy," $E$, is given by a "Boltzmann Factor":

$$
    \text{Prob}(E) \propto e^{-\beta E}\,
$$

where $\beta$ is a parameter that sets the scale for the energy. A high energy state has an energy that is much larger than $1/\beta$, and the probability for such states is exponentially suppressed. In most applications this variable plays the role of an inverse temperature, i.e. $\beta = 1/k_{B} T$, where $k_{B}$ is the Boltzmann Constant, which basically gives temperature the same dimension as energy. The constant relating the Boltzmann Factor to the probability is fixed by the normalization requirement for the probabilities:

$$
    \text{Prob}(E) = \frac{e^{-\beta E}}{Z(\beta)}\,,\quad Z(\beta)\equiv \sum_i e^{-\beta E_i}
$$

where the sum is taken over all energy states of the system, or if the energies are continuous, it becomes an integral. Sometimes this is written as $Z = \tr \,e^{-\beta \hat{H}}$, with the trace operation the appropriate generalization to handle the case when considering the energy as an operator. 

You may have encountered the Maxwell-Boltzmann distribution, which is the probability distribution for the velocities of classical particles in a non-relativistic gas, with kinetic energy $E=m v^2/2$. The Boltzmann factor is the probability of finding a particle with velocity $v$ in a gas is

$$
P(v) \propto e^{-\frac{1}{2}m v^2/2T}\,.
$$

If we think of attention weights as representing Boltzmann factors, then we can identify the states as $x_i^{a}$ with $H_{ij} = -\beta E$. We can interpret the values stored in this tensor as *degrees of freedom*. Specifically, each element of the input sequence, labeled by $i$, has its own $\dmodel$-dimensional "spin" vector, which interacts with the spin vectors of other elements of the input sequence. Each pair of spins contributes $H_{ij}$ to the total energy of the system. The interaction energies result in a force exerted among the two elements. In physics this would be called a spin model, or a spin chain. Although the term "spin" is often associated with quantum phenomena, none of the baggage associated with that is necessary here. A spin vector is a feature vector, but we give a physical prescription for the energy of interaction between two such features. 

### Hopfield Networks: the Physics is the Hash Function

In a traditional mechanics problem, the forces lead to differential equations, via Newton's Laws, which can be solved such that the motion of each degree of freedom is determined as function of time, e.g. $x_i^{a}(t)$, given some initial conditions, e.g. $x_{i\,0}^{a} \equiv x_i^{a}(0)$.

In complicated systems, an exact solution is generally not possible. One method of solution is to start with $x_{i\,0}^{a}$ and use the differential equation to approximate $x$ after some small, but finite amount of time. This is the approach employed by Hopfield Networks. These are recurrent neural networks where the effect of the recurrent layers is to update the values in time using the equations of motion of a spin model. In these types of spin models, the solutions converge to fixed-point attractors. This is a fancy way of saying that the whole arrangement is dynamically stable (values don't fly off to infinity), there are configurations of the system that do not evolve with time, and the system eventually ends up in one such state (an attractor). Moreover, initial conditions that are "nearby" all end up to the same attractor solution. The domain of $x_{i\,0}^{a}$ values converging to a given attractor is its \emph{basin of attraction}. 

An important property of the interactions following from the form of $H_{ij}$ is that they may be engineered, by choosing values of $J$, to have a huge number of such attractors. If we think about the numerical values specifying an attractor configuration as _data_, then the data can be accessed by providing any $x$ value in the attractor's basin of attraction as input to the Hopfield Network, the network will evolve the configuration in time until it reaches the attractor, which is the output by the network. In this sense, the whole thing forms a very dense associative memory structure. You can think about this like using a hash table to implement an associative array, with the dynamics of the spin model defining the hash function. If you could physically build such a spin system, the hash function wouldn't be a function at all but rather the empirical result obtained by observing the system. The hash function is the physics. A Hopfield Network uses recurrent layers to numerically calculate the output, but if you could build a physical system that implements the same dynamics, you could just observe the output.

### Back to Boltzmann

While transformers are not precisely Hopfield Networks, it's clear that the dynamics of such spin models underlies whatever is going on in the transformer.  Now let's make this connection more concrete. Noticeably, the transformer has no explicit time structure (recurrent connections) and has this curious exponential factor that we want to identify as a Boltzmann factor. In the case of Hopfield Networks,the recurrent elements make the whole thing expensive to compute. 

When there are a large number of degrees of freedom, it makes sense to flip the problem around and ask, for a given value of $x$, how many of the degrees of freedom have that value of $x$. In a simpler system, there would be a single equilibrium solution $x$ that represents the configuration of the system after _long times_, whatever that means. If we are only interested in time averages over such long times, we may be able to replace such time averages with averages over the various configurations visited by the state as it evolves with time. This allows us to *replace time averages with ensemble averages*. Explicitly, for some quantity $\mathcal{O}$ computed from the $x$ values,

$$
    \langle \mathcal{O} \rangle_{\text{time}} \equiv \frac{1}{\Delta t}\int_0^{\Delta t} \mathcal{O}(x(t)) \text{d}t \Rightarrow 
    \int \mathcal{O}(x) P(x) \text{d}x \equiv
    \langle \mathcal{O} \rangle_{\text{ensemble}}\,.
$$

This is possible whever the system posses a property known as *ergodicity*. This means that the system is able to explore all of the available configurations, and that the time spent in each configuration is proportional to the probability of finding the system in that configuration. Usually those probablities are strongly peaked around the equilibrium configuration, with $\beta$ or equivalently $T$ controlling location and width of the distribution. This the jumping off point to study the system using thermodynamics, and it allows you to define a equation of state, which is a relationship between the macroscopic properties of the system, e.g. $PV = NkT$ for the ideal gas.

In the case of the Hopfield Network, things are not so simple, as we have many equilibria represented by the attractors. The system is not ergodic, rather than evolve to one global minimum of the energy configuration, the network can get stuck in a local minimum, a metastable configuration, for a very long time.

A more advanced concept in statistical physics, which you may not have encountered in an introductory exposure to the topic, is a modification to the Boltzmann factor (or Gibbs measure):

$$
    P(x) = \sum_i P_i e^{-\beta E_i(x)}/Z_i\,,
$$

where the sum runs over the attractor configurations; each attractor has it's own ensemble and a probability factor for finding the system in that attractor state. One the surface, this doesn't look very advanced, we are just writing something like:

$$
    P(x) = \sum_i P(i) P(x|i)\,,
$$

but the reason why we can, and must, do this in this particular situation (ergodicity breaking) is quite profound. To develop intuition here and understand the extension to transfomers, it's really important to see why it works out this way. First notice that the probability factors attain their maximum values for configurations that minimize the exponent (saddle or stationary points) and ensemble averages will be overwhelmingly dominated by these configurations. The probability for configurations that deviate are exponentially suppressed, with the parameter $\beta$ controlling the suppression. Thus we can say _the behavior of the system near the fixed-point attractors is encoded in the saddle-points of the probability distribution_. However, this isn't the whole story, as we also need to know the probability of being in a given basin of attraction. The physics of how this distribution follows from the interaction energies is kind of spectactular. Lastly, the nomenclature seems to shift at this point. When talking about the time evolution you see the term attractor used, but when discussing the statistical picture, in which one often computes averages over different attractors, the term "disorder" is typically used. 

### Intuition break: why do we use the exponential form?

The exponential form of the probability factors is not a given, but rather a consequence of the various assumptions we make about the system and the states, you may hear terms like microcanonical canonical and grand canonical ensembles. Rather than get into the axioms of ensemble theory, which is probably the part I found most off-putting when learning the subject, I'll motivate the exponential as follows. If I have two such  systems with $E_1$ and $E_2$, respectively, that do not interact with each other, then the total probability for the combined system to be in a given state is

$$
\text{Prob}(E_1,E_2)=P(E_1) P(E_2) = P(E_1+E_2)
$$

where the last equality explicitly uses the fact that the probabilities are exponential. This means that when you combine two systems, the energies add together. The energy is an _extensive_ quantity, that grows with the size of the system. This is in contrast to an _intensitve_ quantity like temperature; if you combine two systems with different temperatures, they will exchange energy until they reach a common temperature, which will be some sort of weighted average of the two, not the sum.

### How Boltzmann Factors Appear in the Transformer Equations

In the Transformer Equations, the attention weights are used to compute,

$$
z_{i}^{a} = \sum_{b} [\WV]^{ab} \sum_{j} \alpha_{ij}  x_{j}^{b}
\rightarrow \vv{z}_{i} = \WV \langle \vv{x} \rangle_{i}
$$

where I have interchanged the orders of the sums over $j$ and $b$ from the expression in the table. The quantity $\langle \vv{x} \rangle_{i}$ looks like the average value of $x$ in the statistical ensemble labeled by $i$. 

In most spin models, the pairwise interactions are restricted to only occur between nearest neighbors. Our situation is more general since the $J$ doesn't depend on $i$, it's non-local. We got rid of this upstream by doing the positional encoding.

When the $i^{\text{th}}$ element of the sequence interacts with the other elements, it acts as an external magnetic field on those elements and results in a net magnetization, $\langle \vv{x} \rangle_i$. To account for the effects of those interactions on the whole system, that net magnetization is treated like an external magnetic field acting on the $i^{\text{th}}$ element. This external field induces a new magnetization for the $i^{\text{th}}$ element, with the matrix $\WV$ playing the role of the magnetic susceptibility. Note that the concept of the external field is used twice, first to compute the effect of $i$ on the the other elements, then to use the average such effect as an external field to correct the magnetization for the $i^{\text{th}}$ element.

$$
\vv{m}_{i} = \mm{\chi}\langle \vv{x} \rangle_i \leftrightarrow 
\vv{z}_{i} = \WV \langle \vv{x} \rangle_{i}
$$

Thus we identify $z$ as a kind of first-order correction using to account for all of these complicated interactions using an approach from _mean field theory_. We express the $x_i^a$ as a mean value plus a small correction, $x_i^a = \langle x^a \rangle_i + \delta x_i^a$ and express the energy in terms of the mean value and drop second order terms in $\delta x_i^a$, which gives $H_{ij}^{\text{MF}}$ that is a function $\langle x^a \rangle_i$ and $x_j^a$. We then use these energies as probabilities to compute the $\langle x^a \rangle_i$ by averaging. This leads to a self-consistency requirement:

$$
    \langle x^a \rangle_i = \sum_j x^a_j \, \frac{e^{H_{ij}^{\text{MF}}(\langle x^a \rangle_i)}}
    {\sum_k e^{H_{ik}^{\text{MF}}(\langle x^a \rangle_i)}}
$$

Usually, this is a transcendental equation, and you have to solve it iteratively or numerically.

One feature that sees a lot of attention in textbooks is to show that the number of solutions to this equation changes depending on the relationship between the values of $J$ and $\beta$. For example, below some critical value of $\beta$, there is a single solution, but above that value, there are multiple solutions. These two different behavior correspond to two different phases of the system. This mean-field approach so long as there is a _separation of scales_ in the problem and works increasingly well when $\dmodel$ is large, the so-called large-$N$ limit of the model.



