---
title: "Transformer Weights "
date: 2026-01-16
excerpt: "Interpreting transformer weights as a dynamical system."
# series: "transformer-spin"      # Must match across all posts in series
# series_order: 1
permalink: /posts/2026/transformer-analysis/
tags:
  - transformers
  - statistical-physics
published: true
draft: true
---
<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-bottom: 20px;">
<strong>⚠️ Work in Progress</strong> - This post is still being developed and may contain incomplete sections or change substantially.
</div>

## Introduction
I've been interested in applying a physics framework to transformers

You can learn more about that here. Link to series landing page

In that approach we view each attention head as as kind of dynamical system of interacting tokens. In particular in the self attention mechanism 
$$
\vec{Q}_{i}^{T} \cdot \vec{K_j} = \vec{x_i}^{T} W_{Q}^{T}W_{K}\vec{x_{j}}
$$
which looks like the energy for the interaction of two vectors (spins) via a matrix
$$
W_{QK} = W_{Q}^{T}W_{K}
$$

At initialization time, each attention head has the elements of $W_{Q}$ and $W_{K}$ drawn from some initial distribution (usually $N(0,\sigma_0)$ ). The distribution for $W_{QK}$ is not set directly but rather follows from the distribution of the underlying weights. 

In the typical transformer architecture, since $W_{Q}$ and $W_{K}$ are $d_{\text{head}} \times d_{\text{model}}$ matrices (most often with $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$). Thus their product, $W_{QK}$ is _low rank_, which is different than what is encountered in most physical systems. This means that it's _rank_, $r$, corresponding to the number of non-zero singular values, has $r \leq d_{\text{head}}$, and thus smallest $d_{\text{model}} - r$ singular values will be consistent with zero.
The effective rank of the matrix depends on the alignment of the directions associated with the singular values (singular vectors) $W_{Q}$ and $W_{K}$. These structures evolve dynamically during training and the rank of $W_{QK}$ for a trained model is _emergent_ rather than fixed by the architecture. 


`https://github.com/angerami/transformer-analysis/blob/main/notebooks/low_rank_SVD_systematics.ipynb`


Over time the distribution changes shape to build in the information learned during training. The result is that the elements of the weights matrices follow some probability distribution. Although the elements are initially chosen randomly, one expects the values they attain throughout the course of the training will be correlated through back-propagation. 

In physics, we try to associate the eigenvalues of $W_{QK}$ with with _modes_ of definite energy (corresponding to the eigenvalues) that represent particular directions in the spin space. 

The physics allows us to recast the problem as an analysis of the _phase space distribution_ of the weights $W_{QK}$. 

## Analysis
Based on the discussion so far one expects that these distributions will change both at an ensemble level (e.g. shift in the mean or change in variance) but also may develop components that are distinct from the initial distribution. Since we typically start with a normal distribution measuring deviations from normality seems to be a good way to start out.

In addition to studying the mean and variance of the $P_{W}$ distributions, I also am curious about various measures of entropy and the Kulback-Leiber Divergence using the initial distribution as a reference. Some quantities sensitive to this
$$
\text{entropy} = S = \langle \ln P \rangle = \sum P \ln P
$$
$$
D_{\text{KL}} (P||P_{0}) = \sum P \ln P/P_0
$$
$$
D_{\text{KL}} (P|N(\mu,\sigma)) = \sum P \ln P/N(\mu,\sigma)
$$
where $\mu$ and $\sigma$ could either be the empirical sample mean and standard deviation or the values obtained from a fit of the $P$ distribution near the peak (e.g. $\mu \pm n_{\sigma}^{\text{fit}}\sigma$) and less sensitive to the tails. 

As a reference, if both distributions are normal but with different $\sigma$ values:
$$
D_{\text{KL}} (N(0,\sigma)|N(0,\sigma_0)) = \ln (\sigma / \sigma_0) + \frac{\sigma^2}{\sigma_0^2} - \frac{1}{2}
$$

### Singular Values of Random Matrices of Low Rank
To develop some intuition, this section contains a few numerical examples with $d_{\text{model}} = 1000$, with the values drawn from $N(0,1)$.
This figure shows what the probability density of matrix elements and singular values a full rank matrix:  $\left[W_{QK}\right]_{ij} \sim N(0,1)$. The left panel shows the probability density of the matrix elements. The center panel shows the singular values, $\lambda$, ordered from smallest to largest, while the right panel shows the same information presented as a probability distribution $P(\lambda)$. 

**Figure 1**

The next set of figures shows the effect of varying $\sigma$, where the shape of the singular value and spectral distributions is qualitatively similar where an increase in the scale of the singular values is matched by a broader spectral distribution.

**Figure 1**

Next we consider a low-rank structure $W_{QK} = W_{Q}^{T}W_{K}$, with $$(W_{Q})_{ij}, (W_{K})_{ij} \sim \mathcal{N}(0,1)$$, using $d_{\text{head}} = d_{\text{model}} / 2 = 500$.

The distributions exhibit a sharp transition at the $d_{\text{head}}$-th singular value, with the values effectively dropping to zero. The probability distribution develops a large component clustered at ~zero singular value resulting in a _gapped spectrum_, where the distribution has support over two exclusive domains. 

To complete the systematic sweep, the next figure fixes $\sigma =1$ but varies $d_{\text{head}}$ which has the expected effect of providing a larger gap in the spectrum. Also note that even though we used the same initial $\sigma$  here for $W_{Q}$ and $W_{K}$, the resulting distribution for $W_{QK}$ has a rank-dependent effect on the width of the probability distributions.

This last figure shows a numerical example with $d_{\text{model}}=1000$, with the elements of the matrices $W_Q$ and $W_K$ drawn from $N(0,1)$. The left plot shows the probability density for the matrix elements of $W_{QK}$ divided by $\sqrt{d_{\text{model}}}$. For full rank ($n_{\text{heads}}= 1 ,\,d_{\text{head}} = d_{\text{model}}$ ), this is $N(0,1)$, but as we increase the number of heads, and thus reduce the rank of $W_{QK}$, we get $N(0, \sigma)$ with $\sigma = \sqrt{d_{\text{head}}/ d_{\text{model}}} = 1/\sqrt{n_{\text{heads}}}$. Similarly, the rank affects the entropy, $S = \langle \ln P \rangle = \frac{1}{2} \ln (2\pi e/n)$. 

Usually the singular values, $\lambda$, are sorted from largest to smallest and plotted as a function of their ordered index as is shown in the middle panel. This distribution shows a sharp cutoff $d_{\text{head}}$, indicating that the remaining $d_{\text{model}} - d_{\text{head}}$ singular values are effectively zero, and thus shows the low-rank structure of $W_{QK}$. 

This information is recast a the probability density $P(\lambda)$, which is shown on the right. For $d_{\text{head}} \approx d_{\text{model}}$. This distribution has a concentration at $\lambda \sim 0$, with the height proportional to  $d_{\text{model}} - d_{\text{head}}$, and a second, continuous contribution separated by a _gap_. As the matrix approaches full rank, this gap begins to fill in, fusing into a single contribution resembling a Marchenko-Pasteur distribution. This distribution has support for $\lambda_{\pm} = \sqrt{d_{\text{head}}} \times (1 \pm \sqrt{d_{\text{model}} / d_{\text{head}}})$.  
In this case since we are dealing with the _product_ of two random low-rank matrices, the curves will only loosely match this expectation.


### Data Samples
The data for this study was the model parameters themselves. I used models available on huggingface.

**table and description of models**

To study the time dependence I used the `Pythia` collection of datasets. This collection is specifically designed for these types of studies and contains models of different sizes, and crucially, the weights at multiple "times", different stages of the training, represented by multiple checkpoints

**Insert table of pythia data**

**further description of pythia arch** 

## Results

** Link to streamlit space for full exporation**
### Time-independent study



### Time-dependent study







