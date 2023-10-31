---
layout: post
title:  "Generalization Toy Models"
date:   2023-10-30
tags:   Theory ML regularization Lasso Ridge
math:   true
# date:   2023-07-26 10:44:59 -0700
categories: ML
---
As part of my attempts at better [understanding Lasso regularization]({% link _posts/2023-07-26-lasso.markdown %}) (more on that to come soon!), I've been thinking more broadly about regularization in ML recently. One thing I was able to understand slightly better, is the ability of certain regularization schemes to reduce overfitting and increase the ability of a model to generalize. In this blog post, I'm going to discuss those findings, mainly in the context of a very simple teacher-student model.

#### Logic and setup

There are two main points I want to make in this blog post:
1. Given a set of training data, it is possible that some direction in model parameters space will have no effect the model loss. This is generally true when the number of data examples is smaller than its dimension. We will refer to such directions as 'flat-directions'. Adding a regularization term tend to 'lift' those flat directions, usually in a way that acts to set such 'flat-parameters' to zero. This effect tends to improve the model ability to generalize, and reduces the dependence on initialization. 
2. An over-parameterized network usually exhibit many flat-directions. In such cases lasso (or any $L_p$ with $p<1$) regularization tend to result in very sparse network, which generally improve generalization. 

To demonstrate the two points above, I will focus on a very simple teacher-student model, which will grow in complexity as the discussion evolves. I will assume my data is a set of $N_{\rm tr}$ training examples, each of which is a vector $\boldsymbol{X}$ of length $D$, such that $N_{\rm tr}\lesssim D$. I will further assume that each vector $\boldsymbol{X}$ is drawn from a standard normal distribution. The design matrix $X\in\mathbb{R}^{D\times N_{\rm tr}}$ is therefore essentially a collection of ${N_{\rm tr}\times D}$ iid standard normal random variables. The Gram matrix for this data set is given by $\Sigma_{\rm tr}=XX^T/N_{\rm tr}$, and its population mean is simply given by a $D\times D$ identity matrix.

## Linear models
The first two examples we are going to study are strictly linear.  $\boldsymbol{S},\boldsymbol{T}\in\mathbb{R}^D$ are the student and teacher vectors respectively, and the difference between them is denoted by $\boldsymbol{D}=\boldsymbol{S}-\boldsymbol{T}$. The MSE loss and its gradient are given by

$$
    \ell_{\rm tr}=\frac{1}{2}\boldsymbol{D}^T\Sigma_{\rm tr}\boldsymbol{D}\;\;\Rightarrow\;\;\frac{\partial \ell_{\rm tr}}{\partial \boldsymbol{S}}=\frac{\partial \ell_{\rm tr}}{\partial \boldsymbol{D}}=\Sigma_{\rm tr}\boldsymbol{D}.
$$

Since the average over all data of $\Sigma_{\rm tr}$ is $\left\langle\Sigma_{\rm tr}\right\rangle=I$, when we test how well this model generalizes, we will assume the population mean of the above loss, $\ell_{\rm gen}=D^TD/2$. For the time being, we care mainly about the steady state of the system, the dynamics will be studied with some more details later on. Under this assumption, we can justify working in the gradient flow limit. 


### Unregularized case

In this case the weights evolve according to 

$$
    \dot{\boldsymbol{D}}=\dot{\boldsymbol{S}}=-\Sigma_{\rm tr}\boldsymbol{D}\;\;,\;\;\boldsymbol{S}(0)=\boldsymbol{S}_0.
$$

This solution to this equation can be expressed formally with matrix exponentiation, however, we find it more instructive to move to the (orthonormal) eigen-basis of $\Sigma_{\rm tr}$

$$
    \Sigma_{\rm tr} \boldsymbol{v}_i=\lambda_i\boldsymbol{v}_i\;\;,\;\;\boldsymbol{v}_i^T\boldsymbol{v}_j=\delta_{ij},
$$

where $\delta_{ij}$ is the Kronecker delta. Writing $S$, $T$, and $D$ in this basis one finds

$$
    S_{i}(t)=T_i+(S_{i,0}-T_i)e^{-\lambda_i t}.
$$

The generalization loss as a function of time in that case is simply given by

$$
    \ell_{\rm gen}=\frac{1}{2}\sum_{i=1}^D(S_{i,0}-T_i)^2e^{-2\lambda_i t}.
$$

We recall that $\Sigma_{\rm tr}$ is the Gram matrix of $N_{\rm tr}$ samples of $D$-vectors. Its eigenvalues are therefore non-negative. Moreover, since we focus on $N_{\rm tr}<D$, it has at least $D-N_{\rm tr}$ vanishing eigenvalues. As $t\to\infty$, the contribution of all eigenvectors associated with positive eigenvalues to the loss will decay exponentially. Vectors in the null-space of $\Sigma_{\rm tr}$ will not evolve, this is the space of 'flat-directions'. The population average (average over all possible data-sets) of the generalization loss above can be evaluated analytically in some limiting cases using the Marchenko-Pastur theorem. However, since our data is drowned from an isotropic distribution, each direction in $\mathbb{R}^D$ will have the same probability to be associated with a $0$ eigenvalue. We also know that on average there will be exactly $D-N_{\rm tr}$ vanishing eigenvalues. The population mean of the generalization loss is therefore given by

$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert S_0-T\Vert_2^2.
$$

It is common to initialize the student parameters $S_0$ by selecting it from a zero-mean distribution. In which case the mean expected loss will be given by

<div id="eq_free_loss">
$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\left(\left\langle\Vert S_0\Vert_2^2\right\rangle+\Vert T\Vert_2^2\right). \tag{1}
$$
</div>


### Ridge regularization
In the ridge case, we add the regularization term $\lambda\Vert S\Vert_2^2$ to the loss $\ell_{\rm tr}$ above, working again in the eigen basis of $\Sigma_{\rm tr}$, the gradient flow equation is now given by

$$
    \dot{S}=-(\lambda_i+\lambda)S_i+\lambda_iT_i,
$$

which is solved by

$$
    S_i(t)=\left(S_{i,0}-\frac{\lambda_i}{\lambda+\lambda_i} T_i\right )e^{-(\lambda+\lambda)t}+\frac{\lambda_i}{\lambda+\lambda_i}T_i.
$$

At the $t\to\infty$ limit, the generalization loss will be given by

$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle=\frac{1}{2}\Vert T\Vert_2^2\left\langle \frac{\lambda^2}{(\lambda+\lambda_i)^2} \right\rangle.
$$

This expression can, too, be analytically estimated using the Marchenko-Pastur eigenvalues distribution. However, if we assume that $\lambda$ is small enough, we see, again, that only the $\lambda_i=0$ subspace contributes to the loss, resulting with the mean estimated loss of

$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert T\Vert_2^2+{\cal O}(\lambda^2).
$$

This loss is generally lower than the one in unregularized case as given in [Eq.[1]](#eq_free_loss). This is a demonstration of the first point we wanted to discuss, and gives a simplistic understanding as to how regularization can improve the ability of a model to generalize.

**A note on convergence:** the convergence time to the final loss values described above will depend on the eigenvalues, $\lambda_i$, or $\lambda_i+\lambda$ in the ridge case. In the more practical case of discrete time steps with learning rate $\eta$, the components of $\boldsymbol{S}$ in the eigen-basis of $\Sigma_{\rm tr}$  evolve as

$$
    S_i^{(t)}=\left[1-\eta(\lambda_i+\lambda)\right]^t\left(S_{0,i}-\frac{\lambda_i}{\lambda_i+\lambda}T_i\right)+\frac{\lambda_i}{\lambda_i+\lambda}T_i,
$$

where in the un-regularized case we simply set $\lambda=0$. We follow the discussion in [Why Momentum Really Works](https://distill.pub/2017/momentum/), and conclude that the optimal convergence rate will depend on either the smallest or largest eigenvalue, 

$$
    {\rm rate}_0=\min_\eta\,\max[\vert 1-\eta\lambda_{\rm min}\vert,\vert 1-\eta\lambda_{\rm max}\vert]\;\;,\;\;{\rm rate}_\lambda=\min_\eta\,\max[\vert 1-\eta\lambda\vert,\vert 1-\eta(\lambda+\lambda_{\rm max})\vert].
$$

The optimal rate is found to be

$$
    {\rm rate}_0=\frac{\lambda_{\rm max}/\lambda_{\rm min}-1}{\lambda_{\rm max}/\lambda_{\rm min}+1}\;\;,\;\;{\rm rate}_{\lambda}=\frac{\lambda_{\rm max}/\lambda}{\lambda_{\rm max}/\lambda+2}
$$

In the limit of $N_{\rm tr},D\to\infty$, ad fixed $N_{\rm tr}/D$ ratio, the smallest anlargest eigenvalue od $\Sigma_{\rm tr}$ are given as part of the Marchenko-Pastur theorem by $\lambda_{\rm max,min}=(1\pm\sqrt{N_{\rm tr}/D})^2$. With this in mind, assuming $\eta$ is set to its optimal value, the ridge rate will be faster given that $\lambda$ is chosen such as
$$
    \lambda>\frac{\lambda_{\rm max}\lambda_{\rm min}}{\lambda_{\rm max}-\lambda_{\rm min}}=\frac{(1-N_{\rm tr}/D)^2}{4\sqrt{N_{\rm tr}/D}}
$$

We note here without derivation that the generalization loss is a strictly increasing function of $\lambda$, which motivates using the smallest possible value of $\lambda$. Using the lower bound on $\lambda$ above, which leads to same convergence time as the un-regularized case, we get that the expected generalization loss is bounded from above by

$$
    \ell_{\rm gen}\leq\frac{3}{32}\left(4+\sqrt{2}\right)\left(1-\frac{N_{\rm tr}}{D}\right)\Vert T\Vert_2^2
$$

As a closing remark, including momentum would not change the lower bound on $\lambda$ and upper bound on $\ell_{\rm gen}$ stated above, however the optimal rate can in principle be much faster and given by ${\rm rate}=\sqrt{N_{\rm tr}/D}$, which is up to $50\%$ faster.