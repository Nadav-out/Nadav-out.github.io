---
layout: post
title:  "Generalization Toy Models - II"
date:   2023-11-2
tags:   theory regularization lasso ridge generalization sparsity
math:   true
categories: ML
---

A second post about generalization toy models. In this post I aim to show how sparsity inducing regularization schemes can often times lead to better generalization.


#### Logic and challenges
[The first post]({% link _posts/2023-10-30-generalization.markdown %}) focused on purely analytical results in linear models and demonstrated how and why ridge regularization can improve the ability of a model to generalize. In this post, I will focus on slightly more complex networks, demonstrating how sparsity inducing norms (specifically $L_1$ norm) can result in better generalization abilities. 

The main idea is that often times when we train a model all the model weights converge to some finite values. However, the loss space might exhibit a continuum of global minima, along which the loss is constant. This suggests that our model is over-parametrized, albeit probably in some very non-trivial way. By sending some of the network's weights to 0, sparsity inducing regularization acts against over parametrization. I don't yet posses a satisfying argument as for why sparsity inducing norms result in better generalization, however, a 'hand-waving' argument might be that a sparser network is 'simpler' in the information-theoretical sense, and therefore tend to generalize better. It will be interesting to try and understand this better from the leans of _singular learning theory_ [(see this great blog post by Jesse Hoogland)](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick).


In all the following examples I will compare the validation loss obtained in the unregulated case, to the one obtained with $L_2$ and $L_1$ regularization. The examples we are going to study are 
1. __Sparse linear teacher-student model:__ This basically models what happens if we use irrelevant features in under-determaind linear regression.
2. __Over-parametrized two layers teacher-student model:__ Here we basically demonstrate what happens when we use an over-complicated for an underlying linear model.
3. __Over-parametrized non-linear teacher-student model:__ This is the most important example, as it shows how sparsity inducing norms dynamically lead to simpler networks that generalize better.

The first step however, will be to introduce how are we going to implement $L_1$ norm in PyTorch. A notebook for reproducing all our results can be found on [GitHub](https://github.com/Nadav-out/Blog_related/blob/main/Generalization_toys_II.ipynb).

## Gradient decent with $L_1$ regularization.
The $L_1$ norm (the absolute value function) is singular around 0, this is known to lead to somewhat unstable gradient decent. To bypass this, we will make use of proximal gradient methods, as described in this [Wikipedia page](https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning). Without getting in to too many details, it can be shown that if $\boldsymbol{w}^*$ is the weight vector that minimizes $\ell(\boldsymbol{w})=\ell_0(\boldsymbol{w})+\lambda\Vert \boldsymbol{w}\Vert_1$, it is also the solution to the equation

$$
    \boldsymbol{w}^*=S_{\lambda\eta}\left[\boldsymbol{w}^*-\eta\boldsymbol{\nabla}_{\boldsymbol{w}^*}\ell_0(\boldsymbol{w}^*)\right],
$$

for any value of $\eta$, and where $S_{\lambda}$ is the _soft thresholding operator_

$$
    S_\lambda(x)={\rm sign}(x){\rm ReLU}({\rm abs}[x]-\lambda).
$$

To incorporate it into any PyTorch optimizer, we can simply add a soft thresholding step after the gradient update step, namely, we identify $\eta$ above as our learning rate, and update like so

$$
    \boldsymbol{w}\leftarrow \boldsymbol{w}-\eta\boldsymbol{\nabla}_{\boldsymbol{w}}\ell_0(\boldsymbol{w})
$$

$$
    \boldsymbol{w}\leftarrow S_{\eta \lambda}(\boldsymbol{w})
$$

In PyTorch, using `torch.optim.SGD` as an example, this is done like so:

```python
class SGD_L1(torch.optim.SGD):
    def __init__(self, params, l1_lambda=0.01, *args, **kwargs):
        super(SGD_L1, self).__init__(params, *args, **kwargs)
        self.l1_lambda = l1_lambda

    @torch.no_grad()
    def step(self, closure=None):
        # standard SGD step
        loss = super(SGD_L1, self).step(closure)  

        # Soft thresholding
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                p.data = torch.sign(p.data) * torch.clamp(torch.abs(p.data) - self.l1_lambda * lr, min=0)

        return loss

```

For the case of SGD, rewriting the whole optimizer would be rather simple, however the nice thing about the above method is that it can be applied to whichever optimizer we want.

## Toy Examples
### First example: sparse linear regression.
The setup here is almost identical to the one used in the [previous post]({% link _posts/2023-10-30-generalization.markdown %}). 
We have a set of $N_{\rm tr}$ training samples, each is vector $\boldsymbol{X}$ in ${\mathbb{R}}^D$, where $N_{\rm tr}\lesssim D$. $\boldsymbol{X}$ is drawn from a standard normal distribution, so that design matrix $X\in\mathbb{R}^{D\times N_{\rm tr}}$ is a matrix of ${N_{\rm tr}\times D}$ iid standard normal random variables.  $\boldsymbol{S},\boldsymbol{T}\in\mathbb{R}^D$ are the student and teacher vectors respectively. The Gram matrix for our data set is given by $\Sigma_{\rm tr}=XX^T/N_{\rm tr}$, and its population mean is simply given by a $D\times D$ identity matrix. We use assume an MSE loss function $(\boldsymbol{S}-\boldsymbol{T})^T\Sigma_{\rm tr}(\boldsymbol{S}-\boldsymbol{T})/2$ for training, and test for generalization by taking the population mean of the loss, $\Vert \boldsymbol{S}-\boldsymbol{T} \Vert_2^2/2$.

We will use normalized teacher vectors, $\Vert\boldsymbol{T} \Vert_2^2=1$ and train the student vector, first by choosing a random unit teacher vector in $\mathbb{R}^D$, and then by choosing a unit vector in $\mathbb{R}^{d}\sub{\mathbb{R}}^D$, and padding it by $D-d$ zeros. The second case is similar to assuming that on $d$ out of the $D$ features we are using actually effect our loss. We train the student using three different optimizers; unregularized, ridge, and lasso. Since the unregularized loss and ridge loss are invariant under orthogonal transformation of the student vector ("Spherical symmetry"), we don't expect to see any difference in the ability of the model to generalize regardless of the subspace from which $\boldsymbol{T}$ is selected. We therefore expect that the generalization loss in those cases will be given by the results in the previous post:

$$
    \ell_{\mathrm{gen},0}\simeq\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{S}_0-\boldsymbol{T}\Vert_2^2\;\;,\;\;\ell_{\mathrm{gen},L_2}\simeq\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{T}\Vert_2^2+{\cal O}(\lambda^2)
$$

The $L_1$ norm, however, does depend on the basis of $\mathbb{R}^D$ we work in. For that reason, together with the fact that it induces sparsity, we expect it to outperform in the sparse case. 

Below we show a plot demonstrating precisely these observations; the generalization in the unregularized and ridge case practically aren't affected by the direction of $\boldsymbol{T}$. The lasso case on the other hand preforms much better than others for sparse teacher. For a random teacher however, ridge outperforms lasso.  The parameters used for these plots are $D=1000,\;N_{\rm tr}=700$, and in the sparse case we set $d=400$. For the hyperparameters (learning rate and weight decay) in the unregularized and ridge cases we used the optimal parameters as described in the previous post. For the lasso case we used $\eta=0.4$ and $\lambda=5\times10^{-3}$, without trying too hard to optimize.


![Desktop View](/assets/images/gen_linear_sparse.png){: width="972" height="589" }
_Average loss curves vs epochs, for a random teacher vector and a $70\%$ sparse vector. We see that while for a random teacher Ridge outperforms Lasso, Lasso generalizes much better in the case of Sparse network._


