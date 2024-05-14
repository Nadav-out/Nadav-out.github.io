---
layout: post
title:  "Generalization Toy Models - II"
date:   2023-11-2
tags:   theory regularization lasso ridge generalization sparsity
math:   true
categories: ML
---

A second post about generalization toy models. In this post I aim to show how sparsity inducing regularization schemes can often times lead to better generalization.

#### Skatching ideas
Looking at the softmax part of the self-attention mechanism in transformers:

$$
    \text{softmax}(QK^T/\sqrt{d_k})
$$

We write the $Q$ and $K$ matrices using the linear transformation $Q=XW^Q$ and $K=XW^K$, namely

$$

    Q K^T=XW^Q{W^K}^TX^T.

$$

The dimension of $W^Q$ and $W^K$ is $D\times d_k$, where $D$ is the dimension of the input vectors, and $d_k$ is the dimension of the query and key vectors. Let's express those matrices as a list of $d_k$ $D$-dimensional vectors, $W^Q=\\{\boldsymbol{q}_1,\boldsymbol{q}_2,\ldots,\boldsymbol{q}_{d_k}\\}$, and $W^K=\\{\boldsymbol{k}_1,\boldsymbol{k}_2,\ldots,\boldsymbol{k}_{d_k}\\}$. The $W^Q(W^K)^T$ matrix can then be written as

$$
    M=W^Q(W^K)^T=\sum_{i=1}^{d_k}\boldsymbol{q}_i\boldsymbol{k}_i^T.
$$

Each of the terms in the sum above is a rank-1 $D\times D$ matrix. Let us further write each of the $d_k$ vectors as a magnitude times a unit vector, $\boldsymbol{q}_i=q_i\hat{q}_i$, and $\boldsymbol{k}_i=k_i\hat{k}_i$. The matrix $M$ can then be written as

$$
    M=\sum_{i=1}^{d_k}q_ik_i\hat{q}_i\hat{k}_i^T:=\sum_{i=1}^{d_k}r_i M_i\;\;:\;\;r_i=q_ik_i\;\;,M_i=\hat{q}_i\hat{k}_i^T.
$$

Now, we recall that the network is usually trained with an L2 regularization term on the weights. In therms of the above decomposition, this regularization is given by

$$
    L_{\rm WD}=\frac{\lambda}{2}\Vert W^Q\Vert_2^2+\frac{\lambda}{2}\Vert W^K\Vert_2^2=\frac{\lambda}{2}\sum_{i=1}^{d_k}\left(q_i^2+k_i^2\right).
$$

setting w.l.o.g. $q_i=r_i/k_i$, we see that the regularization term can be written as

$$
    L_{\rm WD}=\frac{\lambda}{2}\sum_{i=1}^{d_k}\left(\frac{r_i^2}{k_i^2}+k_i^2\right).
$$

This is the only part depending on $k_i$, and it is minimized when $k_i=\sqrt{r_i}$. Plugging this back into the loss, we get

$$
    L_{\rm WD}=\lambda\sum_{i=1}^{d_k}r_i
$$


### Simple example. 
Let's consider a simple example of finding a low rank matrix $M=VU^T$, where $V$ and $U$ are D-dimensional vectors. But we try approximate it with a sum of 2 pairs of D-dimensional vectors, $m=v_1u_1^T+v_2u_2^T$. We assume the objective function is given by the Frobenius norm of the difference between the two matrices, $L=\Vert M-m\Vert_F^2$. We can write the loss as

$$
    L=\Vert VU^T-v_1u_1^T-v_2u_2^T\Vert_F^2={\rm Tr}\left[(VU^T-v_1u_1^T-v_2u_2^T)^T(VU^T-v_1u_1^T-v_2u_2^T)\right].

$$

Expanding the above expression, we can write the loss as a sum of products of two vectors, 

$$
    L=\Vert V\Vert^2\Vert U\Vert^2+\Vert v_1\Vert^2\Vert u_1\Vert^2+\Vert v_2\Vert^2\Vert u_2\Vert^2-2(V\cdot v_1)(U\cdot u_1)-2(V\cdot v_2)(U\cdot u_2)+2(u_1\cdot u_2)(v_1\cdot v_2).

$$

Expressing the vectors in terms of their magnitudes and unit vectors, $V=v\hat{v}$, $U=u\hat{u}$, $v_1=r_1\hat{v}_1$, $u_1=s_1\hat{u}_1$, $v_2=r_2\hat{v}_2$, $u_2=s_2\hat{u}_2$, we can write the loss as  

$$
    L=v^2u^2+r_1^2s_1^2+r_2^2s_2^2-2vu r_1s_1(\hat{v}\cdot\hat{v}_1)(\hat{u}\cdot\hat{u}_1)-2vu r_2s_2(\hat{v}\cdot\hat{v}_2)(\hat{u}\cdot\hat{u}_2)+2s_1s_2r_1r_2(\hat{u}_1\cdot\hat{u}_2)(\hat{v}_1\cdot\hat{v}_2).
$$

Now, we define $r_1s_1=x u v$, $r_2s_2=y u v$, and write the loss as

$$
    L=v^2u^2\left[1+x^2+y^2-2x(\hat{v}\cdot\hat{v}_1)(\hat{u}\cdot\hat{u}_1)-2y(\hat{v}\cdot\hat{v}_2)(\hat{u}\cdot\hat{u}_2)+2x y(\hat{u}_1\cdot\hat{u}_2)(\hat{v}_1\cdot\hat{v}_2)\right].
$$

Now, assume we regularize the loss with an L2 norm on the vectors, $L_{\rm WD}=(\lambda/2)(\Vert v_1\Vert^2+\Vert v_2\Vert^2+\Vert u_1\Vert^2+\Vert u_2\Vert^2)$. We can write this as $L_{\rm WD}=(\lambda/2)(r_1^2+r_2^2+s_1^2+s_2^2)$. Expressing $s_1^2=x^2 u^2v^2/r_1^2$, $s_2^2=y^2 u^2v^2/r_2^2$, we can write the regularized loss as

$$
    L_{\rm WD}=\lambda(r_1^2+r_2^2+x^2 u^2v^2/r_1^2+y^2 u^2v^2/r_2^2).

$$

This is the only place where $r_1$ and $r_2$ appear, so we can directly minimize it to get $r_1^2=u v x$, $r_2^2=u v y$. Plugging this back into the loss, we get

$$
    L=\lambda u v(x+y).

$$

We see that this is basically and $L_1$ regularization term for $x$ and $y$ (Recall $x,y>0$). For cleaner notation, we define $\lambda u v=\Lambda$, the full (original+regularization) loss is then given by

$$
    L=1+x^2+y^2-2x(\hat{v}\cdot\hat{v}_1)(\hat{u}\cdot\hat{u}_1)-2y(\hat{v}\cdot\hat{v}_2)(\hat{u}\cdot\hat{u}_2)+2xy(\hat{u}_1\cdot\hat{u}_2)(\hat{v}_1\cdot\hat{v}_2)+\Lambda(x+y).

$$

In the case of $D=2$, we can further simplify this by defining the angles between the vectors, $\cos\theta_i=\hat{v}\cdot\hat{v}_i$, $\cos\phi_i=\hat{u}\cdot\hat{u}_i$, and write the loss as

$$
    L=1+x^2+y^2-2x\cos\theta_1\cos\phi_1-2y\cos\theta_2\cos\phi_2+2xy\cos(\theta_1-\theta_2)\cos(\phi_1-\phi_2)+\Lambda(x+y).

$$

Nicely enough, this is independent of the dimensions of the vectors. The minimum w.r.t. the angles is simply when all angles are 0, giving

$$
    L=1+x^2+y^2-2(x+y)+\Lambda(x+y).

$$


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
    \ell_{{\rm gen},0}\simeq\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{S}_0-\boldsymbol{T}\Vert_2^2\;\;,\;\;\ell_{{\rm gen},L_2}\simeq\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{T}\Vert_2^2+{\cal O}(\lambda^2)
$$

The $L_1$ norm, however, does depend on the basis of $\mathbb{R}^D$ we work in. For that reason, together with the fact that it induces sparsity, we expect it to outperform in the sparse case. 

Below we show a plot demonstrating precisely these observations; the generalization in the unregularized and ridge case practically aren't affected by the direction of $\boldsymbol{T}$. The lasso case on the other hand preforms much better than others for sparse teacher. For a random teacher however, ridge outperforms lasso.  The parameters used for these plots are $D=1000,\;N_{\rm tr}=700$, and in the sparse case we set $d=400$. For the hyperparameters (learning rate and weight decay) in the unregularized and ridge cases we used the optimal parameters as described in the previous post. For the lasso case we used $\eta=0.4$ and $\lambda=5\times10^{-3}$, without trying too hard to optimize.


![Desktop View](/assets/images/gen_linear_sparse.png){: width="972" height="589" }
_Average loss curves vs epochs, for a random teacher vector and a $70\%$ sparse vector. We see that while for a random teacher Ridge outperforms Lasso, Lasso generalizes much better in the case of Sparse network._


In this post, I will focus on a very simple teacher-student model. I will assume my data is a set of $N_{\rm tr}$ training examples, each of which is a vector $\boldsymbol{X}$ of length $D$, such that $N_{\rm tr}\lesssim D$. I will further assume that each vector $\boldsymbol{X}$ is drawn from a standard normal distribution, the design matrix $X\in\mathbb{R}^{D\times N_{\rm tr}}$ is therefore essentially a collection of ${N_{\rm tr}\times D}$ iid standard normal random variables. The Gram matrix for this data set is given by $\Sigma_{\rm tr}=XX^T/N_{\rm tr}$, and its population mean is simply given by a $D\times D$ identity matrix.
The first two examples we are going to study are strictly linear.  $\boldsymbol{S},\boldsymbol{T}\in\mathbb{R}^D$ are the student and teacher vectors respectively, and the difference between them is denoted by $\boldsymbol{D}=\boldsymbol{S}-\boldsymbol{T}$. The MSE loss and its gradient are given by

$$
    \ell_{\rm tr}=\frac{1}{2}\boldsymbol{D}^T\Sigma_{\rm tr}\boldsymbol{D}\;\;\Rightarrow\;\;\frac{\partial \ell_{\rm tr}}{\partial \boldsymbol{S}}=\frac{\partial \ell_{\rm tr}}{\partial \boldsymbol{D}}=\Sigma_{\rm tr}\boldsymbol{D}.
$$

Since the average over all possible data sets of $\Sigma_{\rm tr}$ is $\left\langle\Sigma_{\rm tr}\right\rangle_X=I$, when we test how well this model generalizes, we will assume the population mean of the above loss, $\ell_{\rm gen}=\boldsymbol{D}^T\boldsymbol{D}/2$. For the time being, we care mainly about the steady state of the system, the dynamics will be studied with some more details towards the end of this post. For that reason, we will currently work in the gradient flow limit. 


### Unregularized case

In this case the weights evolve according to 

$$
    \dot{\boldsymbol{D}}=\dot{\boldsymbol{S}}=-\Sigma_{\rm tr}\boldsymbol{D}\;\;,\;\;\boldsymbol{S}(0)=\boldsymbol{S}_0.
$$

This solution to this equation can be expressed formally with matrix exponentiation, however, we find it more instructive to move to the (orthonormal) eigen-basis of $\Sigma_{\rm tr}$

$$
    \Sigma_{\rm tr} \boldsymbol{v}_i=\lambda_i\boldsymbol{v}_i\;\;,\;\;\boldsymbol{v}_i^T\boldsymbol{v}_j=\delta_{ij},
$$

where $\lambda_i$ are the eigenvalues of $\Sigma_{\rm tr}$ and $\delta_{ij}$ is the Kronecker delta. Writing $S$, $T$, and $D$ in this basis, we get that each component evolves independently of the others, giving

$$
    S_{i}(t)=T_i+(S_{i,0}-T_i)e^{-\lambda_i t}.
$$

The generalization loss as a function of time in that case is thus simply given by

$$
    \ell_{\rm gen}=\frac{1}{2}\sum_{i=1}^D(S_{i,0}-T_i)^2e^{-2\lambda_i t}.
$$

We recall that $\Sigma_{\rm tr}$ is the Gram matrix of $N_{\rm tr}$ samples of $D$-vectors. Its eigenvalues are therefore non-negative. Moreover, since we focus on $N_{\rm tr}<D$, it has at least $D-N_{\rm tr}$ vanishing eigenvalues. As $t\to\infty$, the contribution of all eigenvectors associated with positive eigenvalues to the loss will decay exponentially. Vectors in the null-space of $\Sigma_{\rm tr}$ will not evolve, this is the space of 'flat-directions'. The population average (average over all possible data-sets) of the generalization loss above can be evaluated analytically in some limiting cases using the Marchenko-Pastur theorem. However, since our data is drowned from an isotropic distribution, each direction in $\mathbb{R}^D$ will have the same probability to be associated with a $0$ eigenvalue. We also know that on average there will be exactly $D-N_{\rm tr}$ vanishing eigenvalues. The population mean of the generalization loss is therefore given by

$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle_{X}=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{S}_0-\boldsymbol{T}\Vert_2^2.
$$

It is common to initialize the student weights $\boldsymbol{S}_0$ by selecting it from a zero-mean distribution. In which case the mean expected loss will be given by

<div id="eq_free_loss">
$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle_{X,S_0}=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\left(\left\langle\Vert \boldsymbol{S}_0\Vert_2^2\right\rangle_{S_0}+\Vert \boldsymbol{T}\Vert_2^2\right). \tag{1}
$$
</div>

We see explicitly the sensitivity to weights initialization.


### Ridge regularization
In the ridge case, we add the regularization term $\lambda\Vert \boldsymbol{S}\Vert_2^2$ to the loss $\ell_{\rm tr}$, the components of $S$ eigenbasis of $\Sigma_{\rm tr}$ evolve according to the following gradient flow

$$
    \dot{S}=-(\lambda_i+\lambda)S_i+\lambda_iT_i,
$$

which is solved by

$$
    S_i(t)=\left(S_{i,0}-\frac{\lambda_i}{\lambda+\lambda_i} T_i\right )e^{-(\lambda+\lambda)t}+\frac{\lambda_i}{\lambda+\lambda_i}T_i.
$$

At the $t\to\infty$ limit, the generalization loss will be given by

<div id="eq_lambda_loss_full">
$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle=\frac{1}{2}\Vert \boldsymbol{T}\Vert_2^2\left\langle \frac{\lambda^2}{(\lambda+\lambda_i)^2} \right\rangle. \tag{2}
$$
</div>

This expression can, too, be analytically estimated using the Marchenko-Pastur eigenvalues distribution. However, if we assume that $\lambda$ is small enough, we see, again, that only the $\lambda_i=0$ subspace contributes to the loss, resulting with a mean estimated loss of

<div id="eq_lambda_loss">
$$
    \lim_{t\to\infty}\left\langle\ell_{\rm gen}\right\rangle_X=\frac{1}{2}\left(1-\frac{N_{\rm tr}}{D}\right)\Vert \boldsymbol{T}\Vert_2^2+{\cal O}(\lambda^2). \tag{3}
$$
</div>

This loss is generally lower than the one in unregularized case as given in [Eq.[1]](#eq_free_loss), and independent of $S_0$. This is a demonstration of the first point we wanted to discuss, and gives a simplistic understanding as to how regularization can improve the ability of a model to generalize.


![Desktop View](/assets/images/gen_linear.png){: width="972" height="589" }
_Average loss curves vs epochs. The plot was made for the case of $D=1000$, $N_{\rm tr}=700$, and we averaged over 100 randomally generated data sets ($X$ and $\Sigma_{\rm tr}$). The selection of hyperparameters is discussed below. We see that the finite step size learning match the predictions of [Eq.[1]](#eq_free_loss) and [Eq.[3]](#eq_lambda_loss), where we included the finite $\lambda$ correction to [Eq.[3]](#eq_lambda_loss)._

## Numerics
To verify the above derivations, I've implemented our simple model in PyTorch. The notebook can be found on [GitHub](https://github.com/Nadav-out/Blog_related/blob/main/Generalization_toys_I.ipynb). 

The numerical implementation obviously uses finite learning rates and step sizes, which I selected based on the wonderful discussion about [Why Momentum Really Works](https://distill.pub/2017/momentum/) in [Distill.pub](https://distill.pub/). The convergence rate of the training loss, sensitive to the largest and smallest (non-vanishing) eigenvalues of the Hessian, which in our case is either $\Sigma_{\rm tr}$ or $\Sigma_{\rm tr}+\lambda I$ in the ridge case. The vanishing eigenvalues are associated with flat directions that don't learn, and therefore don't affect the convergence rate. Following the Distill article, the optimal learning rate in our case will be given by

$$
    \eta_{0}=\frac{2}{\lambda_{+}+\lambda_{-}}\;\;,\;\;\eta_{\lambda}=\frac{2}{\lambda_{+}+2\lambda},
$$

where $\eta_0\,, \eta_{\lambda}$ are the learning rate in the unregualted and ridge cases respectively, and $\lambda_{\pm}$ are the largest and smallest eigenvalues of $\Sigma_{\rm tr}$. The corresponding optimal convergence rates are given by

$$
    R_0=\frac{\lambda_{\rm max}/\lambda_{\rm min}-1}{\lambda_{\rm max}/\lambda_{\rm min}+1}\;\;,\;\;R_{\lambda}=\frac{\lambda_{\rm max}/\lambda}{\lambda_{\rm max}/\lambda+2}\;.
$$

As seen from [Eq.[2]](#eq_lambda_loss_full), the generalization loss is an increasing function of $\lambda$, which motivates using the smallest $\lambda$ possible. The convergence rate $R_\lambda$ above, however, shows that learning in the $\lambda\to 0$ limit will be very slow. To somewhat balance the two, we will choose $\lambda$ such that the learning rates are equal, $R_0=R_\lambda$, giving

$$
    \lambda_R=\frac{\lambda_+\lambda_-}{\lambda_+-\lambda_-}
$$

In the limit of $D\to\infty$ ad fixed $q=N_{\rm tr}/D$, $\lambda_{\pm}$ are given as part of the Marchenko-Pastur theorem by $\lambda_{\pm}=(1\pm\sqrt{D/N_{\rm tr}})^2$, the learning rates and $\lambda_R$ are then given by

$$
    \lambda_R=\frac{(1-q)^2}{4q^{3/2}}\;\;;\;\;\eta_0=\frac{q}{1+q}\;\;,\;\;\eta_{\lambda}=\frac{4q^{3/2}}{(1+q)(1+\sqrt{q})^2}.
$$

In the same limit, when setting $\lambda=\lambda_R$, it can be shown that the correction to [Eq.[3]](#eq_lambda_loss) is smaller than $3\%$ for all $q<1$. The derivation is included as a Mathematica notebook on [GitHub](https://github.com/Nadav-out/Blog_related/blob/main/Generalization_toys_I.nb).

The plot above shows the resulting generalization loss curves, for the specific choice of $D=1000$, $q=0.7$, and $\boldsymbol{S}_0$ whose elements are drawn randomly from ${\mathcal N}(0,1/\sqrt{D})$, such that $\Vert \boldsymbol{S}_0\Vert_2^2$  has expectation of 1. 

Note that the ridge case seem to converge slower, although we aimed to set $R_{\lambda}=R_0$ This is because we are plotting the generalization loss, while the rates $R$, are controlling the convergence of $\ell_{\rm tr}$.

On the same plot we also show the loss curve when optimizing also the momentum.  Including momentum Does not change $\lambda_R$ and the final value of $\ell_{\rm gen}$ stated above, however the optimal rate can in principle be much faster and given by ${\rm rate}=\sqrt{N_{\rm tr}/D}$, which is up to $50\%$ faster. The details are not so illuminating, however it is fun to note that in both the ridge and the unregulated case we consider, the optimal momentum is given by $\beta=q$, the, and learning rates are given by $\eta_0=q$ and $\eta_\lambda=(1-\sqrt{q})^2/\lambda_R$.

