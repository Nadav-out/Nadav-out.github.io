---
layout: post
title:  "Transformers like a physicist"
date:   2024-04-29
tags:   derivation transformers 
math:   true
# date:   2023-06-07 10:44:59 -0700
categories: ML LLM 
---

When I started learning about transformers, I couldn't find a discussion of the transformer architecture that was easy for me to follow. I assume it is mainly because of my background in physics, and the way I'm used to reading/learning about new mathematical concepts. So, I decided to try and write the mathematical intro to transformers I wish I had when I started learning about them. I assume I'm about to fail, but let's see how it goes.

## Notation
I'm going to use somewhat unorthodox notations in this post. This is probably a bad idea, but I just can't stand the use of sub/superscripts when naming matrices, specifically the matrices $W^Q,\,W^K,\,W^V$ in the self-attention mechanism. 

<!-- list of variables, dimensions, and indicies -->
#### Variables
To differentiate between learnable and non-learnable variables, I will use a tilde to denote learnable variables. Only 'fundamental' learnable variables will be denoted with a tilde. For example, the embedding matrix will be denoted by $\tilde{E}$, while the embedded sequence will be denoted by $X=\tilde{E}x$.
- $x\in \mathbb{R}^{T\times V}$: A sequence of $T$ tokens, $x=\\{x_1,x_2,...,x_T\\}^T$. Each token is a one-hot vector (a.k.a. unit vector) in $\mathbb{R}^V$, where $V$ is the size of the vocabulary.
- $\tilde{E}\in \mathbb{R}^{V\times D}$: The embedding matrix. $D$ is the embedding dimension. Each token in the sequence is embedded into a $D$-dimensional vector given by $X_i= x_i\tilde{E}$.
- $X\in \mathbb{R}^{T\times D}$: The embedded sequence, $X=x\tilde{E}$. Each row of $X$ is the embedding of a token in the sequence. Namely, $X=\\{X_1,X_2,...,X_{T}\\}^T$.
- $\tilde{Q}$, $\tilde{K}$, $\tilde{V}\in\mathbb{R}^{D\times d}$: The learnable linear transformations for the query, key, and value, respectively. $d=D/H$ where $H$ is the number of heads in the self-attention mechanism, which is chosen such that $d$ is an integer.
- $Q$, $K$, $V\in \mathbb{R}^{T\times d}$: The query, key, and value for each token in the sequence, $Q=X\tilde{Q}$, $K=X\tilde{K}$, $V=X\tilde{V}$.
- $A\in \mathbb{R}^{T\times T}$: The attention weights, $A=\text{softmax}_{\text{rows}}(QK^T/\sqrt{d})$.
- $\tilde{O}\in \mathbb{R}^{D\times D}$: A learnable linear transformation that maps the output of the self-attention mechanism to the output of the layer. 

#### Indices
- $t$, $t'$: Index for the tokens in the sequence, $t=\\{1,2,...,T\\}$.
- $\epsilon$,$\epsilon'$: Index for the embedding dimension, $\epsilon=\\{1,2,...,D\\}$.
- $a$, $a'$: Index for the attention head dimension, $a=\\{1,2,...,d\\}$.
- $h$: Index for the attention head, $h=\\{1,2,...,H\\}$.

For all indices, but $h$, we will use the Einstein summation convention. Namely, we will sum over repeated indices. For example, $X^{t,\epsilon}X_{t'}^{\epsilon}=\sum_{\epsilon}X^{t,\epsilon}X_{t'}^{\epsilon}$.

## Birds eye view
The transformer architecture is a neural network architecture. It takes a text as input and outputs a prediction for the next token in the sequence. The input passed through a series of layers, each of which is composed of two main components: the self-attention mechanism and the feed-forward neural network. We will focus on a decoder-only transformer, which is the architecture used in the GPT models.

#### Data and Embedding
The first step in the transformer is to take a text and convert it into a sequence of tokens. The tokenization of the text is independent form the transformer architecture, and is done using a predefined tokenizer.  Our vocabulary has a total of $V$ tokens and the text is tokenized into a sequence of $T$ such tokens.  Each token in the sequence is represented by a one-hot vector (unit vector) in $\mathbb{R}^V$. The first layer of the transformer is an embedding layer, which maps each one-hot vector to a $D$-dimensional vector (usually $D\ll V$). Our post-emebdding sequence is then a tensor $X\in \mathbb{R}^{T\times D}$.

__Note on batching__: Mainly for computational reasons, the transformer processes the sequence in batches. Meaning, we process $B$ sequences at a time, where $B$ is the batch size. The tensor $X$ is then of size $B\times T\times D$. Since no operation in the transformer mixes the different sequences in the batch, we can think of the tensor $X$ as a collection of $B$ sequences, each of length $T$ and dimension $D$. We will therefore quite generally suppress the batch index $B$ in our notation. The fact that we use batches is very important for the training of the model, but not so much for understanding the architecture itself.

#### The transformer layers

After the embedding layer, the data, $X$, is passed through a series of $L$ layers. Each layer is composed of two main components: self-attention block and the feed-forward block. The feed-forward part is a simple two-layer neural network, with a ReLU activation function:

$$

    \text{FFN}(x)=\text{ReLU}(xW_1+b_1)W_2+b_2.

$$

Another important component of the transformer is the positional encoding, which is added to the embedded sequence before it is passed to the first layer. I'm not going to discuss the positional encoding in this post, but it is a very important part of the transformer architecture.

So the only thing left to discuss is the self-attention mechanism. We will dive into the math of the self-attention mechanism in the next section, but here is a high-level overview. The self-attention mechanism looks at the input sequence and decides how much 'attention' to give to each token in the sequence. It has three main components: the query, $Q$, the key, $K$, and the value, $V$. The query, key, and value are all learnable linear transformations of the input sequence, $X$. 

## Self-attention mechanism
So, let's dive into the math of the self-attention mechanism. The self-attention mechanism is a function that takes the embedded input sequence, $X$, and returns a new sequence, $Y$, of the same shape. In the transformer, it is made up of $H$ attention heads, making it a multi-head self-attention mechanism. Each head takes the input sequence, $X$, and returns a new sequence, $y\in \mathbb{R}^{T\times d}$, where $d=D/H$, and $D$ and $H$ are chosen such that $d$ is an integer.  Each head is a non-linear projection of the input sequence, $X$, based on three learnable linear transformations: $\tilde{Q}_h$, $\tilde{K}_h$, $\tilde{V}_h \in \mathbb{R}^{D\times d}$. (Usually, in the literature, the learnable linear transformations are denoted by $W^Q_h$, $W^K_h$, and $W^V_h$, but I used a different notation to reserve super/subscripts for indexing.) We use these matrices to get the query, key, and value for each head

$$

    Q_h^{t,a}=X^{t,\epsilon}\tilde{Q}_h^{\epsilon,a},\quad K_h^{t,a}=X^{t,\epsilon}\tilde{K}_h^{\epsilon,a},\quad V_h^{t,a}=X^{t,\epsilon}\tilde{V}_h^{\epsilon,a},

$$

Recall, $\epsilon=\\{1,2,...,D\\}$, $a=\\{1,2,...,d\\}$, and $t=\\{1,2,...,T\\}$. Using the query and key, we can calculate the attention weights, $A_h$, for each head

<div id="eq_softmax">
$$
    A_h^{t,t'}=\frac{\exp(Q_h^{t,a}K_h^{t',a}/\sqrt{d})}{\sum_{t'}\exp(Q_h^{t,a}K_h^{t',a}/\sqrt{d})}\quad (\text{not summing over }h). \tag{1}
$$
</div>

Note the normalized nature of the attention weights, $\sum_{t'}A_h^{t,t'}=1$, for each $t$. This means that when dotted with the value, $V_h$, the attention weights give a weighted average of the values. The output of the head is then

$$

    y_h^{t,a}=A_h^{t,t'}V_h^{t',a} = A_h^{t,t'}X^{t',\epsilon}\tilde{V}_h^{\epsilon,a}.

$$

To get the output of the self-attention layer, we concatenate the outputs of each head and pass them through a learnable linear transformation, $\tilde{O}\in \mathbb{R}^{D\times D}$

$$
    
        Y^{t,\epsilon}=\tilde{O}^{\epsilon,\epsilon'}Z^{t,\epsilon'},\quad Z^{t,\epsilon}=\left[y_1^{t,a_1(\epsilon)},y_2^{t,a_2(\epsilon)},...,y_H^{t,a_H(\epsilon)}\right].

$$

This notation is probably very confusing, let's break it down. The 'tensor' $Z^{t,\epsilon}$ is a concatenation of the outputs of each head, it lives in $\mathbb{R}^{T\times D}$. The $t$ index is simple to understand, it is the token index. As for the embedding index, $\epsilon$, this should be read as follows; for $\epsilon=1,...,d$, we have $Z^{t,\epsilon}=y_1^{t,\epsilon}$, for $\epsilon'=d+1,...,2d$, we have $Z^{t,\epsilon}=y_2^{t,\epsilon-d}$, and so on. The learnable linear transformation, $\tilde{O}$, is then used to map the concatenated outputs of the heads to the output of the self-attention mechanism.

<!-- Note that the concatenation can be achieved by multiplying each $y_h$ by a block-diagonal matrix, $B_h\in \mathbb{R}^{D\times d}$, with $H-1$ blocks of zeros and the identity matrix in the remaining block. The output of the self-attention mechanism is then

$$

    \boxed{Y^{t,\alpha}=\sum_h \tilde{O}^{\alpha,\beta}B_h^{\beta,a}y_h^{t,a}}= \sum_h A_h^{t,t'}\left[\tilde{O}^{\alpha,\beta}B_h^{\beta,a}\tilde{V}_h^{\gamma,a}\right]X^{t',\gamma} := \left[\sum_h A_h^{t,t'}O_h^{\alpha,\beta}\right]X^{t',\beta}.

$$

The last step we did, where we defined $O^{\alpha,\beta}=\tilde{O}_h^{\alpha,\gamma}B_h^{\gamma,a}\tilde{V}_h^{\beta,a}$, is just a redefinition of the learnable linear transformation, but it is not a standard practice in the literature. Let's count the number of parameters using both notations. The standard notation has $D^2$ parameters in $\tilde{O}$, and $H\times d\times D$ in $\tilde{V}$. Meaning, a total of $2D^2$ parameters in $\tilde{O}^{\alpha,\gamma}B_h^{\gamma,a}\tilde{V}_h^{a,\beta}$. On the other hand, $O_h^{\alpha,\beta}$ can have $H\times D^2$ parameters, which is potentially very redundant.  -->

#### Residual connection and layer normalization

The output of the self-attention mechanism is then passed through a residual connection, namely, the output of the self-attention mechanism is $Y+X$. The output is then passed through a layer normalization, 

$$

    \text{Out}^{t,\alpha}=\text{LayerNorm}(Y^{t,\alpha}+X^{t,\alpha})=\frac{Y^{t,\alpha}+X^{t,\alpha}-\mu_t}{\sigma_t},

$$

with

$$

    \mu_t=\frac{1}{D}\sum_{\alpha}\left[Y^{t,\alpha}+X^{t,\alpha}\right],\quad \sigma_t=\sqrt{\frac{1}{D}\sum_{\alpha}(Y^{t,\alpha}+X^{t,\alpha}-\mu_t)^2}.    

$$

#### Masked self-attention

In the context of language modeling, we want to preserve a notion of causality. Meaning, we want to predict the next token in the sequence using only the tokens that came before it. To achieve this, we use a mask out past information before calculating the attention weights. This is done by adding $-\infty$ in the upper triangular part (without the diagonal)  of $Q_h^{t,a}K_h^{t',a}$ before the softmax operation. This is equivalent to

 


#### The $\sqrt{d}$ factor in the softmax
The justifications for the $\sqrt{d}$ factor in the softmax [Eq.[1]](#eq_softmax) goes as follows. At fidxed $t$ and $h$, the argument of the softmax is a dot product of two $d$-dimensional vectors. Assuming each element of these vectors is drown from a distribution with variance $\sigma^2$, and zero mean, the dot product will have a variance of $d\sigma^2$. The factor of $1/\sqrt{d}$ will cancel the scaling with $d$, potentially making the softmax more stable, especially when $d$ is large.

<!-- We see that in the masked self attentions, this argument is somewhat broken, as some elements of the dot product are set to zero. However, in practice, the $\sqrt{d}$ factor is still used, and seems to work well. Maybe it is worth investigating whether this could be improved. -->

