# Equation Bible for Production LLMs, RAG, and AI Agents

**Author:** Hesham Haroon

> This README is a GitHub-ready mathematical reference for Large Language Models, Retrieval-Augmented Generation, and AI Agents. It is designed as a production study guide and interview reference. It prioritizes breadth. It includes foundational, common, derivative, production, and less frequently cited equations. No finite document can literally contain every equation ever used in the literature, but this covers the major equation families required to reason about modern LLM, RAG, and agent systems.

## Table of Contents
- [Notation](#notation)
- [Part I: Large Language Models](#part-i-large-language-models)
  - [1. Probabilistic Foundations](#1-probabilistic-foundations)
  - [2. Embeddings and Transformer Blocks](#2-embeddings-and-transformer-blocks)
  - [3. Attention Variants](#3-attention-variants)
  - [4. Positional Encoding and Long Context](#4-positional-encoding-and-long-context)
  - [5. Normalization](#5-normalization)
  - [6. Activations and Gated MLPs](#6-activations-and-gated-mlps)
  - [7. Training Losses](#7-training-losses)
  - [8. Optimization](#8-optimization)
  - [9. Sampling and Decoding](#9-sampling-and-decoding)
  - [10. Alignment and Preference Optimization](#10-alignment-and-preference-optimization)
  - [11. PEFT and Adapter Methods](#11-peft-and-adapter-methods)
  - [12. Quantization](#12-quantization)
  - [13. Scaling Laws](#13-scaling-laws)
  - [14. Mixture of Experts](#14-mixture-of-experts)
  - [15. Tokenization and Embedding Pretraining](#15-tokenization-and-embedding-pretraining)
  - [16. Evaluation Metrics](#16-evaluation-metrics)
  - [17. Production Inference Math](#17-production-inference-math)
  - [18. Non-Transformer Sequence Models](#18-non-transformer-sequence-models)
  - [19. Calibration and Uncertainty](#19-calibration-and-uncertainty)
- [Part II: Retrieval-Augmented Generation](#part-ii-retrieval-augmented-generation)
  - [20. Similarity and Distance Metrics](#20-similarity-and-distance-metrics)
  - [21. Sparse Retrieval](#21-sparse-retrieval)
  - [22. Dense Retrieval](#22-dense-retrieval)
  - [23. Hybrid Search and Fusion](#23-hybrid-search-and-fusion)
  - [24. ANN and Vector Index Math](#24-ann-and-vector-index-math)
  - [25. Reranking and Learning to Rank](#25-reranking-and-learning-to-rank)
  - [26. Retrieval Metrics](#26-retrieval-metrics)
  - [27. Chunking and Context Packing](#27-chunking-and-context-packing)
  - [28. Diversity and Redundancy Control](#28-diversity-and-redundancy-control)
  - [29. RAG Objectives and RAGAS-Style Metrics](#29-rag-objectives-and-ragas-style-metrics)
  - [30. Query Expansion and Feedback](#30-query-expansion-and-feedback)
  - [31. GraphRAG](#31-graphrag)
- [Part III: AI Agents](#part-iii-ai-agents)
  - [32. MDP and POMDP Foundations](#32-mdp-and-pomdp-foundations)
  - [33. Temporal-Difference and Value-Based RL](#33-temporal-difference-and-value-based-rl)
  - [34. Policy Gradient and Actor-Critic](#34-policy-gradient-and-actor-critic)
  - [35. Advantage Estimation](#35-advantage-estimation)
  - [36. Exploration and Bandits](#36-exploration-and-bandits)
  - [37. Planning and Tree Search](#37-planning-and-tree-search)
  - [38. Offline RL and Imitation](#38-offline-rl-and-imitation)
  - [39. Multi-Agent RL](#39-multi-agent-rl)
  - [40. LLM-Agent Specific Math](#40-llm-agent-specific-math)
  - [41. Reward Modeling](#41-reward-modeling)
  - [42. Memory and Context](#42-memory-and-context)
  - [43. Risk, Safety, and Constrained Agents](#43-risk-safety-and-constrained-agents)
- [Important Python/PyTorch Snippets](#important-pythonpytorch-snippets)
- [References](#references)

## Notation

- $x_{1:T}$: token sequence of length $T$.
- $V$: vocabulary size.
- $d$: model dimension.
- $d_h$: attention head dimension.
- $H$: number of attention heads.
- $L$: number of Transformer layers.
- $Q,K,V$: query, key, value matrices.
- $\theta$: model parameters.
- $\pi_\theta$: policy induced by an LLM or RL agent.
- $D$: document corpus.
- $q$: query.
- $d_i$: document or chunk.
- $s_t,a_t,r_t$: state, action, reward at time $t$.
- $\gamma$: reward discount factor.
- $\epsilon$: small numerical constant.

# Part I: Large Language Models

## 1. Probabilistic Foundations

#### Autoregressive factorization

**Equation**

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{1:t-1})
$$

**Plain-English explanation.** A language model assigns probability to a full sequence by multiplying conditional next-token probabilities.

**Symbols.** $x_t$ is token $t$; $x_{\lt t}$ are earlier tokens; $\theta$ are model parameters.

**Practical use case.** Defines causal language modeling, pretraining, scoring, perplexity, and decoding.

**Source.** Bengio et al. (2003); Mikolov et al. (2010); Radford et al. (2018).

#### Conditional generation factorization

**Equation**

$$
p_\theta(y_{1:T}\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{1:t-1})
$$

**Plain-English explanation.** The model generates an output sequence conditioned on an input prompt or source sequence.

**Symbols.** $x$ is prompt/context; $y_t$ is generated token.

**Practical use case.** Instruction following, translation, summarization, tool-call generation.

**Source.** Sutskever et al. (2014); Vaswani et al. (2017).

#### Softmax over logits

**Equation**

$$
p_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

**Plain-English explanation.** Converts raw logits into a probability distribution over vocabulary tokens.

**Symbols.** $z_i$ is logit for token $i$; $V$ is vocabulary size.

**Practical use case.** The final step before sampling or loss computation.

**Source.** Bridle (1990); Goodfellow, Bengio, Courville (2016).

#### Log-softmax

**Equation**

$$
\log p_i = z_i - \log\sum_{j=1}^{V} e^{z_j}
$$

**Plain-English explanation.** Computes log-probabilities in a numerically stable way.

**Symbols.** $z_i$ is a logit; $p_i$ is normalized probability.

**Practical use case.** Stable cross-entropy and beam search scoring.

**Source.** Goodfellow, Bengio, Courville (2016).

#### Log-sum-exp trick

**Equation**

$$
\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m},\quad m=\max_i z_i
$$

**Plain-English explanation.** Prevents overflow when exponentiating large logits.

**Symbols.** $m$ is the maximum logit.

**Practical use case.** Stable inference and training kernels.

**Source.** Numerical optimization folklore; Bishop (2006).

#### Negative log-likelihood

**Equation**

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{1:t-1})
$$

**Plain-English explanation.** Penalizes the model when it assigns low probability to the correct next token.

**Symbols.** $T$ is sequence length; $p_\theta$ is the predicted token probability.

**Practical use case.** Standard causal LM training objective.

**Source.** Bengio et al. (2003); Radford et al. (2018).

#### Cross-entropy

**Equation**

$$
H(y,p)=-\sum_{i=1}^{V} y_i\log p_i
$$

**Plain-English explanation.** Measures mismatch between the target distribution and predicted distribution.

**Symbols.** $y_i$ is target probability; $p_i$ is predicted probability.

**Practical use case.** Training classifiers and next-token predictors.

**Source.** Shannon (1948); Goodfellow, Bengio, Courville (2016).

#### Cross-entropy decomposition

**Equation**

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)
$$

**Plain-English explanation.** Cross-entropy equals irreducible target entropy plus KL divergence from target to model.

**Symbols.** $P$ is data distribution; $Q$ is model distribution.

**Practical use case.** Explains why minimizing cross-entropy minimizes KL to the data distribution.

**Source.** Cover and Thomas (1991).

#### KL divergence

**Equation**

$$
D_{\mathrm{KL}}(P\Vert Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

**Plain-English explanation.** Measures how much one probability distribution differs from another.

**Symbols.** $P,Q$ are distributions over the same support.

**Practical use case.** RLHF KL penalties, distillation, variational inference, calibration.

**Source.** Kullback and Leibler (1951).

#### Reverse KL divergence

**Equation**

$$
D_{\mathrm{KL}}(Q\Vert P)=\sum_x Q(x)\log\frac{Q(x)}{P(x)}
$$

**Plain-English explanation.** A mode-seeking divergence often behaving differently from forward KL.

**Symbols.** $P$ is reference/target distribution; $Q$ is approximating distribution.

**Practical use case.** Preference optimization, variational inference, policy regularization analysis.

**Source.** Kullback and Leibler (1951).

#### Entropy

**Equation**

$$
H(P)=-\sum_x P(x)\log P(x)
$$

**Plain-English explanation.** Measures uncertainty or randomness in a distribution.

**Symbols.** $P(x)$ is probability of event $x$.

**Practical use case.** Sampling diversity, RL exploration, calibration, active learning.

**Source.** Shannon (1948).

#### Sequence log-likelihood

**Equation**

$$
\log p_\theta(x_{1:T})=\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{1:t-1})
$$

**Plain-English explanation.** Turns a product of probabilities into a sum of token log-probabilities.

**Symbols.** $x_{1:T}$ is a token sequence.

**Practical use case.** Scoring completions, preference data, reranking generations.

**Source.** Bengio et al. (2003).

#### Length-normalized score

**Equation**

$$
S(x_{1:T})=\frac{1}{T^\alpha}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{1:t-1})
$$

**Plain-English explanation.** Normalizes sequence likelihood to reduce short-sequence bias.

**Symbols.** $\alpha$ controls strength of length penalty.

**Practical use case.** Beam search, response reranking, translation decoding.

**Source.** Wu et al. (2016).

#### Teacher forcing objective

**Equation**

$$
\mathcal{L}_{\mathrm{TF}}=-\sum_t \log p_\theta(y_t\mid y_{1:t-1}^{\star},x)
$$

**Plain-English explanation.** The model predicts each next target token using gold previous tokens during training.

**Symbols.** $y_{\lt t}^{\star}$ are ground-truth previous tokens.

**Practical use case.** Seq2seq training and instruction tuning.

**Source.** Williams and Zipser (1989); Sutskever et al. (2014).

#### Perplexity

**Equation**

$$
\mathrm{PPL}=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{1:t-1})\right)
$$

**Plain-English explanation.** Exponentiated average negative log-likelihood; lower is better.

**Symbols.** $T$ is token count.

**Practical use case.** Language model evaluation.

**Source.** Jelinek et al. (1977); Bengio et al. (2003).

#### Bits per byte

**Equation**

$$
\mathrm{BPB}= -\frac{1}{B\log 2}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{1:t-1})
$$

**Plain-English explanation.** Measures compression quality normalized by byte count.

**Symbols.** $B$ is number of bytes in the text.

**Practical use case.** Byte-level model comparison across tokenizers.

**Source.** Shannon (1948); Graves (2013).

## 2. Embeddings and Transformer Blocks

#### Token embedding lookup

**Equation**

$$
X = E[\mathrm{ids}],\quad E\in\mathbb{R}^{V\times d}
$$

**Plain-English explanation.** Maps token IDs to dense vectors.

**Symbols.** $E$ is embedding matrix; $V$ vocabulary size; $d$ hidden dimension.

**Practical use case.** First layer of almost every language model.

**Source.** Bengio et al. (2003).

#### Token plus position embedding

**Equation**

$$
h_i^{(0)} = E_{x_i} + P_i
$$

**Plain-English explanation.** Combines token identity with position information.

**Symbols.** $E_{x_i}$ token embedding; $P_i$ positional embedding.

**Practical use case.** Original Transformer and GPT-style models with learned or sinusoidal positions.

**Source.** Vaswani et al. (2017); Radford et al. (2018).

#### LM head

**Equation**

$$
z_t = h_t W_U + b_U
$$

**Plain-English explanation.** Projects hidden state to vocabulary logits.

**Symbols.** $h_t$ is final hidden state; $W_U$ is unembedding matrix.

**Practical use case.** Next-token prediction.

**Source.** Bengio et al. (2003); Radford et al. (2018).

#### Tied input-output embeddings

**Equation**

$$
W_U = E^\top
$$

**Plain-English explanation.** Reuses the token embedding matrix as the output projection.

**Symbols.** $E$ is token embedding matrix; $W_U$ unembedding matrix.

**Practical use case.** Reduces parameters and can improve language modeling.

**Source.** Press and Wolf (2017); Inan et al. (2017).

#### Pre-norm Transformer block

**Equation**

$$
\tilde{h}^{(\ell)}=h^{(\ell)}+\mathrm{Attn}(\mathrm{LN}(h^{(\ell)})),\quad h^{(\ell+1)}=\tilde{h}^{(\ell)}+\mathrm{MLP}(\mathrm{LN}(\tilde{h}^{(\ell)}))
$$

**Plain-English explanation.** Applies normalization before each sublayer and adds residual connections.

**Symbols.** $h^{(\ell)}$ is layer input; $\ell$ is layer index.

**Practical use case.** Stable training for deep decoder-only LLMs.

**Source.** Baevski and Auli (2019); Xiong et al. (2020).

#### Post-norm Transformer block

**Equation**

$$
\tilde{h}^{(\ell)}=\mathrm{LN}(h^{(\ell)}+\mathrm{Attn}(h^{(\ell)})),\quad h^{(\ell+1)}=\mathrm{LN}(\tilde{h}^{(\ell)}+\mathrm{MLP}(\tilde{h}^{(\ell)}))
$$

**Plain-English explanation.** Applies normalization after the residual addition.

**Symbols.** $h^{(\ell)}$ is layer input.

**Practical use case.** Original Transformer architecture.

**Source.** Vaswani et al. (2017).

#### Transformer feed-forward network

**Equation**

$$
\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
$$

**Plain-English explanation.** Applies a position-wise nonlinear transformation after attention.

**Symbols.** $W_1,W_2$ are learned matrices; $\phi$ is activation.

**Practical use case.** The MLP part of a Transformer block.

**Source.** Vaswani et al. (2017).

#### Dropout

**Equation**

$$
\mathrm{Dropout}(x)=\frac{m\odot x}{1-p},\quad m_i\sim\mathrm{Bernoulli}(1-p)
$$

**Plain-English explanation.** Randomly zeroes activations during training and rescales survivors.

**Symbols.** $p$ is dropout probability; $m$ is binary mask.

**Practical use case.** Regularization during pretraining/fine-tuning.

**Source.** Srivastava et al. (2014).

#### Stochastic depth

**Equation**

$$
h^{(\ell+1)}=h^{(\ell)}+b_\ell f_\ell(h^{(\ell)}),\quad b_\ell\sim\mathrm{Bernoulli}(p_\ell)
$$

**Plain-English explanation.** Randomly skips residual branches during training.

**Symbols.** $b_\ell$ controls whether layer branch is active.

**Practical use case.** Regularizes very deep networks.

**Source.** Huang et al. (2016).

#### Xavier initialization

**Equation**

$$
W_{\mathrm{ij}}\sim U\left[-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right]
$$

**Plain-English explanation.** Initializes weights so variance is controlled across layers.

**Symbols.** $n_{\text{in}},n_{\text{out}}$ are fan-in and fan-out.

**Practical use case.** Stable model initialization.

**Source.** Glorot and Bengio (2010).

#### He initialization

**Equation**

$$
W_{\mathrm{ij}}\sim \mathcal{N}\left(0,\frac{2}{n_{\mathrm{in}}}\right)
$$

**Plain-English explanation.** Initialization designed for ReLU-like activations.

**Symbols.** $n_{\text{in}}$ is fan-in.

**Practical use case.** Training deep networks with ReLU-family activations.

**Source.** He et al. (2015).

## 3. Attention Variants

#### Scaled dot-product attention

**Equation**

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Plain-English explanation.** Computes a weighted average of values using query-key similarity.

**Symbols.** $Q,K,V$ are query/key/value matrices; $d_k$ is key dimension.

**Practical use case.** Core Transformer attention.

**Source.** Vaswani et al. (2017).

#### Causal masked attention

**Equation**

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\quad M_{\mathrm{ij}}=\begin{cases}0,&j\leq i\\-\infty,&j> i\end{cases}
$$

**Plain-English explanation.** Prevents a token from attending to future tokens.

**Symbols.** $M$ is causal mask; $i,j$ are token positions.

**Practical use case.** Autoregressive decoder-only LLMs.

**Source.** Vaswani et al. (2017).

#### Multi-head attention

**Equation**

$$
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W^O,\quad \mathrm{head}_h=\mathrm{Attention}(XW_h^Q,XW_h^K,XW_h^V)
$$

**Plain-English explanation.** Runs attention in multiple learned subspaces then combines the results.

**Symbols.** $H$ heads; $W^Q,W^K,W^V,W^O$ projections.

**Practical use case.** Allows different heads to capture different patterns.

**Source.** Vaswani et al. (2017).

#### Cross-attention

**Equation**

$$
\mathrm{CrossAttn}(Q_x,K_y,V_y)=\mathrm{softmax}\left(\frac{Q_xK_y^\top}{\sqrt{d_k}}\right)V_y
$$

**Plain-English explanation.** Queries from one sequence attend to keys/values from another sequence.

**Symbols.** $x$ is decoder/query sequence; $y$ is encoder/source sequence.

**Practical use case.** Encoder-decoder translation, multimodal models, RAG context fusion.

**Source.** Bahdanau et al. (2015); Vaswani et al. (2017).

#### Multi-query attention

**Equation**

$$
\mathrm{head}_h=\mathrm{Attention}(XW_h^Q,XW^K,XW^V)
$$

**Plain-English explanation.** Each query head has its own query projection but all heads share keys and values.

**Symbols.** $W_h^Q$ is per-head; $W^K,W^V$ are shared.

**Practical use case.** Reduces KV-cache memory and decode bandwidth.

**Source.** Shazeer (2019).

#### Grouped-query attention

**Equation**

$$
\mathrm{head}_h=\mathrm{Attention}(XW_h^Q,XW_{g(h)}^K,XW_{g(h)}^V)
$$

**Plain-English explanation.** Groups query heads so multiple query heads share a key/value head.

**Symbols.** $g(h)$ maps query head $h$ to a KV group.

**Practical use case.** Middle ground between MHA quality and MQA efficiency.

**Source.** Ainslie et al. (2023).

#### Sliding-window attention

**Equation**

$$
A_{\mathrm{ij}}=0\quad\mathrm{if}\quad |i-j|> w
$$

**Plain-English explanation.** Restricts attention to a local window.

**Symbols.** $w$ is window radius.

**Practical use case.** Long-context inference with lower memory and compute.

**Source.** Beltagy et al. (2020); Child et al. (2019).

#### Sparse attention

**Equation**

$$
\mathrm{SparseAttn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M_S\right)V
$$

**Plain-English explanation.** Uses a sparsity mask so only selected query-key pairs interact.

**Symbols.** $M_S$ is zero for allowed positions and $-\infty$ otherwise.

**Practical use case.** Efficient long-sequence modeling.

**Source.** Child et al. (2019); Beltagy et al. (2020).

#### Linear attention kernelization

**Equation**

$$
\mathrm{Attn}(Q,K,V)_i=\frac{\phi(q_i)^\top\sum_j\phi(k_j)v_j^\top}{\phi(q_i)^\top\sum_j\phi(k_j)}
$$

**Plain-English explanation.** Rewrites attention using kernel features to avoid explicit quadratic attention matrix.

**Symbols.** $\phi$ is a positive feature map.

**Practical use case.** Linear-time sequence models and long-context approximations.

**Source.** Katharopoulos et al. (2020).

#### Performer FAVOR+ attention

**Equation**

$$
\exp(q^\top k)\approx \phi(q)^\top\phi(k)
$$

**Plain-English explanation.** Approximates softmax attention with random feature maps.

**Symbols.** $\phi$ is a random feature mapping.

**Practical use case.** Efficient attention approximation for long sequences.

**Source.** Choromanski et al. (2021).

#### Linformer projection

**Equation**

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{Q(EK)^\top}{\sqrt{d_k}}\right)FV
$$

**Plain-English explanation.** Projects sequence-length dimension of keys and values to lower rank.

**Symbols.** $E,F\in\mathbb{R}^{k\times n}$ are learned projections.

**Practical use case.** Low-rank approximation of attention.

**Source.** Wang et al. (2020).

#### Reformer LSH attention

**Equation**

$$
h(q)=\mathrm{argmax}([qR;-qR])
$$

**Plain-English explanation.** Hashes similar queries/keys into buckets using random projections.

**Symbols.** $R$ is random rotation/projection matrix.

**Practical use case.** Approximate attention with locality-sensitive hashing.

**Source.** Kitaev et al. (2020).

#### FlashAttention online softmax

**Equation**

$$
m_i=\max(m_i^{\mathrm{old}},m_i^{\mathrm{block}}),\quad l_i=e^{m_i^{\mathrm{old}}-m_i}l_i^{\mathrm{old}}+e^{m_i^{\mathrm{block}}-m_i}l_i^{\mathrm{block}}
$$

**Plain-English explanation.** Computes exact attention in blocks while maintaining stable softmax statistics.

**Symbols.** $m_i$ is row max; $l_i$ is row normalizer.

**Practical use case.** Memory-efficient exact attention kernels on GPU.

**Source.** Dao et al. (2022).

#### Attention output with online normalization

**Equation**

$$
o_i=\frac{e^{m_i^{\mathrm{old}}-m_i}l_i^{\mathrm{old}}o_i^{\mathrm{old}}+e^{m_i^{\mathrm{block}}-m_i}l_i^{\mathrm{block}}o_i^{\mathrm{block}}}{l_i}
$$

**Plain-English explanation.** Merges partial block outputs into the exact softmax attention output.

**Symbols.** $o_i$ output row; $l_i$ softmax denominator.

**Practical use case.** FlashAttention and tiled GPU attention.

**Source.** Dao et al. (2022).

## 4. Positional Encoding and Long Context

#### Sinusoidal positional encoding

**Equation**

$$
PE_{\mathrm{pos},2i}=\sin\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right),\quad PE_{\mathrm{pos},2i+1}=\cos\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

**Plain-English explanation.** Encodes absolute position using sinusoids at different frequencies.

**Symbols.** $pos$ is token position; $i$ dimension index; $d$ model dimension.

**Practical use case.** Original Transformer positional representation.

**Source.** Vaswani et al. (2017).

#### Learned absolute position embedding

**Equation**

$$
h_i^{(0)}=E_{x_i}+P_i,\quad P\in\mathbb{R}^{n_{\max}\times d}
$$

**Plain-English explanation.** Learns one position vector per position.

**Symbols.** $P_i$ is learned position embedding.

**Practical use case.** GPT-style models with finite context length.

**Source.** Radford et al. (2018); Devlin et al. (2019).

#### RoPE rotation matrix

**Equation**

$$
R_{\theta,i}=\begin{bmatrix}\cos(i\theta)&-\sin(i\theta)\\\sin(i\theta)&\cos(i\theta)\end{bmatrix}
$$

**Plain-English explanation.** Rotates query/key vector pairs by position-dependent angles.

**Symbols.** $i$ is position; $\theta$ is frequency.

**Practical use case.** Relative-position-aware attention without additive embeddings.

**Source.** Su et al. (2021).

#### RoPE query-key property

**Equation**

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k
$$

**Plain-English explanation.** Dot products depend on relative position difference.

**Symbols.** $m,n$ are positions.

**Practical use case.** Long-context decoder attention and LLaMA-style models.

**Source.** Su et al. (2021).

#### RoPE complex form

**Equation**

$$
\tilde{x}_{2j}+i\tilde{x}_{2j+1}=(x_{2j}+ix_{2j+1})e^{i m\theta_j}
$$

**Plain-English explanation.** Applies RoPE as complex multiplication.

**Symbols.** $m$ is position; $\theta_j$ frequency for pair $j$.

**Practical use case.** Efficient implementation and derivation of RoPE.

**Source.** Su et al. (2021).

#### ALiBi bias

**Equation**

$$
\mathrm{score}_{\mathrm{ij}}=\frac{q_i^\top k_j}{\sqrt{d_k}}-m_h(i-j)
$$

**Plain-English explanation.** Adds head-specific linear distance penalty to attention scores.

**Symbols.** $m_h$ is slope for head $h$; $i-j$ is distance.

**Practical use case.** Length extrapolation without learned position embeddings.

**Source.** Press et al. (2021).

#### T5 relative position bias

**Equation**

$$
\mathrm{score}_{\mathrm{ij}}=\frac{q_i^\top k_j}{\sqrt{d_k}}+b_{\mathrm{bucket}(i-j)}
$$

**Plain-English explanation.** Adds a learned bias based on bucketed relative distance.

**Symbols.** $b$ is learned relative position bias.

**Practical use case.** Encoder-decoder and text-to-text Transformer models.

**Source.** Raffel et al. (2020).

#### xPos scale

**Equation**

$$
q_m'=R_m q_m\odot s_m,\quad k_n'=R_n k_n\odot s_n^{-1}
$$

**Plain-English explanation.** Adds a multiplicative scale to RoPE to improve length extrapolation.

**Symbols.** $s_m$ is position-dependent scale.

**Practical use case.** Long-context extrapolation.

**Source.** Sun et al. (2022).

#### NoPE

**Equation**

$$
h_i^{(0)}=E_{x_i}
$$

**Plain-English explanation.** Uses no explicit positional encoding.

**Symbols.** $E_{x_i}$ is token embedding only.

**Practical use case.** Analyzing whether attention can infer position implicitly.

**Source.** Haviv et al. (2022); Kazemnejad et al. (2023).

#### Position interpolation

**Equation**

$$
m' = m\cdot\frac{L_{\mathrm{train}}}{L_{\mathrm{target}}}
$$

**Plain-English explanation.** Compresses target positions into the training position range.

**Symbols.** $L_{train}$ original context; $L_{target}$ extended context.

**Practical use case.** Extending RoPE models to longer context.

**Source.** Chen et al. (2023).

#### NTK-aware RoPE scaling

**Equation**

$$
\theta_j' = \theta_j\cdot \alpha^{-2j/(d-2)}
$$

**Plain-English explanation.** Changes RoPE frequencies to support longer context.

**Symbols.** $\alpha$ is scale factor; $j$ frequency index.

**Practical use case.** Long-context LLaMA-style inference and fine-tuning.

**Source.** bloc97 community method; RoPE scaling literature (2023).

#### YaRN ramp scaling

**Equation**

$$
\theta_j' = \theta_j / s_j,\quad s_j = (1-r_j)s_{\mathrm{low}}+r_j s_{\mathrm{high}}
$$

**Plain-English explanation.** Uses dimension-dependent scaling for RoPE frequencies.

**Symbols.** $r_j$ is ramp interpolation by dimension; $s_j$ is scale.

**Practical use case.** Efficient context-window extension with limited fine-tuning.

**Source.** Peng et al. (2023).

#### Attention sink score

**Equation**

$$
\alpha_{\mathrm{ij}}=\frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_{t}\exp(q_i^\top k_t/\sqrt{d})}
$$

**Plain-English explanation.** Certain early tokens can receive persistent attention mass and act as sinks.

**Symbols.** $\alpha_{ij}$ is attention probability.

**Practical use case.** Streaming LLM and long-context stability analysis.

**Source.** Xiao et al. (2023).

## 5. Normalization

#### LayerNorm

**Equation**

$$
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\quad \mu=\frac{1}{d}\sum_i x_i,\quad \sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2
$$

**Plain-English explanation.** Normalizes features within one token representation.

**Symbols.** $\gamma,\beta$ are learned scale and bias.

**Practical use case.** Standard Transformer normalization.

**Source.** Ba et al. (2016).

#### RMSNorm

**Equation**

$$
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
$$

**Plain-English explanation.** Normalizes by root mean square without subtracting mean.

**Symbols.** $d$ feature dimension; $\gamma$ learned scale.

**Practical use case.** Efficient normalization in LLaMA-style models.

**Source.** Zhang and Sennrich (2019).

#### BatchNorm

**Equation**

$$
\mathrm{BN}(x)=\gamma\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta
$$

**Plain-English explanation.** Normalizes using mini-batch statistics.

**Symbols.** $\mu_B,\sigma_B^2$ are batch mean and variance.

**Practical use case.** Mostly vision/CNNs; less common in autoregressive LLMs.

**Source.** Ioffe and Szegedy (2015).

#### GroupNorm

**Equation**

$$
\mathrm{GN}(x)=\gamma\frac{x-\mu_G}{\sqrt{\sigma_G^2+\epsilon}}+\beta
$$

**Plain-English explanation.** Normalizes groups of channels independently.

**Symbols.** $G$ denotes a feature group.

**Practical use case.** Vision-language and convolutional components.

**Source.** Wu and He (2018).

#### DeepNorm residual scaling

**Equation**

$$
x_{l+1}=\mathrm{LN}(\alpha x_l+G_l(x_l))
$$

**Plain-English explanation.** Scales residual paths to stabilize very deep Transformers.

**Symbols.** $\alpha$ is residual scaling coefficient; $G_l$ sublayer.

**Practical use case.** Training extremely deep Transformers.

**Source.** Wang et al. (2022).

## 6. Activations and Gated MLPs

#### ReLU

**Equation**

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

**Plain-English explanation.** Keeps positive values and zeroes negative values.

**Symbols.** $x$ is scalar or tensor.

**Practical use case.** Basic neural activation.

**Source.** Nair and Hinton (2010).

#### GELU exact

**Equation**

$$
\mathrm{GELU}(x)=x\Phi(x)
$$

**Plain-English explanation.** Weights input by the Gaussian CDF.

**Symbols.** $\Phi$ is standard normal CDF.

**Practical use case.** BERT/GPT-family activation.

**Source.** Hendrycks and Gimpel (2016).

#### GELU tanh approximation

**Equation**

$$
\mathrm{GELU}(x)\approx 0.5x\left(1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right)
$$

**Plain-English explanation.** Fast approximation of GELU.

**Symbols.** $x$ is activation.

**Practical use case.** Efficient Transformer MLPs.

**Source.** Hendrycks and Gimpel (2016).

#### SiLU / Swish

**Equation**

$$
\mathrm{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}
$$

**Plain-English explanation.** Smooth gated activation.

**Symbols.** $\sigma$ is logistic sigmoid.

**Practical use case.** Used in SwiGLU and modern LLM MLPs.

**Source.** Elfwing et al. (2018); Ramachandran et al. (2017).

#### Mish

**Equation**

$$
\mathrm{Mish}(x)=x\tanh(\log(1+e^x))
$$

**Plain-English explanation.** Smooth non-monotonic activation.

**Symbols.** $\log(1+e^x)$ is softplus.

**Practical use case.** Alternative activation in deep networks.

**Source.** Misra (2019).

#### GLU

**Equation**

$$
\mathrm{GLU}(x)= (xW_a+b_a)\odot\sigma(xW_b+b_b)
$$

**Plain-English explanation.** One projection gates another projection.

**Symbols.** $W_a,W_b$ are learned matrices.

**Practical use case.** Gated feed-forward layers.

**Source.** Dauphin et al. (2017).

#### ReGLU

**Equation**

$$
\mathrm{ReGLU}(x)= (xW_a)\odot\mathrm{ReLU}(xW_b)
$$

**Plain-English explanation.** Uses a ReLU gate in a GLU-style block.

**Symbols.** $\odot$ is elementwise product.

**Practical use case.** Transformer feed-forward alternatives.

**Source.** Shazeer (2020).

#### GeGLU

**Equation**

$$
\mathrm{GeGLU}(x)= (xW_a)\odot\mathrm{GELU}(xW_b)
$$

**Plain-English explanation.** Uses GELU as the gate.

**Symbols.** $W_a,W_b$ are projections.

**Practical use case.** T5 and modern Transformer FFN variants.

**Source.** Shazeer (2020).

#### SwiGLU

**Equation**

$$
\mathrm{SwiGLU}(x)= (xW_a)\odot\mathrm{SiLU}(xW_b)
$$

**Plain-English explanation.** Uses SiLU as the multiplicative gate.

**Symbols.** $W_a,W_b$ are learned matrices.

**Practical use case.** LLaMA-family gated MLPs.

**Source.** Shazeer (2020); Touvron et al. (2023).

## 7. Training Losses

#### Label smoothing

**Equation**

$$
y_i^{\mathrm{LS}}=(1-\epsilon)y_i+\frac{\epsilon}{V}
$$

**Plain-English explanation.** Moves some probability mass from the correct class to other classes.

**Symbols.** $\epsilon$ smoothing coefficient; $V$ vocabulary size.

**Practical use case.** Regularizes classification and seq2seq models.

**Source.** Szegedy et al. (2016).

#### Focal loss

**Equation**

$$
\mathcal{L}_{\mathrm{focal}}=-(1-p_t)^\gamma\log p_t
$$

**Plain-English explanation.** Down-weights easy examples and focuses on hard examples.

**Symbols.** $p_t$ probability assigned to the true class; $\gamma$ focusing parameter.

**Practical use case.** Imbalanced classification and safety classifiers.

**Source.** Lin et al. (2017).

#### InfoNCE

**Equation**

$$
\mathcal{L}_{\mathrm{InfoNCE}}=-\log\frac{\exp(s(q,k^+)/\tau)}{\exp(s(q,k^+)/\tau)+\sum_{k^-}\exp(s(q,k^-)/\tau)}
$$

**Plain-English explanation.** Contrasts a positive pair against negatives.

**Symbols.** $s$ similarity; $\tau$ temperature; $k^+$ positive; $k^-$ negatives.

**Practical use case.** Contrastive sentence embeddings, DPR, CLIP-style training.

**Source.** Oord et al. (2018).

#### NT-Xent

**Equation**

$$
\ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i,z_k)/\tau)}
$$

**Plain-English explanation.** Normalized temperature-scaled contrastive loss.

**Symbols.** $z_i,z_j$ are paired representations.

**Practical use case.** SimCLR-style contrastive learning.

**Source.** Chen et al. (2020).

#### Triplet loss

**Equation**

$$
\mathcal{L}=\max(0,d(a,p)-d(a,n)+m)
$$

**Plain-English explanation.** Pushes anchor-positive closer than anchor-negative by a margin.

**Symbols.** $a$ anchor; $p$ positive; $n$ negative; $m$ margin.

**Practical use case.** Embedding training and retrieval.

**Source.** Schroff et al. (2015).

#### Masked language modeling loss

**Equation**

$$
\mathcal{L}_{\mathrm{MLM}}=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})
$$

**Plain-English explanation.** Predicts masked tokens using bidirectional context.

**Symbols.** $M$ masked positions.

**Practical use case.** BERT-style pretraining.

**Source.** Devlin et al. (2019).

#### Causal language modeling loss

**Equation**

$$
\mathcal{L}_{\mathrm{CLM}}=-\sum_{t}\log p_\theta(x_t\mid x_{1:t-1})
$$

**Plain-English explanation.** Predicts each token from previous tokens only.

**Symbols.** $x_{\lt t}$ past context.

**Practical use case.** GPT-style pretraining.

**Source.** Radford et al. (2018).

#### Span corruption objective

**Equation**

$$
\mathcal{L}_{\mathrm{span}}=-\sum_{t}\log p_\theta(y_t\mid y_{1:t-1},\tilde{x})
$$

**Plain-English explanation.** Reconstructs masked spans from corrupted input.

**Symbols.** $\tilde{x}$ corrupted input; $y$ target spans.

**Practical use case.** T5-style denoising pretraining.

**Source.** Raffel et al. (2020).

#### Knowledge distillation

**Equation**

$$
\mathcal{L}_{\mathrm{KD}}=T^2D_{\mathrm{KL}}\left(\mathrm{softmax}(z_T/T)\Vert\mathrm{softmax}(z_S/T)\right)
$$

**Plain-English explanation.** Trains a student to match teacher soft logits.

**Symbols.** $T$ temperature; $z_T,z_S$ teacher/student logits.

**Practical use case.** Compressing LLMs and transferring behavior.

**Source.** Hinton et al. (2015).

#### Hidden-state distillation

**Equation**

$$
\mathcal{L}_{\mathrm{hid}}=\sum_{\ell}\left\|H_{S}^{(\ell)}W_\ell-H_T^{(m(\ell))}\right\|_2^2
$$

**Plain-English explanation.** Matches student hidden states to teacher hidden states.

**Symbols.** $H_S,H_T$ student/teacher states; $m(\ell)$ layer mapping.

**Practical use case.** DistilBERT-style model compression.

**Source.** Sanh et al. (2019).

## 8. Optimization

#### SGD

**Equation**

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}(\theta_t)
$$

**Plain-English explanation.** Moves parameters opposite the gradient.

**Symbols.** $\eta$ learning rate.

**Practical use case.** Baseline neural optimization.

**Source.** Robbins and Monro (1951).

#### Momentum

**Equation**

$$
v_t=\mu v_{t-1}+g_t,\quad \theta_{t+1}=\theta_t-\eta v_t
$$

**Plain-English explanation.** Accumulates gradient direction over time.

**Symbols.** $v_t$ velocity; $\mu$ momentum; $g_t$ gradient.

**Practical use case.** Faster, smoother optimization.

**Source.** Polyak (1964).

#### Adam moments

**Equation**

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

**Plain-English explanation.** Tracks first and second moments of gradients.

**Symbols.** $m_t$ mean; $v_t$ uncentered variance.

**Practical use case.** Default optimizer family for LLMs.

**Source.** Kingma and Ba (2015).

#### Adam update

**Equation**

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad \hat{v}_t=\frac{v_t}{1-\beta_2^t},\quad \theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

**Plain-English explanation.** Bias-corrects moments and applies adaptive step size.

**Symbols.** $\beta_1,\beta_2$ decay rates.

**Practical use case.** Pretraining and fine-tuning.

**Source.** Kingma and Ba (2015).

#### AdamW

**Equation**

$$
\theta_{t+1}=\theta_t-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\theta_t\right)
$$

**Plain-English explanation.** Decouples weight decay from adaptive gradient update.

**Symbols.** $\lambda$ weight decay coefficient.

**Practical use case.** Standard LLM training optimizer.

**Source.** Loshchilov and Hutter (2019).

#### Lion

**Equation**

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad \theta_{t+1}=\theta_t-\eta\mathrm{sign}(m_t)
$$

**Plain-English explanation.** Uses sign of momentum-like update.

**Symbols.** $m_t$ momentum estimate.

**Practical use case.** Memory-efficient optimization experiments.

**Source.** Chen et al. (2023).

#### Adafactor factored second moment

**Equation**

$$
\hat{V}_{\mathrm{ij}}=\frac{r_i c_j}{\bar{r}},\quad r_i=\frac{1}{n}\sum_j V_{\mathrm{ij}},\quad c_j=\frac{1}{m}\sum_i V_{\mathrm{ij}}
$$

**Plain-English explanation.** Approximates second-moment matrix using row and column factors.

**Symbols.** $r,c$ are row/column statistics.

**Practical use case.** Memory-efficient training of large Transformers.

**Source.** Shazeer and Stern (2018).

#### Global gradient clipping

**Equation**

$$
g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2}\right)
$$

**Plain-English explanation.** Scales gradients down when norm exceeds threshold.

**Symbols.** $c$ clipping threshold.

**Practical use case.** Prevents exploding gradients.

**Source.** Pascanu et al. (2013).

#### Cosine decay schedule

**Equation**

$$
\eta_t=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

**Plain-English explanation.** Smoothly decays learning rate following a cosine curve.

**Symbols.** $T$ total steps.

**Practical use case.** LLM pretraining and fine-tuning schedules.

**Source.** Loshchilov and Hutter (2017).

#### Linear warmup

**Equation**

$$
\eta_t=\eta_{\max}\frac{t}{T_{\mathrm{warm}}},\quad t\leq T_{\mathrm{warm}}
$$

**Plain-English explanation.** Gradually increases learning rate at the beginning of training.

**Symbols.** $T_{warm}$ warmup steps.

**Practical use case.** Stabilizes early large-scale training.

**Source.** Vaswani et al. (2017).

#### Inverse square-root schedule

**Equation**

$$
\eta_t=d_{\mathrm{model}}^{-1/2}\min(t^{-1/2},tT_{\mathrm{warm}}^{-3/2})
$$

**Plain-English explanation.** Warmup followed by inverse-square-root decay.

**Symbols.** $d_{model}$ hidden size.

**Practical use case.** Original Transformer schedule.

**Source.** Vaswani et al. (2017).

## 9. Sampling and Decoding

#### Greedy decoding

**Equation**

$$
x_t=\mathrm{argmax}_{i} p_\theta(i\mid x_{1:t-1})
$$

**Plain-English explanation.** Selects the most likely next token.

**Symbols.** $i$ ranges over vocabulary.

**Practical use case.** Deterministic generation.

**Source.** Standard decoding method.

#### Temperature sampling

**Equation**

$$
p_i^{(\tau)}=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

**Plain-English explanation.** Controls sharpness of the output distribution.

**Symbols.** $\tau$ temperature; lower is more deterministic.

**Practical use case.** Creative vs precise generation control.

**Source.** Ackley et al. (1985); neural LM practice.

#### Top-k sampling

**Equation**

$$
S_k=\mathrm{TopK}(p,k),\quad p_i'=\frac{p_i\mathbf{1}[i\in S_k]}{\sum_{j\in S_k}p_j}
$$

**Plain-English explanation.** Keeps only the k most likely tokens and renormalizes.

**Symbols.** $S_k$ top-k token set.

**Practical use case.** Open-ended generation.

**Source.** Fan et al. (2018).

#### Nucleus top-p sampling

**Equation**

$$
S_p=\min\left\{S:\sum_{i\in S}p_i\geq p\right\},\quad p_i'=\frac{p_i\mathbf{1}[i\in S_p]}{\sum_{j\in S_p}p_j}
$$

**Plain-English explanation.** Keeps the smallest token set whose cumulative probability exceeds threshold p.

**Symbols.** $p$ is nucleus threshold.

**Practical use case.** Adaptive sampling for open-ended text.

**Source.** Holtzman et al. (2020).

#### Typical sampling

**Equation**

$$
S=\left\{i:\left| -\log p_i-H(p)\right|\leq \tau\right\}
$$

**Plain-English explanation.** Keeps tokens whose surprise is close to distribution entropy.

**Symbols.** $H(p)$ entropy; $\tau$ typicality threshold.

**Practical use case.** Avoids both dull and unlikely tokens.

**Source.** Meister et al. (2023).

#### Min-p sampling

**Equation**

$$
S=\{i:p_i\geq \alpha\max_jp_j\}
$$

**Plain-English explanation.** Keeps tokens whose probability is at least a fraction of the top token.

**Symbols.** $\alpha$ min-p threshold.

**Practical use case.** Sharper adaptive sampling for LLM inference.

**Source.** Community decoding method, 2023.

#### Beam search recursion

**Equation**

$$
B_t=\mathrm{TopB}_{y_{1:t}}\left[\log p_\theta(y_{1:t}\mid x)\right]
$$

**Plain-English explanation.** Maintains the top B partial sequences by log-probability.

**Symbols.** $B_t$ beam set at step $t$.

**Practical use case.** Translation and structured decoding.

**Source.** Lowerre (1976); Sutskever et al. (2014).

#### Beam length penalty

**Equation**

$$
s(y)=\frac{\log p(y\mid x)}{\left(\frac{5+|y|}{6}\right)^\alpha}
$$

**Plain-English explanation.** Penalizes or normalizes beam scores by sequence length.

**Symbols.** $|y|$ output length; $\alpha$ penalty strength.

**Practical use case.** Reduces short-output bias.

**Source.** Wu et al. (2016).

#### Contrastive decoding

**Equation**

$$
x_t=\mathrm{argmax}_{x}\left[\log p_{\mathrm{expert}}(x\mid c)-\alpha\log p_{\mathrm{amateur}}(x\mid c)\right]
$$

**Plain-English explanation.** Selects tokens preferred by a strong model over a weaker model.

**Symbols.** $\alpha$ controls amateur penalty.

**Practical use case.** Reducing degeneration and improving factuality.

**Source.** Li et al. (2023).

#### Speculative decoding acceptance

**Equation**

$$
a_t=\min\left(1,\frac{p(x_t\mid x_{1:t-1})}{q(x_t\mid x_{1:t-1})}\right)
$$

**Plain-English explanation.** Accepts draft-model tokens with probability based on target/draft likelihood ratio.

**Symbols.** $p$ target model; $q$ draft model.

**Practical use case.** Low-latency LLM serving.

**Source.** Leviathan et al. (2023); Chen et al. (2023).

#### Mirostat target surprise update

**Equation**

$$
\mu_{t+1}=\mu_t-\eta(s_t-\tau)
$$

**Plain-English explanation.** Adjusts sampling to maintain target surprise/perplexity.

**Symbols.** $s_t$ observed surprise; $\tau$ target; $\eta$ update rate.

**Practical use case.** Diversity control in text generation.

**Source.** Basu et al. (2021).

## 10. Alignment and Preference Optimization

#### Supervised fine-tuning

**Equation**

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)=-\mathbb{E}_{(x,y)\sim D}\sum_t\log\pi_\theta(y_t\mid x,y_{1:t-1})
$$

**Plain-English explanation.** Trains the model on demonstration responses.

**Symbols.** $D$ supervised instruction dataset.

**Practical use case.** Instruction tuning before preference optimization.

**Source.** Ouyang et al. (2022).

#### Bradley-Terry preference model

**Equation**

$$
P(y_w\succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

**Plain-English explanation.** Models probability that a winning response is preferred to a losing response.

**Symbols.** $r_\phi$ reward model; $y_w,y_l$ preferred/rejected responses.

**Practical use case.** Reward modeling for RLHF.

**Source.** Bradley and Terry (1952); Ouyang et al. (2022).

#### Reward model loss

**Equation**

$$
\mathcal{L}_{\mathrm{RM}}=-\mathbb{E}\left[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\right]
$$

**Plain-English explanation.** Trains rewards so preferred responses score higher.

**Symbols.** $\phi$ reward model parameters.

**Practical use case.** RLHF reward model training.

**Source.** Christiano et al. (2017); Ouyang et al. (2022).

#### RLHF KL-regularized objective

**Equation**

$$
\max_\theta\;\mathbb{E}_{y\sim\pi_\theta}\left[r_\phi(x,y)-\beta D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x))\right]
$$

**Plain-English explanation.** Optimizes reward while preventing policy from drifting too far from reference model.

**Symbols.** $\beta$ KL penalty; $\pi_{ref}$ reference policy.

**Practical use case.** RLHF policy training.

**Source.** Ziegler et al. (2019); Ouyang et al. (2022).

#### PPO clipped objective

**Equation**

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min(r_t(\theta)A_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)\right]
$$

**Plain-English explanation.** Limits how much a policy update can change action probabilities.

**Symbols.** $r_t=\pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$; $A_t$ advantage.

**Practical use case.** RLHF fine-tuning.

**Source.** Schulman et al. (2017); Ouyang et al. (2022).

#### Value function loss

**Equation**

$$
\mathcal{L}_V=\mathbb{E}_t\left[(V_\psi(s_t)-R_t)^2\right]
$$

**Plain-English explanation.** Trains a critic to predict returns.

**Symbols.** $V_\psi$ value function; $R_t$ return.

**Practical use case.** PPO/RLHF critic training.

**Source.** Sutton and Barto (2018); Schulman et al. (2017).

#### GAE

**Equation**

$$
\hat{A}_t^{\mathrm{GAE}}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},\quad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

**Plain-English explanation.** Computes a low-variance advantage estimate from TD residuals.

**Symbols.** $\gamma$ discount; $\lambda$ bias-variance parameter.

**Practical use case.** PPO and actor-critic training.

**Source.** Schulman et al. (2016).

#### DPO implicit reward

**Equation**

$$
r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+C(x)
$$

**Plain-English explanation.** Defines an implicit reward from policy/reference log-ratio.

**Symbols.** $\beta$ temperature; $C(x)$ prompt-only constant.

**Practical use case.** Direct preference optimization.

**Source.** Rafailov et al. (2023).

#### DPO loss

**Equation**

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)
$$

**Plain-English explanation.** Optimizes preference pairs directly without an explicit reward model.

**Symbols.** $y_w,y_l$ winner/loser responses.

**Practical use case.** Alignment after SFT.

**Source.** Rafailov et al. (2023).

#### IPO loss

**Equation**

$$
\mathcal{L}_{\mathrm{IPO}}=\mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w\mid x)\pi_{\mathrm{ref}}(y_l\mid x)}{\pi_\theta(y_l\mid x)\pi_{\mathrm{ref}}(y_w\mid x)}-\frac{1}{2\beta}\right)^2\right]
$$

**Plain-English explanation.** Uses a squared loss on preference log-ratio margins.

**Symbols.** $\beta$ controls target margin.

**Practical use case.** Stable preference optimization alternative to DPO.

**Source.** Azar et al. (2023).

#### KTO utility

**Equation**

$$
v(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

**Plain-English explanation.** Computes desirability from policy/reference log-ratio.

**Symbols.** $v$ is human utility surrogate.

**Practical use case.** Preference tuning with desirable/undesirable labels.

**Source.** Ethayarajh et al. (2024).

#### ORPO loss

**Equation**

$$
\mathcal{L}_{\mathrm{ORPO}}=\mathcal{L}_{\mathrm{SFT}}-\lambda\log\sigma\left(\log\frac{\pi_\theta(y_w\mid x)}{1-\pi_\theta(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{1-\pi_\theta(y_l\mid x)}\right)
$$

**Plain-English explanation.** Combines supervised learning with an odds-ratio preference penalty.

**Symbols.** $\lambda$ preference weight.

**Practical use case.** Reference-free preference alignment.

**Source.** Hong et al. (2024).

#### SimPO loss

**Equation**

$$
\mathcal{L}_{\mathrm{SimPO}}=-\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)-\gamma\right)
$$

**Plain-English explanation.** Uses length-normalized policy likelihood margin without reference model.

**Symbols.** $\gamma$ target margin.

**Practical use case.** Simple reference-free preference optimization.

**Source.** Meng et al. (2024).

#### GRPO group advantage

**Equation**

$$
\hat{A}_{i}=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})}
$$

**Plain-English explanation.** Computes advantage relative to other sampled responses for the same prompt.

**Symbols.** $G$ samples per prompt; $r_i$ reward of sample $i$.

**Practical use case.** RL for reasoning models without a value model.

**Source.** Shao et al. (2024).

#### GRPO objective

**Equation**

$$
\mathcal{J}_{\mathrm{GRPO}}=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min(\rho_i\hat{A}_i,\mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat{A}_i)-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\right]
$$

**Plain-English explanation.** PPO-style clipped objective with group-relative advantages.

**Symbols.** $\rho_i=\pi_\theta(o_i\mid q)/\pi_{old}(o_i\mid q)$.

**Practical use case.** RLVR training for math/reasoning models.

**Source.** Shao et al. (2024).

#### RLOO baseline

**Equation**

$$
\hat{A}_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j
$$

**Plain-English explanation.** Uses leave-one-out average reward as baseline.

**Symbols.** $G$ completions for the same prompt.

**Practical use case.** Low-variance RL for LLMs.

**Source.** Kool et al. (2019); Ahmadian et al. (2024).

#### Rejection sampling fine-tuning

**Equation**

$$
D' = \{(x,y): r_\phi(x,y)\geq \tau\},\quad \mathcal{L}=-\mathbb{E}_{(x,y)\in D'}\log\pi_\theta(y\mid x)
$$

**Plain-English explanation.** Filters generated samples by reward then trains on accepted samples.

**Symbols.** $\tau$ reward threshold.

**Practical use case.** Improving instruction models without online RL.

**Source.** Ouyang et al. (2022).

## 11. PEFT and Adapter Methods

#### LoRA

**Equation**

$$
W'=W+\Delta W,\quad \Delta W=\frac{\alpha}{r}BA
$$

**Plain-English explanation.** Adds a trainable low-rank update to frozen weights.

**Symbols.** $A\in\mathbb{R}^{r\times d_{in}}$, $B\in\mathbb{R}^{d_{out}\times r}$; $r$ rank.

**Practical use case.** Memory-efficient fine-tuning.

**Source.** Hu et al. (2022).

#### QLoRA dequantized LoRA layer

**Equation**

$$
y=x\mathrm{dequant}(W_q)+x\frac{\alpha}{r}BA
$$

**Plain-English explanation.** Combines quantized frozen weights with trainable LoRA adapters.

**Symbols.** $W_q$ quantized base weight.

**Practical use case.** Fine-tuning large models on limited GPU memory.

**Source.** Dettmers et al. (2023).

#### DoRA

**Equation**

$$
W'=m\frac{W+\Delta W}{\|W+\Delta W\|_c}
$$

**Plain-English explanation.** Separates weight magnitude and direction adaptation.

**Symbols.** $m$ magnitude parameter; $\|\cdot\|_c$ column norm.

**Practical use case.** Improves LoRA-style fine-tuning quality.

**Source.** Liu et al. (2024).

#### AdaLoRA rank allocation

**Equation**

$$
\Delta W=\sum_{i=1}^{r}s_i u_i v_i^\top
$$

**Plain-English explanation.** Allocates rank budget by learned importance scores.

**Symbols.** $s_i$ singular value/importance; $u_i,v_i$ factors.

**Practical use case.** Adaptive parameter-efficient fine-tuning.

**Source.** Zhang et al. (2023).

#### Prefix tuning

**Equation**

$$
K'=[K_p;K],\quad V'=[V_p;V]
$$

**Plain-English explanation.** Prepends trainable key/value vectors to attention.

**Symbols.** $K_p,V_p$ learned prefix states.

**Practical use case.** Task adaptation without updating model weights.

**Source.** Li and Liang (2021).

#### Prompt tuning

**Equation**

$$
h^{(0)}=[P_1,\dots,P_m,E_{x_1},\dots,E_{x_n}]
$$

**Plain-English explanation.** Prepends trainable soft prompt embeddings.

**Symbols.** $P_i$ learned prompt vectors.

**Practical use case.** Parameter-efficient adaptation.

**Source.** Lester et al. (2021).

#### P-tuning v2 deep prompts

**Equation**

$$
h_\ell'=[P_\ell;h_\ell]
$$

**Plain-English explanation.** Adds trainable prompts at multiple layers.

**Symbols.** $P_\ell$ prompt vectors for layer $\ell$.

**Practical use case.** Deep prompt tuning for NLU/NLG.

**Source.** Liu et al. (2022).

#### IA3

**Equation**

$$
y=(l_W\odot W)x,\quad K'=l_K\odot K,\quad V'=l_V\odot V
$$

**Plain-English explanation.** Learns multiplicative vectors that rescale activations or weights.

**Symbols.** $l_W,l_K,l_V$ learned scaling vectors.

**Practical use case.** Very lightweight adaptation.

**Source.** Liu et al. (2022).

#### BitFit

**Equation**

$$
\theta_{\mathrm{train}}=\{b: b\in\theta\}
$$

**Plain-English explanation.** Fine-tunes only bias parameters.

**Symbols.** $b$ bias vectors.

**Practical use case.** Cheap task adaptation baseline.

**Source.** Ben Zaken et al. (2022).

## 12. Quantization

#### Affine quantization

**Equation**

$$
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,q_{\min},q_{\max}\right),\quad \hat{x}=s(q-z)
$$

**Plain-English explanation.** Maps floating-point values to integers using scale and zero-point.

**Symbols.** $s$ scale; $z$ zero-point.

**Practical use case.** INT8/INT4 deployment.

**Source.** Jacob et al. (2018).

#### Symmetric quantization scale

**Equation**

$$
s=\frac{\max(|x|)}{2^{b-1}-1},\quad q=\mathrm{round}(x/s)
$$

**Plain-English explanation.** Uses zero-centered integer range.

**Symbols.** $b$ number of bits.

**Practical use case.** Weight quantization.

**Source.** Jacob et al. (2018).

#### Asymmetric quantization scale

**Equation**

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad z=\mathrm{round}\left(q_{\min}-\frac{x_{\min}}{s}\right)
$$

**Plain-English explanation.** Uses non-zero zero-point to cover arbitrary ranges.

**Symbols.** $x_{min},x_{max}$ floating range.

**Practical use case.** Activation quantization.

**Source.** Jacob et al. (2018).

#### Per-channel quantization

**Equation**

$$
s_c=\frac{\max_i |W_{c,i}|}{2^{b-1}-1}
$$

**Plain-English explanation.** Computes a separate scale for each channel.

**Symbols.** $c$ output channel.

**Practical use case.** Improves quantized weight accuracy.

**Source.** Jacob et al. (2018).

#### GPTQ objective

**Equation**

$$
\min_{\hat{W}}\|WX-\hat{W}X\|_2^2
$$

**Plain-English explanation.** Quantizes weights while minimizing output reconstruction error.

**Symbols.** $X$ calibration activations; $\hat{W}$ quantized weights.

**Practical use case.** Post-training LLM quantization.

**Source.** Frantar et al. (2023).

#### AWQ salient-channel scaling

**Equation**

$$
\min_s\left\|Q(W\mathrm{diag}(s))\mathrm{diag}(s)^{-1}X-WX\right\|_2^2
$$

**Plain-English explanation.** Scales channels before quantization to protect important weights.

**Symbols.** $s$ channel scale vector; $Q$ quantizer.

**Practical use case.** Accurate low-bit weight-only quantization.

**Source.** Lin et al. (2023).

#### SmoothQuant transform

**Equation**

$$
\hat{X}=X\mathrm{diag}(s)^{-1},\quad \hat{W}=\mathrm{diag}(s)W
$$

**Plain-English explanation.** Migrates activation outlier difficulty into weights.

**Symbols.** $s$ smoothing vector.

**Practical use case.** W8A8 LLM quantization.

**Source.** Xiao et al. (2023).

#### LLM.int8 outlier decomposition

**Equation**

$$
XW=X_{\mathrm{normal}}W_{\mathrm{int8}}+X_{\mathrm{outlier}}W_{\mathrm{fp16}}
$$

**Plain-English explanation.** Computes most values in int8 while keeping outliers in higher precision.

**Symbols.** $X_{outlier}$ activation outlier subspace.

**Practical use case.** 8-bit inference for large LMs.

**Source.** Dettmers et al. (2022).

#### NF4 quantization

**Equation**

$$
q_i=\mathrm{argmin}_{c_j\in C_{\mathrm{NF4}}}|x_i/s-c_j|
$$

**Plain-English explanation.** Maps normalized values to a learned/non-uniform 4-bit codebook optimized for normal weights.

**Symbols.** $C_{NF4}$ NF4 codebook; $s$ block scale.

**Practical use case.** QLoRA 4-bit finetuning.

**Source.** Dettmers et al. (2023).

#### Double quantization

**Equation**

$$
q_s=Q(s),\quad \hat{x}=\mathrm{dequant}(q_x,\mathrm{dequant}(q_s))
$$

**Plain-English explanation.** Quantizes the quantization scales themselves.

**Symbols.** $q_s$ quantized scale.

**Practical use case.** Memory savings in QLoRA.

**Source.** Dettmers et al. (2023).

#### KV-cache quantization

**Equation**

$$
\hat{K}=s_K(Q(K)-z_K),\quad \hat{V}=s_V(Q(V)-z_V)
$$

**Plain-English explanation.** Stores keys and values in lower precision and dequantizes for attention.

**Symbols.** $K,V$ cached attention tensors.

**Practical use case.** Long-context serving memory reduction.

**Source.** LLM inference quantization literature, 2023-2025.

#### Quantization-aware training loss

**Equation**

$$
\min_\theta \mathbb{E}_{(x,y)}\mathcal{L}(f_{Q(\theta)}(x),y)
$$

**Plain-English explanation.** Trains while simulating quantized weights/activations.

**Symbols.** $Q(\theta)$ fake-quantized parameters.

**Practical use case.** Maintaining accuracy after deployment quantization.

**Source.** Jacob et al. (2018).

## 13. Scaling Laws

#### Kaplan scaling law

**Equation**

$$
L(N,D)=L_\infty + aN^{-\alpha}+bD^{-\beta}
$$

**Plain-English explanation.** Models loss as a power law of parameter count and data size.

**Symbols.** $N$ parameters; $D$ training tokens.

**Practical use case.** Predicting model performance and compute trade-offs.

**Source.** Kaplan et al. (2020).

#### Compute estimate

**Equation**

$$
C\approx 6ND
$$

**Plain-English explanation.** Approximates training compute for dense Transformer training.

**Symbols.** $C$ FLOPs; $N$ parameters; $D$ tokens.

**Practical use case.** Budgeting pretraining runs.

**Source.** Kaplan et al. (2020); Hoffmann et al. (2022).

#### Chinchilla compute-optimal relation

**Equation**

$$
N_{\mathrm{opt}}\propto C^a,\quad D_{\mathrm{opt}}\propto C^b,\quad a\approx b\approx 0.5
$$

**Plain-English explanation.** Compute-optimal training scales parameters and data roughly equally.

**Symbols.** $C$ compute budget.

**Practical use case.** Choosing model/data size under fixed compute.

**Source.** Hoffmann et al. (2022).

#### Loss versus compute

**Equation**

$$
L(C)=L_\infty + AC^{-\alpha}
$$

**Plain-English explanation.** Represents loss decreasing as a power law with compute.

**Symbols.** $A,\alpha$ fitted constants.

**Practical use case.** Forecasting model improvements.

**Source.** Kaplan et al. (2020).

#### Emergence-style logistic fit

**Equation**

$$
P(\mathrm{success}\mid C)=\frac{1}{1+\exp[-a(\log C-b)]}
$$

**Plain-English explanation.** A smooth proxy for abrupt benchmark transitions.

**Symbols.** $a$ slope; $b$ midpoint.

**Practical use case.** Analyzing threshold-like capabilities.

**Source.** Wei et al. (2022); Schaeffer et al. (2023).

## 14. Mixture of Experts

#### MoE gating

**Equation**

$$
g(x)=\mathrm{softmax}(W_gx)
$$

**Plain-English explanation.** Computes expert-selection probabilities.

**Symbols.** $W_g$ router weights.

**Practical use case.** Routing tokens to experts.

**Source.** Shazeer et al. (2017).

#### Top-k expert routing

**Equation**

$$
S(x)=\mathrm{TopK}(g(x),k)
$$

**Plain-English explanation.** Selects the highest-scoring experts for a token.

**Symbols.** $k$ number of selected experts.

**Practical use case.** Sparse MoE activation.

**Source.** Shazeer et al. (2017).

#### MoE output

**Equation**

$$
y=\sum_{e\in S(x)}g_e(x)E_e(x)
$$

**Plain-English explanation.** Combines selected expert outputs weighted by router probabilities.

**Symbols.** $E_e$ expert network $e$.

**Practical use case.** Sparse scaling of Transformer MLPs.

**Source.** Shazeer et al. (2017).

#### Switch Transformer routing

**Equation**

$$
e^*=\mathrm{argmax}_{e} g_e(x),\quad y=g_{e^*}(x)E_{e^*}(x)
$$

**Plain-English explanation.** Routes each token to one expert only.

**Symbols.** $e^*$ selected expert.

**Practical use case.** Efficient sparse MoE training.

**Source.** Fedus et al. (2021).

#### MoE load balancing loss

**Equation**

$$
\mathcal{L}_{\mathrm{aux}}=\alpha N\sum_{e=1}^{N}f_ep_e
$$

**Plain-English explanation.** Encourages balanced expert usage.

**Symbols.** $f_e$ fraction of tokens routed to expert $e$; $p_e$ mean router probability.

**Practical use case.** Avoids expert collapse.

**Source.** Fedus et al. (2021).

#### Expert capacity

**Equation**

$$
C_e=\left\lceil\frac{T}{N}\cdot c\right\rceil
$$

**Plain-English explanation.** Limits how many tokens each expert can process.

**Symbols.** $T$ tokens; $N$ experts; $c$ capacity factor.

**Practical use case.** MoE training/inference memory control.

**Source.** Fedus et al. (2021).

#### Router z-loss

**Equation**

$$
\mathcal{L}_z=\frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_j e^{z_{\mathrm{ij}}}\right)^2
$$

**Plain-English explanation.** Penalizes large router logits for stability.

**Symbols.** $z_{ij}$ router logit.

**Practical use case.** Stable MoE router training.

**Source.** Zoph et al. (2022).

## 15. Tokenization and Embedding Pretraining

#### BPE pair frequency

**Equation**

$$
(a^*,b^*)=\mathrm{argmax}_{(a,b)}\mathrm{count}(a,b)
$$

**Plain-English explanation.** Merges the most frequent adjacent symbol pair.

**Symbols.** $a,b$ are symbols/subwords.

**Practical use case.** Byte-pair tokenization.

**Source.** Sennrich et al. (2016).

#### WordPiece score

**Equation**

$$
\mathrm{score}(a,b)=\frac{\mathrm{count}(ab)}{\mathrm{count}(a)\mathrm{count}(b)}
$$

**Plain-English explanation.** Chooses merges using likelihood-inspired association score.

**Symbols.** $ab$ candidate merged token.

**Practical use case.** BERT-style tokenization.

**Source.** Schuster and Nakajima (2012); Devlin et al. (2019).

#### Unigram LM tokenization

**Equation**

$$
P(x)=\prod_{t\in\mathrm{seg}(x)}p(t)
$$

**Plain-English explanation.** Assigns probability to a segmentation as product of token probabilities.

**Symbols.** $t$ subword token.

**Practical use case.** SentencePiece unigram tokenization.

**Source.** Kudo (2018).

#### Unigram best segmentation

**Equation**

$$
\mathrm{seg}^*(x)=\mathrm{argmax}_{s\in\mathcal{S}(x)}\sum_{t\in s}\log p(t)
$$

**Plain-English explanation.** Finds most likely subword segmentation.

**Symbols.** $\mathcal{S}(x)$ all segmentations.

**Practical use case.** Tokenizing text with unigram models.

**Source.** Kudo (2018).

#### Skip-gram negative sampling

**Equation**

$$
\mathcal{L}=-\log\sigma(v_c^\top v_w)-\sum_{i=1}^{k}\mathbb{E}_{n_i\sim P_n}\log\sigma(-v_{n_i}^\top v_w)
$$

**Plain-English explanation.** Predicts context words from a center word while contrasting negatives.

**Symbols.** $v_w$ center embedding; $v_c$ context embedding.

**Practical use case.** word2vec embeddings.

**Source.** Mikolov et al. (2013).

#### CBOW

**Equation**

$$
p(w_t\mid \mathrm{context})=\mathrm{softmax}\left(W\frac{1}{2m}\sum_{-m\leq j\leq m,j\ne0}v_{w_{t+j}}\right)
$$

**Plain-English explanation.** Predicts a word from averaged context embeddings.

**Symbols.** $m$ context window radius.

**Practical use case.** word2vec CBOW training.

**Source.** Mikolov et al. (2013).

#### GloVe

**Equation**

$$
J=\sum_{i,j}f(X_{\mathrm{ij}})(w_i^\top\tilde{w}_j+b_i+\tilde{b}_j-\log X_{\mathrm{ij}})^2
$$

**Plain-English explanation.** Fits word vectors to log co-occurrence counts.

**Symbols.** $X_{ij}$ co-occurrence count.

**Practical use case.** Static word embeddings.

**Source.** Pennington et al. (2014).

#### fastText subword embedding

**Equation**

$$
v_w=\sum_{g\in G_w}z_g
$$

**Plain-English explanation.** Represents a word as sum of character n-gram embeddings.

**Symbols.** $G_w$ n-grams of word $w$.

**Practical use case.** Robust embeddings for rare/morphological words.

**Source.** Bojanowski et al. (2017).

## 16. Evaluation Metrics

#### BLEU modified precision

**Equation**

$$
\mathrm{BLEU}=BP\cdot\exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

**Plain-English explanation.** Measures n-gram overlap with brevity penalty.

**Symbols.** $p_n$ modified n-gram precision; $BP$ brevity penalty.

**Practical use case.** Machine translation evaluation.

**Source.** Papineni et al. (2002).

#### BLEU brevity penalty

**Equation**

$$
BP=\begin{cases}
1,& c>r\\
e^{1-r/c},& c\leq r
\end{cases}
$$

**Plain-English explanation.** Penalizes translations shorter than reference.

**Symbols.** $c$ candidate length; $r$ reference length.

**Practical use case.** Translation evaluation.

**Source.** Papineni et al. (2002).

#### ROUGE-N

**Equation**

$$
\mathrm{ROUGE}-N=\frac{\sum_{S\in \mathrm{Ref}}\sum_{\mathrm{gram}_n\in S}\min(\mathrm{count}_{\mathrm{cand}}(\mathrm{gram}_n),\mathrm{count}_S(\mathrm{gram}_n))}{\sum_{S\in \mathrm{Ref}}\sum_{\mathrm{gram}_n\in S}\mathrm{count}_S(\mathrm{gram}_n)}
$$

**Plain-English explanation.** Measures recall of reference n-grams.

**Symbols.** $gram_n$ n-gram.

**Practical use case.** Summarization evaluation.

**Source.** Lin (2004).

#### ROUGE-L

**Equation**

$$
\mathrm{ROUGE}-L=\frac{(1+\beta^2)R_{\mathrm{LCS}}P_{\mathrm{LCS}}}{R_{\mathrm{LCS}}+\beta^2P_{\mathrm{LCS}}}
$$

**Plain-English explanation.** Uses longest common subsequence precision and recall.

**Symbols.** $P_{LCS},R_{LCS}$ LCS precision/recall.

**Practical use case.** Summarization evaluation.

**Source.** Lin (2004).

#### METEOR

**Equation**

$$
\mathrm{METEOR}=F_{\mathrm{mean}}(1-\mathrm{Penalty})
$$

**Plain-English explanation.** Combines unigram precision/recall with fragmentation penalty.

**Symbols.** $F_{mean}$ weighted harmonic mean.

**Practical use case.** Translation evaluation.

**Source.** Banerjee and Lavie (2005).

#### BERTScore precision

**Equation**

$$
P=\frac{1}{|\hat{x}|}\sum_{\hat{x}_i}\max_{x_j} \hat{x}_i^\top x_j
$$

**Plain-English explanation.** Matches candidate tokens to reference tokens in embedding space.

**Symbols.** $\hat{x}_i,x_j$ contextual token embeddings.

**Practical use case.** Semantic text evaluation.

**Source.** Zhang et al. (2020).

#### Exact match

**Equation**

$$
EM=\mathbf{1}[\mathrm{normalize}(\hat{y})=\mathrm{normalize}(y)]
$$

**Plain-English explanation.** Checks whether normalized prediction exactly equals answer.

**Symbols.** $\hat{y}$ prediction; $y$ reference.

**Practical use case.** QA evaluation.

**Source.** Rajpurkar et al. (2016).

#### Token F1

**Equation**

$$
F1=\frac{2\mathrm{PR}}{P+R}
$$

**Plain-English explanation.** Harmonic mean of token precision and recall.

**Symbols.** $P$ precision; $R$ recall.

**Practical use case.** Extractive QA and answer matching.

**Source.** Rajpurkar et al. (2016).

#### pass@k

**Equation**

$$
\mathrm{pass}@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

**Plain-English explanation.** Estimates probability that at least one of k samples passes tests.

**Symbols.** $n$ samples; $c$ correct samples.

**Practical use case.** Code generation evaluation.

**Source.** Chen et al. (2021).

#### MAUVE divergence curve

**Equation**

$$
\mathrm{MAUVE}=\mathrm{Area}\left(\{(e^{-D_{\mathrm{KL}}(P\Vert Q_\lambda)},e^{-D_{\mathrm{KL}}(Q\Vert P_\lambda)})\}_{\lambda}\right)
$$

**Plain-English explanation.** Compares generated and real text distributions using divergence frontier.

**Symbols.** $P,Q$ text distributions; $\lambda$ mixture parameter.

**Practical use case.** Open-ended text generation evaluation.

**Source.** Pillutla et al. (2021).

## 17. Production Inference Math

#### Self-attention complexity

**Equation**

$$
\mathrm{FLOPs}_{\mathrm{attn}}=O(n^2d)
$$

**Plain-English explanation.** Attention cost grows quadratically with sequence length.

**Symbols.** $n$ sequence length; $d$ hidden dimension.

**Practical use case.** Capacity planning for long-context LLMs.

**Source.** Vaswani et al. (2017).

#### MLP complexity

**Equation**

$$
\mathrm{FLOPs}_{\mathrm{MLP}}\approx 2nd\,d_{\mathrm{ff}}
$$

**Plain-English explanation.** Feed-forward layers cost proportional to sequence length and intermediate dimension.

**Symbols.** $d_{ff}$ MLP hidden dimension.

**Practical use case.** Estimating dense layer cost.

**Source.** Transformer engineering literature.

#### KV-cache memory

**Equation**

$$
M_{\mathrm{KV}}=2\cdot L\cdot B\cdot S\cdot H_{\mathrm{kv}}\cdot d_h\cdot \mathrm{bytes}
$$

**Plain-English explanation.** Memory needed to store cached keys and values.

**Symbols.** $L$ layers; $B$ batch; $S$ sequence length; $H_{kv}$ KV heads.

**Practical use case.** Sizing GPU memory for serving.

**Source.** Transformer inference engineering literature.

#### MQA/GQA KV memory ratio

**Equation**

$$
\frac{M_{\mathrm{GQA}}}{M_{\mathrm{MHA}}}=\frac{H_{\mathrm{kv}}}{H_q}
$$

**Plain-English explanation.** KV-cache memory scales with KV heads, not query heads.

**Symbols.** $H_{kv}$ KV heads; $H_q$ query heads.

**Practical use case.** Choosing MQA/GQA for latency and memory.

**Source.** Shazeer (2019); Ainslie et al. (2023).

#### Prefill latency

**Equation**

$$
T_{\mathrm{prefill}}\approx\frac{\mathrm{FLOPs}_{\mathrm{prefill}}}{\mathrm{FLOPs/s}_{\mathrm{GPU}}}+T_{\mathrm{overhead}}
$$

**Plain-English explanation.** Estimates time to process prompt tokens.

**Symbols.** $T_{overhead}$ launch/scheduling overhead.

**Practical use case.** TTFT budgeting.

**Source.** LLM serving practice.

#### Decode latency per token

**Equation**

$$
T_{\mathrm{decode}}\approx\frac{\mathrm{FLOPs}_{\mathrm{decode}}}{\mathrm{FLOPs/s}_{\mathrm{GPU}}}+\frac{M_{\mathrm{read}}}{BW}+T_{\mathrm{overhead}}
$$

**Plain-English explanation.** Decode is often limited by memory reads and bandwidth.

**Symbols.** $BW$ memory bandwidth; $M_{read}$ bytes read.

**Practical use case.** Optimizing TPOT.

**Source.** LLM serving practice.

#### TTFT

**Equation**

$$
\mathrm{TTFT}=T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{schedule}}
$$

**Plain-English explanation.** Time until first generated token.

**Symbols.** $T_{queue}$ wait time; $T_{schedule}$ scheduler overhead.

**Practical use case.** Latency SLOs for chat systems.

**Source.** LLM serving practice.

#### TPOT

**Equation**

$$
\mathrm{TPOT}=\frac{T_{\mathrm{decode},\mathrm{total}}}{N_{\mathrm{generated}}}
$$

**Plain-English explanation.** Average time per output token.

**Symbols.** $N_{generated}$ generated tokens.

**Practical use case.** Streaming latency measurement.

**Source.** LLM serving practice.

#### Throughput

**Equation**

$$
\mathrm{throughput}=\frac{N_{\mathrm{tokens}}}{T_{\mathrm{wall}}}
$$

**Plain-English explanation.** Tokens processed per unit time.

**Symbols.** $N_{tokens}$ total tokens; $T_{wall}$ elapsed time.

**Practical use case.** Serving benchmark.

**Source.** Systems performance standard.

#### Arithmetic intensity

**Equation**

$$
AI=\frac{\mathrm{FLOPs}}{\text{Bytes moved}}
$$

**Plain-English explanation.** Measures compute per byte of memory traffic.

**Symbols.** $AI$ arithmetic intensity.

**Practical use case.** Identifying compute-bound vs bandwidth-bound kernels.

**Source.** Williams et al. (2009).

#### Roofline model

**Equation**

$$
P\leq \min(P_{\mathrm{peak}},BW\cdot AI)
$$

**Plain-English explanation.** Performance is bounded by peak compute or memory bandwidth.

**Symbols.** $P$ achieved performance.

**Practical use case.** GPU inference bottleneck analysis.

**Source.** Williams et al. (2009).

#### PagedAttention block allocation

**Equation**

$$
M_{\mathrm{alloc}}=N_{\mathrm{blocks}}\cdot \mathrm{block\_size}\cdot H_{\mathrm{kv}}\cdot d_h\cdot 2\cdot \mathrm{bytes}
$$

**Plain-English explanation.** Allocates KV cache in blocks instead of contiguous sequence buffers.

**Symbols.** $N_{blocks}$ allocated KV blocks.

**Practical use case.** High-throughput serving with fragmented requests.

**Source.** Kwon et al. (2023).

#### Continuous batching effective utilization

**Equation**

$$
U=\frac{\sum_i \mathrm{tokens}_i}{B_{\max}\cdot \mathrm{steps}}
$$

**Plain-English explanation.** Measures how full decode batches are over time.

**Symbols.** $B_{max}$ max batch slots.

**Practical use case.** Scheduler efficiency in vLLM/TGI/SGLang-like serving.

**Source.** Yu et al. (2022); Kwon et al. (2023).

## 18. Non-Transformer Sequence Models

#### Continuous-time state space model

**Equation**

$$
x'(t)=Ax(t)+Bu(t),\quad y(t)=Cx(t)+Du(t)
$$

**Plain-English explanation.** Models sequence dynamics with a latent state.

**Symbols.** $A,B,C,D$ state-space matrices.

**Practical use case.** Foundation for S4/Mamba-style models.

**Source.** Kalman (1960); Gu et al. (2021).

#### Discrete SSM

**Equation**

$$
x_k=\bar{A}x_{k-1}+\bar{B}u_k,\quad y_k=Cx_k+Du_k
$$

**Plain-English explanation.** Discretized recurrent form of state-space model.

**Symbols.** $x_k$ hidden state at step $k$.

**Practical use case.** Efficient sequence modeling.

**Source.** Gu et al. (2021).

#### SSM convolution kernel

**Equation**

$$
y_k=\sum_{i=0}^{k}K_i u_{k-i},\quad K_i=C\bar{A}^{i}\bar{B}
$$

**Plain-English explanation.** Represents SSM recurrence as convolution.

**Symbols.** $K_i$ convolution kernel.

**Practical use case.** Parallel S4 training.

**Source.** Gu et al. (2021).

#### Mamba selective SSM

**Equation**

$$
h_t=\bar{A}_t h_{t-1}+\bar{B}_t x_t,\quad y_t=C_t h_t
$$

**Plain-English explanation.** Makes SSM parameters input-dependent.

**Symbols.** $\bar{A}_t,\bar{B}_t,C_t$ depend on token $x_t$.

**Practical use case.** Linear-time sequence modeling with content-based selection.

**Source.** Gu and Dao (2023).

#### RWKV recurrence

**Equation**

$$
\mathrm{wkv}_t=\frac{\sum_{i=1}^{t}e^{-(t-i)w+k_i}v_i}{\sum_{i=1}^{t}e^{-(t-i)w+k_i}}
$$

**Plain-English explanation.** Computes attention-like weighted values through recurrence.

**Symbols.** $w$ decay; $k_i,v_i$ key/value.

**Practical use case.** RNN-like Transformer alternative.

**Source.** Peng et al. (2023).

#### RetNet retention

**Equation**

$$
\mathrm{Retention}(Q,K,V)= (QK^\top\odot D)V
$$

**Plain-English explanation.** Uses decay matrix for causal retention.

**Symbols.** $D_{ij}=\gamma^{i-j}$ for $i\ge j$.

**Practical use case.** Parallel/recurrent long-sequence modeling.

**Source.** Sun et al. (2023).

#### Hyena implicit convolution

**Equation**

$$
y = x * h_\theta
$$

**Plain-English explanation.** Applies long convolution with an implicit filter.

**Symbols.** $*$ convolution; $h_\theta$ learned implicit filter.

**Practical use case.** Subquadratic long-context sequence modeling.

**Source.** Poli et al. (2023).

## 19. Calibration and Uncertainty

#### Expected calibration error

**Equation**

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
$$

**Plain-English explanation.** Measures mismatch between confidence and empirical accuracy.

**Symbols.** $B_m$ confidence bin.

**Practical use case.** Reliability analysis for classifiers and LLM judges.

**Source.** Naeini et al. (2015).

#### Maximum calibration error

**Equation**

$$
\mathrm{MCE}=\max_m\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
$$

**Plain-English explanation.** Worst-bin calibration gap.

**Symbols.** $B_m$ confidence bin.

**Practical use case.** Safety-sensitive calibration.

**Source.** Naeini et al. (2015).

#### Brier score

**Equation**

$$
\mathrm{BS}=\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}(p_{\mathrm{ic}}-y_{\mathrm{ic}})^2
$$

**Plain-English explanation.** Mean squared error of predicted probabilities.

**Symbols.** $p_{ic}$ predicted class probability.

**Practical use case.** Probabilistic forecast quality.

**Source.** Brier (1950).

#### Temperature scaling

**Equation**

$$
p_i=\mathrm{softmax}(z_i/T)
$$

**Plain-English explanation.** Calibrates confidence by rescaling logits.

**Symbols.** $T$ learned temperature.

**Practical use case.** Post-hoc calibration.

**Source.** Guo et al. (2017).

#### Selective risk

**Equation**

$$
R(f,g)=\frac{\mathbb{E}[\ell(f(x),y)g(x)]}{\mathbb{E}[g(x)]}
$$

**Plain-English explanation.** Risk over examples the model chooses to answer.

**Symbols.** $g(x)\in\{0,1\}$ accept/reject decision.

**Practical use case.** Abstention and human handoff.

**Source.** Geifman and El-Yaniv (2017).

#### Conformal prediction set

**Equation**

$$
C(x)=\{y:s(x,y)\leq q_{1-\alpha}\}
$$

**Plain-English explanation.** Returns a set of labels with finite-sample coverage.

**Symbols.** $s$ nonconformity score; $q_{1-\alpha}$ calibration quantile.

**Practical use case.** Uncertainty-aware prediction.

**Source.** Vovk et al. (2005).

# Part II: Retrieval-Augmented Generation

## 20. Similarity and Distance Metrics

#### Cosine similarity

**Equation**

$$
\cos(x,y)=\frac{x^\top y}{\|x\|_2\|y\|_2}
$$

**Plain-English explanation.** Measures angle similarity between vectors.

**Symbols.** $x,y$ embeddings.

**Practical use case.** Dense retrieval and reranking.

**Source.** Salton et al. (1975).

#### Dot product similarity

**Equation**

$$
s(x,y)=x^\top y
$$

**Plain-English explanation.** Measures unnormalized vector alignment.

**Symbols.** $x,y$ embeddings.

**Practical use case.** Maximum inner product search.

**Source.** Linear algebra standard.

#### Euclidean distance

**Equation**

$$
d_2(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
$$

**Plain-English explanation.** Straight-line distance between vectors.

**Symbols.** $x_i,y_i$ vector components.

**Practical use case.** Vector search and clustering.

**Source.** Euclid; metric space standard.

#### Manhattan distance

**Equation**

$$
d_1(x,y)=\sum_i|x_i-y_i|
$$

**Plain-English explanation.** Sum of absolute coordinate differences.

**Symbols.** $x_i,y_i$ components.

**Practical use case.** Robust distance metric.

**Source.** Minkowski (1896).

#### Minkowski distance

**Equation**

$$
d_p(x,y)=\left(\sum_i|x_i-y_i|^p\right)^{1/p}
$$

**Plain-English explanation.** Generalizes L1 and L2 distance.

**Symbols.** $p$ order parameter.

**Practical use case.** Similarity search and clustering.

**Source.** Minkowski (1896).

#### Jaccard similarity

**Equation**

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

**Plain-English explanation.** Compares overlap between sets.

**Symbols.** $A,B$ token/shingle sets.

**Practical use case.** Deduplication and sparse retrieval.

**Source.** Jaccard (1901).

#### Hamming distance

**Equation**

$$
d_H(x,y)=\sum_i\mathbf{1}[x_i\ne y_i]
$$

**Plain-English explanation.** Counts positions where two discrete vectors differ.

**Symbols.** $x,y$ binary/discrete strings.

**Practical use case.** Binary vector retrieval and hashing.

**Source.** Hamming (1950).

#### Mahalanobis distance

**Equation**

$$
d_M(x,y)=\sqrt{(x-y)^\top\Sigma^{-1}(x-y)}
$$

**Plain-English explanation.** Distance adjusted by covariance structure.

**Symbols.** $\Sigma$ covariance matrix.

**Practical use case.** Anomaly detection and metric learning.

**Source.** Mahalanobis (1936).

#### Angular distance

**Equation**

$$
d_{\mathrm{ang}}(x,y)=\frac{\arccos(\cos(x,y))}{\pi}
$$

**Plain-English explanation.** Converts cosine similarity to normalized angle distance.

**Symbols.** $\cos(x,y)$ cosine similarity.

**Practical use case.** Embedding retrieval distance.

**Source.** Metric geometry standard.

## 21. Sparse Retrieval

#### Term frequency

**Equation**

$$
\mathrm{tf}(t,d)=\mathrm{count}(t,d)
$$

**Plain-English explanation.** Counts how often a term appears in a document.

**Symbols.** $t$ term; $d$ document.

**Practical use case.** Sparse lexical retrieval.

**Source.** Salton and Buckley (1988).

#### Log-scaled term frequency

**Equation**

$$
\mathrm{tf}(t,d)=1+\log \mathrm{count}(t,d)
$$

**Plain-English explanation.** Dampens the effect of repeated terms.

**Symbols.** $count(t,d)\gt 0$.

**Practical use case.** TF-IDF variants.

**Source.** Salton and Buckley (1988).

#### Inverse document frequency

**Equation**

$$
\mathrm{idf}(t)=\log\frac{N}{\mathrm{df}(t)}
$$

**Plain-English explanation.** Down-weights common terms and up-weights rare terms.

**Symbols.** $N$ documents; $df(t)$ document frequency.

**Practical use case.** TF-IDF/BM25 retrieval.

**Source.** Sparck Jones (1972).

#### RSJ IDF

**Equation**

$$
\mathrm{idf}(t)=\log\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}
$$

**Plain-English explanation.** Probabilistic IDF with smoothing.

**Symbols.** $df(t)$ number of docs containing term.

**Practical use case.** BM25 scoring.

**Source.** Robertson and Sparck Jones (1976).

#### TF-IDF

**Equation**

$$
w(t,d)=\mathrm{tf}(t,d)\cdot \mathrm{idf}(t)
$$

**Plain-English explanation.** Weights a term by within-document frequency and corpus rarity.

**Symbols.** $w(t,d)$ term weight.

**Practical use case.** Classical lexical retrieval.

**Source.** Salton and Buckley (1988).

#### BM25

**Equation**

$$
\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\frac{\mathrm{tf}(t,d)(k_1+1)}{\mathrm{tf}(t,d)+k_1(1-b+b\frac{|d|}{\mathrm{avgdl}})}
$$

**Plain-English explanation.** Scores documents by saturated term frequency and length normalization.

**Symbols.** $k_1,b$ parameters; $|d|$ document length.

**Practical use case.** Default strong sparse retrieval baseline.

**Source.** Robertson et al. (1994).

#### BM25+

**Equation**

$$
\mathrm{BM25}^+(q,d)=\sum_{t\in q}\mathrm{idf}(t)\left(\frac{\mathrm{tf}(t,d)(k_1+1)}{\mathrm{tf}(t,d)+k_1(1-b+b\frac{|d|}{\mathrm{avgdl}})}+\delta\right)
$$

**Plain-English explanation.** Adds a lower-bound delta to reduce over-penalization of long docs.

**Symbols.** $\delta$ additive constant.

**Practical use case.** Sparse retrieval for variable-length documents.

**Source.** Lv and Zhai (2011).

#### BM25L

**Equation**

$$
\mathrm{BM25L}=\sum_{t\in q}\mathrm{idf}(t)\frac{(k_1+1)(c(t,d)+\delta)}{k_1+c(t,d)+\delta},\quad c(t,d)=\frac{\mathrm{tf}(t,d)}{1-b+b|d|/\mathrm{avgdl}}
$$

**Plain-English explanation.** A BM25 variant with modified term-frequency normalization.

**Symbols.** $c(t,d)$ length-normalized term frequency.

**Practical use case.** Long-document retrieval.

**Source.** Lv and Zhai (2011).

#### BM25F

**Equation**

$$
\mathrm{BM25F}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\frac{w_t(d)}{k_1+w_t(d)},\quad w_t(d)=\sum_f\frac{w_f \mathrm{tf}_{t,f}}{1-b_f+b_f|d_f|/\mathrm{avgdl}_f}
$$

**Plain-English explanation.** Combines weighted fields such as title, body, anchor text.

**Symbols.** $f$ field; $w_f$ field weight.

**Practical use case.** Structured document retrieval.

**Source.** Robertson et al. (2004).

#### Query likelihood

**Equation**

$$
\mathrm{score}(q,d)=\log P(q\mid d)=\sum_{t\in q}\mathrm{count}(t,q)\log P(t\mid d)
$$

**Plain-English explanation.** Ranks documents by probability of generating the query.

**Symbols.** $P(t\mid d)$ document language model.

**Practical use case.** Language-model retrieval.

**Source.** Ponte and Croft (1998).

#### Dirichlet smoothing

**Equation**

$$
P(t\mid d)=\frac{\mathrm{count}(t,d)+\mu P(t\mid C)}{|d|+\mu}
$$

**Plain-English explanation.** Smooths document term probabilities using collection probabilities.

**Symbols.** $\mu$ smoothing parameter; $C$ corpus.

**Practical use case.** Query likelihood retrieval.

**Source.** Zhai and Lafferty (2001).

#### Jelinek-Mercer smoothing

**Equation**

$$
P(t\mid d)=(1-\lambda)P_{\mathrm{ML}}(t\mid d)+\lambda P(t\mid C)
$$

**Plain-English explanation.** Interpolates document and collection language models.

**Symbols.** $\lambda$ smoothing weight.

**Practical use case.** Language-model retrieval.

**Source.** Zhai and Lafferty (2001).

#### Probability Ranking Principle

**Equation**

$$
d_i \succ d_j \iff P(R=1\mid d_i,q)> P(R=1\mid d_j,q)
$$

**Plain-English explanation.** Rank by probability of relevance.

**Symbols.** $R$ relevance event.

**Practical use case.** Theoretical basis of ranking.

**Source.** Robertson (1977).

#### Binary Independence Model odds

**Equation**

$$
\mathrm{score}(d,q)=\sum_{t\in q\cap d}\log\frac{p_t(1-u_t)}{u_t(1-p_t)}
$$

**Plain-English explanation.** Scores terms using relevance and non-relevance occurrence probabilities.

**Symbols.** $p_t,u_t$ term probabilities in relevant/non-relevant docs.

**Practical use case.** Foundation of probabilistic IR and BM25.

**Source.** Robertson and Sparck Jones (1976).

## 22. Dense Retrieval

#### Dual encoder score

**Equation**

$$
s(q,d)=f_\theta(q)^\top g_\phi(d)
$$

**Plain-English explanation.** Scores query and document using separately encoded embeddings.

**Symbols.** $f,g$ query/document encoders.

**Practical use case.** DPR, bi-encoder retrieval.

**Source.** Karpukhin et al. (2020).

#### DPR loss

**Equation**

$$
\mathcal{L}=-\log\frac{\exp(s(q,d^+)/\tau)}{\exp(s(q,d^+)/\tau)+\sum_{d^-}\exp(s(q,d^-)/\tau)}
$$

**Plain-English explanation.** Trains dense retriever to score positive passages above negatives.

**Symbols.** $d^+$ positive document; $d^-$ negative.

**Practical use case.** Question-answering retrieval.

**Source.** Karpukhin et al. (2020).

#### ColBERT MaxSim

**Equation**

$$
s(q,d)=\sum_{i\in q}\max_{j\in d}E_{q_i}^\top E_{d_j}
$$

**Plain-English explanation.** Late interaction: each query token matches its best document token.

**Symbols.** $E_{q_i},E_{d_j}$ token embeddings.

**Practical use case.** High-quality neural retrieval.

**Source.** Khattab and Zaharia (2020).

#### SPLADE scoring

**Equation**

$$
s(q,d)=\sum_{j=1}^{V}w_{\mathrm{qj}}w_{\mathrm{dj}}
$$

**Plain-English explanation.** Scores sparse learned lexical expansion vectors.

**Symbols.** $w_{qj},w_{dj}$ learned sparse weights.

**Practical use case.** Neural sparse retrieval.

**Source.** Formal et al. (2021).

#### SPLADE log-saturation

**Equation**

$$
w_j=\max_i \log(1+\mathrm{ReLU}(z_{\mathrm{ij}}))
$$

**Plain-English explanation.** Builds sparse vocabulary weights from MLM logits.

**Symbols.** $z_{ij}$ logit for vocab term $j$ at position $i$.

**Practical use case.** Learned expansion retrieval.

**Source.** Formal et al. (2021).

#### SPLADE FLOPS regularization

**Equation**

$$
\mathcal{L}_{\mathrm{FLOPS}}=\sum_j \left(\frac{1}{N}\sum_i w_{\mathrm{ij}}\right)^2
$$

**Plain-English explanation.** Penalizes dense usage of vocabulary dimensions.

**Symbols.** $w_{ij}$ sparse weight.

**Practical use case.** Efficiency regularization in sparse neural retrieval.

**Source.** Formal et al. (2021).

## 23. Hybrid Search and Fusion

#### Convex hybrid score

**Equation**

$$
s(q,d)=\lambda s_{\mathrm{dense}}(q,d)+(1-\lambda)s_{\mathrm{sparse}}(q,d)
$$

**Plain-English explanation.** Blends dense and sparse retrieval scores.

**Symbols.** $\lambda\in[0,1]$ fusion weight.

**Practical use case.** Hybrid search in production RAG.

**Source.** IR fusion practice.

#### Weighted normalized fusion

**Equation**

$$
s(q,d)=\sum_m w_m\frac{s_m(q,d)-\mu_m}{\sigma_m}
$$

**Plain-English explanation.** Combines normalized scores from multiple rankers.

**Symbols.** $m$ retriever/ranker index.

**Practical use case.** Combining BM25, dense, metadata, recency.

**Source.** Learning-to-rank and IR fusion literature.

#### Reciprocal rank fusion

**Equation**

$$
\mathrm{RRF}(d)=\sum_{r\in R}\frac{1}{k+\mathrm{rank}_r(d)}
$$

**Plain-English explanation.** Fuses ranked lists using reciprocal ranks.

**Symbols.** $k$ smoothing constant; $R$ rankers.

**Practical use case.** Robust hybrid search without score calibration.

**Source.** Cormack et al. (2009).

#### Learned linear fusion

**Equation**

$$
s(q,d)=w^\top\phi(q,d)
$$

**Plain-English explanation.** Learns rank score from features.

**Symbols.** $\phi$ feature vector; $w$ learned weights.

**Practical use case.** Production search ranking.

**Source.** Liu (2009).

## 24. ANN and Vector Index Math

#### k-means IVF objective

**Equation**

$$
\min_{c_1,\dots,c_K}\sum_i\min_k\|x_i-c_k\|_2^2
$$

**Plain-English explanation.** Learns coarse clusters for inverted file indexing.

**Symbols.** $c_k$ centroid.

**Practical use case.** FAISS IVF indexes.

**Source.** Lloyd (1982); Johnson et al. (2017).

#### IVF assignment

**Equation**

$$
a(x)=\mathrm{argmin}_{k}\|x-c_k\|_2
$$

**Plain-English explanation.** Assigns vector to nearest centroid.

**Symbols.** $a(x)$ cluster ID.

**Practical use case.** Approximate nearest neighbor search.

**Source.** Johnson et al. (2017).

#### Product quantization

**Equation**

$$
q(x)=[q_1(x^{(1)}),\dots,q_M(x^{(M)})]
$$

**Plain-English explanation.** Splits vector into subspaces and quantizes each subvector.

**Symbols.** $M$ number of subspaces.

**Practical use case.** Compressed vector search.

**Source.** Jégou et al. (2011).

#### PQ distance approximation

**Equation**

$$
\|q-y\|^2\approx\sum_{m=1}^{M}\|c_{m,k_m}-y^{(m)}\|^2
$$

**Plain-English explanation.** Approximates distance using sub-codebook lookups.

**Symbols.** $c_{m,k_m}$ selected centroid in subspace $m$.

**Practical use case.** Fast approximate vector search.

**Source.** Jégou et al. (2011).

#### HNSW greedy step

**Equation**

$$
v_{\mathrm{next}}=\mathrm{argmin}_{u\in N(v)}d(u,q)
$$

**Plain-English explanation.** Moves through graph neighbors toward query.

**Symbols.** $N(v)$ neighbors of node $v$.

**Practical use case.** HNSW ANN search.

**Source.** Malkov and Yashunin (2018).

#### HNSW layer probability

**Equation**

$$
P(\mathrm{level}\geq l)=e^{-l/\lambda}
$$

**Plain-English explanation.** Assigns fewer nodes to higher graph layers.

**Symbols.** $l$ layer index; $\lambda$ level parameter.

**Practical use case.** Hierarchical navigable small-world graph construction.

**Source.** Malkov and Yashunin (2018).

#### ScaNN anisotropic score

**Equation**

$$
\mathrm{score}(q,x)=q^\top x - \lambda\|x-q_{\mathrm{proj}}\|^2
$$

**Plain-English explanation.** Combines inner product with quantization-aware penalty.

**Symbols.** $q_{proj}$ projected query component.

**Practical use case.** Efficient MIPS retrieval.

**Source.** Guo et al. (2020).

## 25. Reranking and Learning to Rank

#### Cross-encoder score

**Equation**

$$
s(q,d)=w^\top h_{[CLS]}(q,d)
$$

**Plain-English explanation.** Jointly encodes query and document to produce relevance score.

**Symbols.** $h_{[CLS]}$ pooled representation.

**Practical use case.** High-accuracy reranking.

**Source.** Nogueira and Cho (2019).

#### monoT5 relevance probability

**Equation**

$$
s(q,d)=\log p_\theta(\mathrm{true}\mid q,d)
$$

**Plain-English explanation.** Scores relevance as probability of generating token true.

**Symbols.** $\theta$ seq2seq model parameters.

**Practical use case.** T5-based reranking.

**Source.** Nogueira et al. (2020).

#### Pointwise ranking loss

**Equation**

$$
\mathcal{L}=-(y\log\sigma(s)+(1-y)\log(1-\sigma(s)))
$$

**Plain-English explanation.** Treats relevance prediction as binary classification.

**Symbols.** $y$ relevance label; $s$ score.

**Practical use case.** Reranker training.

**Source.** Liu (2009).

#### Pairwise hinge ranking loss

**Equation**

$$
\mathcal{L}=\max(0,m-s(q,d^+)+s(q,d^-))
$$

**Plain-English explanation.** Requires positive doc score to exceed negative by margin.

**Symbols.** $m$ margin.

**Practical use case.** Pairwise reranker training.

**Source.** Joachims (2002).

#### RankNet loss

**Equation**

$$
\mathcal{L}= -\bar{P}_{\mathrm{ij}}\log P_{\mathrm{ij}}-(1-\bar{P}_{\mathrm{ij}})\log(1-P_{\mathrm{ij}}),\quad P_{\mathrm{ij}}=\sigma(s_i-s_j)
$$

**Plain-English explanation.** Learns pairwise ordering probabilities.

**Symbols.** $s_i,s_j$ item scores.

**Practical use case.** Learning to rank.

**Source.** Burges et al. (2005).

#### ListMLE

**Equation**

$$
\mathcal{L}=-\sum_{i=1}^{n}\log\frac{e^{s_{\pi_i}}}{\sum_{j=i}^{n}e^{s_{\pi_j}}}
$$

**Plain-English explanation.** Maximizes likelihood of the correct ranked permutation.

**Symbols.** $\pi$ ground-truth permutation.

**Practical use case.** Listwise ranking.

**Source.** Xia et al. (2008).

#### ListNet top-one

**Equation**

$$
\mathcal{L}=-\sum_i P_y(i)\log P_s(i),\quad P_s(i)=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

**Plain-English explanation.** Matches top-one probability distributions from labels and scores.

**Symbols.** $P_y$ label-induced distribution.

**Practical use case.** Listwise ranking.

**Source.** Cao et al. (2007).

#### LambdaRank gradient idea

**Equation**

$$
\lambda_{\mathrm{ij}}=\frac{-\sigma}{1+e^{\sigma(s_i-s_j)}}|\Delta \mathrm{NDCG}_{\mathrm{ij}}|
$$

**Plain-English explanation.** Scales pairwise gradients by NDCG impact.

**Symbols.** $\Delta NDCG_{ij}$ NDCG change from swapping items.

**Practical use case.** Optimizing ranking metrics.

**Source.** Burges et al. (2006).

## 26. Retrieval Metrics

#### Precision@k

**Equation**

$$
P@k=\frac{1}{k}\sum_{i=1}^{k}\mathrm{rel}_i
$$

**Plain-English explanation.** Fraction of top-k results that are relevant.

**Symbols.** $rel_i\in\{0,1\}$ relevance.

**Practical use case.** Retrieval quality measurement.

**Source.** Manning et al. (2008).

#### Recall@k

**Equation**

$$
R@k=\frac{\sum_{i=1}^{k}\mathrm{rel}_i}{\sum_{j=1}^{N}\mathrm{rel}_j}
$$

**Plain-English explanation.** Fraction of all relevant documents retrieved in top k.

**Symbols.** $N$ corpus/result set size.

**Practical use case.** RAG retrieval evaluation.

**Source.** Manning et al. (2008).

#### Hit Rate@k

**Equation**

$$
\mathrm{HR}@k=\mathbf{1}\left[\sum_{i=1}^{k}\mathrm{rel}_i> 0\right]
$$

**Plain-English explanation.** Checks whether at least one relevant document appears in top k.

**Symbols.** $rel_i$ relevance indicator.

**Practical use case.** QA retrieval evaluation.

**Source.** Manning et al. (2008).

#### MRR

**Equation**

$$
\mathrm{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\mathrm{rank}_q}
$$

**Plain-English explanation.** Averages reciprocal rank of first relevant result.

**Symbols.** $rank_q$ rank of first relevant item.

**Practical use case.** Search and QA retrieval.

**Source.** Voorhees (1999).

#### Average Precision

**Equation**

$$
\mathrm{AP}=\frac{1}{R}\sum_{k=1}^{N}P@k\cdot \mathrm{rel}_k
$$

**Plain-English explanation.** Averages precision at every relevant result.

**Symbols.** $R$ total relevant documents.

**Practical use case.** Ranking evaluation.

**Source.** Manning et al. (2008).

#### MAP

**Equation**

$$
\mathrm{MAP}=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{AP}(q)
$$

**Plain-English explanation.** Mean average precision over queries.

**Symbols.** $Q$ query set.

**Practical use case.** Retrieval benchmark metric.

**Source.** Manning et al. (2008).

#### DCG

**Equation**

$$
\mathrm{DCG}@k=\sum_{i=1}^{k}\frac{2^{\mathrm{rel}_i}-1}{\log_2(i+1)}
$$

**Plain-English explanation.** Discounted gain rewards relevant docs higher in ranking.

**Symbols.** $rel_i$ graded relevance.

**Practical use case.** Ranking evaluation.

**Source.** Järvelin and Kekäläinen (2002).

#### NDCG

**Equation**

$$
\mathrm{NDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}
$$

**Plain-English explanation.** Normalizes DCG by the ideal ranking.

**Symbols.** $IDCG$ ideal DCG.

**Practical use case.** Search/reranker evaluation.

**Source.** Järvelin and Kekäläinen (2002).

#### R-Precision

**Equation**

$$
\mathrm{R\text{-}Prec}=\frac{\mathrm{relcount}(R)}{R}
$$

**Plain-English explanation.** Precision at R, where R is number of relevant docs.

**Symbols.** $R$ relevant doc count.

**Practical use case.** IR evaluation.

**Source.** Manning et al. (2008).

## 27. Chunking and Context Packing

#### Fixed-size chunking

**Equation**

$$
C_i=x_{iS:iS+L-1}
$$

**Plain-English explanation.** Splits text into chunks of length L using stride S.

**Symbols.** $L$ chunk length; $S$ stride.

**Practical use case.** Basic RAG preprocessing.

**Source.** IR/document processing practice.

#### Sliding-window overlap

**Equation**

$$
\mathrm{overlap}=L-S
$$

**Plain-English explanation.** Defines repeated tokens between adjacent chunks.

**Symbols.** $S\lt L$ creates overlap.

**Practical use case.** Avoiding boundary information loss.

**Source.** IR/document processing practice.

#### Semantic chunk boundary

**Equation**

$$
b_i=\mathbf{1}[1-\cos(e_i,e_{i+1})> \tau]
$$

**Plain-English explanation.** Creates a boundary when adjacent sentence embeddings differ enough.

**Symbols.** $e_i$ sentence embedding; $\tau$ threshold.

**Practical use case.** Semantic chunking.

**Source.** Modern RAG engineering practice.

#### Token-budgeted context knapsack

**Equation**

$$
\max_{S}\sum_{i\in S}u_i\quad\mathrm{s.t.}\quad\sum_{i\in S}l_i\leq B
$$

**Plain-English explanation.** Selects chunks maximizing utility under token budget.

**Symbols.** $u_i$ utility; $l_i$ token length; $B$ budget.

**Practical use case.** Packing retrieved context into limited prompt length.

**Source.** Knapsack optimization standard.

#### Lost-in-the-middle weighting

**Equation**

$$
w(\mathrm{pos})=\alpha+\beta\left|2\frac{\mathrm{pos}}{L}-1\right|
$$

**Plain-English explanation.** Models higher attention/usefulness near beginning and end than middle.

**Symbols.** $pos$ chunk position in context.

**Practical use case.** Ordering retrieved chunks in prompts.

**Source.** Liu et al. (2024).

#### Context compression score

**Equation**

$$
\mathrm{score}(c)=\lambda \mathrm{rel}(q,c)-(1-\lambda)\mathrm{len}(c)
$$

**Plain-English explanation.** Balances relevance against token cost.

**Symbols.** $rel$ relevance; $len$ token length.

**Practical use case.** RAG compression and passage selection.

**Source.** RAG engineering practice.

## 28. Diversity and Redundancy Control

#### MMR

**Equation**

$$
\mathrm{MMR}(d)=\lambda \mathrm{Sim}_1(d,q)-(1-\lambda)\max_{d'\in S}\mathrm{Sim}_2(d,d')
$$

**Plain-English explanation.** Selects relevant but non-redundant documents.

**Symbols.** $S$ already selected docs.

**Practical use case.** Diversified retrieval for RAG.

**Source.** Carbonell and Goldstein (1998).

#### DPP probability

**Equation**

$$
P(S)\propto\det(L_S)
$$

**Plain-English explanation.** Gives high probability to diverse subsets.

**Symbols.** $L_S$ kernel submatrix for selected set $S$.

**Practical use case.** Diverse context selection.

**Source.** Kulesza and Taskar (2012).

#### DPP marginal kernel

**Equation**

$$
K=L(L+I)^{-1}
$$

**Plain-English explanation.** Converts DPP L-ensemble kernel to marginal kernel.

**Symbols.** $I$ identity matrix.

**Practical use case.** Diversity modeling.

**Source.** Kulesza and Taskar (2012).

## 29. RAG Objectives and RAGAS-Style Metrics

#### RAG-Sequence likelihood

**Equation**

$$
p(y\mid x)=\sum_{z\in \mathrm{topk}(p_\eta(\cdot\mid x))}p_\eta(z\mid x)p_\theta(y\mid x,z)
$$

**Plain-English explanation.** Marginalizes answer likelihood over retrieved documents.

**Symbols.** $z$ retrieved passage; $p_\eta$ retriever; $p_\theta$ generator.

**Practical use case.** End-to-end RAG training.

**Source.** Lewis et al. (2020).

#### RAG-Token likelihood

**Equation**

$$
p(y\mid x)=\prod_{i=1}^{N}\sum_z p_\eta(z\mid x,y_{1:i-1})p_\theta(y_i\mid x,z,y_{1:i-1})
$$

**Plain-English explanation.** Allows retrieval distribution to vary per generated token.

**Symbols.** $y_i$ answer token.

**Practical use case.** Token-level retrieval-augmented generation.

**Source.** Lewis et al. (2020).

#### Context precision

**Equation**

$$
\mathrm{ContextPrecision}@k=\frac{1}{k}\sum_{i=1}^{k}\mathbf{1}[\mathrm{relevant}(c_i)]
$$

**Plain-English explanation.** Measures fraction of retrieved contexts that are relevant.

**Symbols.** $c_i$ retrieved context.

**Practical use case.** RAG evaluation.

**Source.** RAGAS framework; Es et al. (2023).

#### Context recall

**Equation**

$$
\mathrm{ContextRecall}=\frac{|\{\text{grounded facts covered by contexts}\}|}{|\{\text{ground-truth facts}\}|}
$$

**Plain-English explanation.** Measures how much required evidence is retrieved.

**Symbols.** Facts are atomic answer-supporting claims.

**Practical use case.** RAG evidence coverage.

**Source.** RAGAS framework; Es et al. (2023).

#### Faithfulness

**Equation**

$$
\mathrm{Faithfulness}=\frac{|\{\text{claims supported by context}\}|}{|\{\text{claims in answer}\}|}
$$

**Plain-English explanation.** Measures whether answer claims are supported by context.

**Symbols.** Claims are extracted atomic statements.

**Practical use case.** Hallucination detection in RAG.

**Source.** RAGAS framework; Es et al. (2023).

#### Answer relevance

**Equation**

$$
\mathrm{AnswerRelevance}=\frac{1}{m}\sum_{i=1}^{m}\cos(e_{q},e_{q_i'})
$$

**Plain-English explanation.** Measures whether generated answer addresses the original question via generated reverse questions.

**Symbols.** $e_q$ original question embedding; $q_i'$ generated question.

**Practical use case.** RAG answer quality evaluation.

**Source.** RAGAS framework; Es et al. (2023).

#### Attribution precision

**Equation**

$$
\mathrm{AttrPrecision}=\frac{|\{\text{cited claims supported}\}|}{|\{\text{cited claims}\}|}
$$

**Plain-English explanation.** Measures correctness of citations/attributions.

**Symbols.** Cited claims are answer claims with source references.

**Practical use case.** Grounded answer citation evaluation.

**Source.** RAG evaluation practice.

## 30. Query Expansion and Feedback

#### Rocchio update

**Equation**

$$
q'=\alpha q+\frac{\beta}{|D_r|}\sum_{d\in D_r}d-\frac{\gamma}{|D_{\mathrm{nr}}|}\sum_{d\in D_{\mathrm{nr}}}d
$$

**Plain-English explanation.** Moves query vector toward relevant docs and away from non-relevant docs.

**Symbols.** $D_r,D_{nr}$ relevant/non-relevant docs.

**Practical use case.** Relevance feedback.

**Source.** Rocchio (1971).

#### RM3 relevance model

**Equation**

$$
P(w\mid R)=\sum_{d\in D_{\mathrm{PRF}}}P(w\mid d)P(d\mid q)
$$

**Plain-English explanation.** Builds expansion term distribution from pseudo-relevant docs.

**Symbols.** $D_{PRF}$ top retrieved docs.

**Practical use case.** Pseudo-relevance feedback.

**Source.** Lavrenko and Croft (2001); Abdul-Jaleel et al. (2004).

#### RM3 interpolation

**Equation**

$$
P'(w\mid q)=\lambda P(w\mid q)+(1-\lambda)P(w\mid R)
$$

**Plain-English explanation.** Combines original query model with feedback model.

**Symbols.** $\lambda$ interpolation coefficient.

**Practical use case.** Query expansion.

**Source.** Abdul-Jaleel et al. (2004).

#### HyDE retrieval

**Equation**

$$
d^*=\mathrm{argmax}_{d\in D}\cos(f(h_\theta(q)),g(d))
$$

**Plain-English explanation.** Generates a hypothetical answer/document and retrieves similar real documents.

**Symbols.** $h_\theta(q)$ hypothetical document generated from query.

**Practical use case.** Zero-shot dense retrieval improvement.

**Source.** Gao et al. (2023).

## 31. GraphRAG

#### Graph adjacency

**Equation**

$$
A_{\mathrm{ij}}=w(e_i,e_j)
$$

**Plain-English explanation.** Represents weighted relationships between extracted entities.

**Symbols.** $e_i,e_j$ entities; $w$ edge weight.

**Practical use case.** Entity graph construction.

**Source.** Graph retrieval literature; Edge et al. (2024).

#### PageRank

**Equation**

$$
\mathrm{PR}(v)=\frac{1-d}{N}+d\sum_{u\in \mathrm{In}(v)}\frac{\mathrm{PR}(u)}{\mathrm{out}(u)}
$$

**Plain-English explanation.** Scores graph nodes by incoming importance.

**Symbols.** $d$ damping factor.

**Practical use case.** Ranking entities or documents in GraphRAG.

**Source.** Page et al. (1999).

#### Personalized PageRank

**Equation**

$$
p=(1-\alpha)v+\alpha P^\top p
$$

**Plain-English explanation.** PageRank biased toward a personalization/query vector.

**Symbols.** $v$ personalization distribution; $P$ transition matrix.

**Practical use case.** Query-specific graph retrieval.

**Source.** Jeh and Widom (2003).

#### Modularity

**Equation**

$$
Q=\frac{1}{2m}\sum_{\mathrm{ij}}\left(A_{\mathrm{ij}}-\frac{k_ik_j}{2m}\right)\mathbf{1}[c_i=c_j]
$$

**Plain-English explanation.** Measures quality of graph community partition.

**Symbols.** $k_i$ degree; $c_i$ community label.

**Practical use case.** Community detection for GraphRAG summaries.

**Source.** Newman (2006).

#### Path relevance

**Equation**

$$
\mathrm{score}(\mathrm{path})=\prod_{(i,j)\in \mathrm{path}}w_{\mathrm{ij}}
$$

**Plain-English explanation.** Scores multi-hop graph paths by edge weights.

**Symbols.** $w_{ij}$ edge relevance weight.

**Practical use case.** Multi-hop reasoning over entity graphs.

**Source.** Graph search standard.

#### Graph context aggregation

**Equation**

$$
h_v^{(l+1)}=\sigma\left(W_{\mathrm{self}} h_v^{(l)}+\sum_{u\in N(v)}\alpha_{\mathrm{uv}}W_{\mathrm{neigh}} h_u^{(l)}\right)
$$

**Plain-English explanation.** Aggregates neighboring node representations.

**Symbols.** $N(v)$ neighbors; $\alpha_{uv}$ edge attention/weight.

**Practical use case.** Graph neural retrieval and GraphRAG ranking.

**Source.** Kipf and Welling (2017); Veličković et al. (2018).

# Part III: AI Agents

## 32. MDP and POMDP Foundations

#### MDP definition

**Equation**

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)
$$

**Plain-English explanation.** Defines states, actions, transition dynamics, rewards, and discount.

**Symbols.** $\mathcal{S}$ states; $\mathcal{A}$ actions.

**Practical use case.** Formal agent environment modeling.

**Source.** Bellman (1957); Sutton and Barto (2018).

#### Transition probability

**Equation**

$$
P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a)
$$

**Plain-English explanation.** Probability of next state given current state and action.

**Symbols.** $s,s'$ states; $a$ action.

**Practical use case.** Planning and RL.

**Source.** Bellman (1957).

#### Expected return

**Equation**

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
$$

**Plain-English explanation.** Discounted sum of future rewards.

**Symbols.** $\gamma$ discount factor.

**Practical use case.** RL objective.

**Source.** Sutton and Barto (2018).

#### State value

**Equation**

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
$$

**Plain-English explanation.** Expected return starting from state under policy.

**Symbols.** $\pi$ policy.

**Practical use case.** Policy evaluation.

**Source.** Bellman (1957).

#### Action value

**Equation**

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
$$

**Plain-English explanation.** Expected return after taking action then following policy.

**Symbols.** $a$ action.

**Practical use case.** Action selection and Q-learning.

**Source.** Watkins (1989).

#### Bellman expectation for V

**Equation**

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^\pi(s')]
$$

**Plain-English explanation.** Recursive value equation under a fixed policy.

**Symbols.** $R$ reward function.

**Practical use case.** Policy evaluation.

**Source.** Bellman (1957).

#### Bellman optimality for V

**Equation**

$$
V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^*(s')]
$$

**Plain-English explanation.** Optimal value equals best expected one-step reward plus future value.

**Symbols.** $V^*$ optimal value.

**Practical use case.** Planning and value iteration.

**Source.** Bellman (1957).

#### Bellman optimality for Q

**Equation**

$$
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')]
$$

**Plain-English explanation.** Optimal action value recursively uses the best next action.

**Symbols.** $Q^*$ optimal action-value function.

**Practical use case.** Q-learning and planning.

**Source.** Bellman (1957); Watkins (1989).

#### POMDP

**Equation**

$$
\mathcal{P}=(\mathcal{S},\mathcal{A},T,R,\Omega,O,\gamma)
$$

**Plain-English explanation.** Extends MDP with hidden state and observations.

**Symbols.** $\Omega$ observations; $O$ observation model.

**Practical use case.** Partially observable agents and dialogue systems.

**Source.** Kaelbling et al. (1998).

#### Belief update

**Equation**

$$
b'(s')=\eta O(o'\mid s',a)\sum_s T(s'\mid s,a)b(s)
$$

**Plain-English explanation.** Updates probability over hidden states after action and observation.

**Symbols.** $b$ belief state; $\eta$ normalizer.

**Practical use case.** POMDP planning.

**Source.** Kaelbling et al. (1998).

## 33. Temporal-Difference and Value-Based RL

#### TD error

**Equation**

$$
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)
$$

**Plain-English explanation.** One-step prediction error for value estimates.

**Symbols.** $r_{t+1}$ reward.

**Practical use case.** TD learning and advantage estimation.

**Source.** Sutton (1988).

#### TD(0) update

**Equation**

$$
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
$$

**Plain-English explanation.** Updates value toward one-step bootstrapped target.

**Symbols.** $\alpha$ learning rate.

**Practical use case.** Policy evaluation.

**Source.** Sutton (1988).

#### Q-learning

**Equation**

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+\gamma\max_aQ(s_{t+1},a)-Q(s_t,a_t)]
$$

**Plain-English explanation.** Off-policy TD control using greedy next action.

**Symbols.** $\alpha$ learning rate.

**Practical use case.** Learning optimal action values.

**Source.** Watkins (1989).

#### SARSA

**Equation**

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
$$

**Plain-English explanation.** On-policy TD control using actually selected next action.

**Symbols.** $a_{t+1}$ next policy action.

**Practical use case.** Safer on-policy learning.

**Source.** Rummery and Niranjan (1994).

#### Expected SARSA

**Equation**

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+\gamma\sum_a\pi(a\mid s_{t+1})Q(s_{t+1},a)-Q(s_t,a_t)]
$$

**Plain-English explanation.** Uses expected next value under policy.

**Symbols.** $\pi$ behavior/target policy.

**Practical use case.** Lower variance TD control.

**Source.** van Seijen et al. (2009).

#### Double Q-learning target

**Equation**

$$
Y=r+\gamma Q_{\theta^-}(s',\mathrm{argmax}_{a} Q_\theta(s',a))
$$

**Plain-English explanation.** Uses online network for action selection and target network for evaluation.

**Symbols.** $\theta^-$ target network parameters.

**Practical use case.** Reduces overestimation bias.

**Source.** van Hasselt et al. (2010); van Hasselt et al. (2016).

#### DQN loss

**Equation**

$$
\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')}[(r+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a))^2]
$$

**Plain-English explanation.** Trains neural Q-function against bootstrapped targets.

**Symbols.** $\theta^-$ target network.

**Practical use case.** Deep reinforcement learning.

**Source.** Mnih et al. (2015).

#### Dueling DQN

**Equation**

$$
Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')
$$

**Plain-English explanation.** Decomposes Q into state value and action advantage.

**Symbols.** $A$ advantage stream.

**Practical use case.** Better value estimation when many actions have similar value.

**Source.** Wang et al. (2016).

## 34. Policy Gradient and Actor-Critic

#### Policy gradient theorem

**Equation**

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta(a\mid s)Q^{\pi}(s,a)]
$$

**Plain-English explanation.** Gradient of expected return can be estimated from action log-prob gradients.

**Symbols.** $J$ expected return.

**Practical use case.** Policy optimization.

**Source.** Sutton et al. (1999).

#### REINFORCE

**Equation**

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right]
$$

**Plain-English explanation.** Monte Carlo policy gradient estimator.

**Symbols.** $G_t$ return.

**Practical use case.** Policy learning without value function.

**Source.** Williams (1992).

#### Baseline subtraction

**Equation**

$$
\nabla_\theta J=\mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)(Q(s,a)-b(s))]
$$

**Plain-English explanation.** Subtracts baseline without biasing gradient.

**Symbols.** $b(s)$ baseline.

**Practical use case.** Variance reduction.

**Source.** Williams (1992); Sutton et al. (1999).

#### Actor-critic advantage update

**Equation**

$$
\nabla_\theta J\approx \nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t
$$

**Plain-English explanation.** Policy gradient weighted by estimated advantage.

**Symbols.** $\hat{A}_t$ advantage estimate.

**Practical use case.** A2C/A3C/PPO.

**Source.** Konda and Tsitsiklis (2000).

#### A3C n-step target

**Equation**

$$
R_t=\sum_{i=0}^{n-1}\gamma^ir_{t+i}+\gamma^nV(s_{t+n})
$$

**Plain-English explanation.** Uses n-step bootstrapped return.

**Symbols.** $n$ rollout length.

**Practical use case.** Asynchronous actor-critic.

**Source.** Mnih et al. (2016).

#### TRPO constraint

**Equation**

$$
\max_\theta\mathbb{E}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}A_{\mathrm{old}}(s,a)\right]\quad\mathrm{s.t.}\quad \mathbb{E}[D_{\mathrm{KL}}(\pi_{\mathrm{old}}\Vert\pi_\theta)]\le\delta
$$

**Plain-English explanation.** Maximizes surrogate objective under KL trust region.

**Symbols.** $\delta$ trust-region radius.

**Practical use case.** Stable policy optimization.

**Source.** Schulman et al. (2015).

#### DDPG critic loss

**Equation**

$$
\mathcal{L}=\mathbb{E}[(Q_\phi(s,a)-(r+\gamma Q_{\phi'}(s',\mu_{\theta'}(s'))))^2]
$$

**Plain-English explanation.** Learns Q-function for deterministic continuous-control actor.

**Symbols.** $\mu$ deterministic policy.

**Practical use case.** Continuous action RL.

**Source.** Lillicrap et al. (2016).

#### Deterministic policy gradient

**Equation**

$$
\nabla_\theta J\approx \mathbb{E}[\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)]
$$

**Plain-English explanation.** Updates deterministic actor through critic gradient.

**Symbols.** $\mu_\theta$ actor.

**Practical use case.** DDPG/TD3.

**Source.** Silver et al. (2014).

#### TD3 target

**Equation**

$$
y=r+\gamma\min_{i=1,2}Q_{\phi_i'}(s',\mu_{\theta'}(s')+\epsilon)
$$

**Plain-English explanation.** Uses clipped double critics and target policy smoothing.

**Symbols.** $\epsilon$ clipped noise.

**Practical use case.** Reducing overestimation in continuous RL.

**Source.** Fujimoto et al. (2018).

#### SAC objective

**Equation**

$$
J(\pi)=\mathbb{E}\left[\sum_t\gamma^t(r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot\mid s_t)))\right]
$$

**Plain-English explanation.** Maximizes reward plus entropy.

**Symbols.** $\alpha$ entropy temperature.

**Practical use case.** Robust stochastic control.

**Source.** Haarnoja et al. (2018).

#### SAC soft value

**Equation**

$$
V(s)=\mathbb{E}_{a\sim\pi}[Q(s,a)-\alpha\log\pi(a\mid s)]
$$

**Plain-English explanation.** State value includes entropy bonus.

**Symbols.** $\alpha$ entropy coefficient.

**Practical use case.** Soft actor-critic.

**Source.** Haarnoja et al. (2018).

#### IMPALA V-trace

**Equation**

$$
v_s=V(x_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV
$$

**Plain-English explanation.** Corrects off-policy trajectories with truncated importance sampling.

**Symbols.** $c_i$ clipped importance weights.

**Practical use case.** Distributed RL.

**Source.** Espeholt et al. (2018).

## 35. Advantage Estimation

#### Advantage

**Equation**

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

**Plain-English explanation.** Measures how much better an action is than average at a state.

**Symbols.** $Q,V$ action/state values.

**Practical use case.** Policy gradient and PPO.

**Source.** Baird (1993); Sutton and Barto (2018).

#### Monte Carlo return

**Equation**

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

**Plain-English explanation.** Actual discounted return from a trajectory.

**Symbols.** $T$ episode end.

**Practical use case.** REINFORCE and evaluation.

**Source.** Sutton and Barto (2018).

#### n-step return

**Equation**

$$
G_t^{(n)}=\sum_{i=0}^{n-1}\gamma^ir_{t+i+1}+\gamma^nV(s_{t+n})
$$

**Plain-English explanation.** Combines sampled rewards with bootstrapped value.

**Symbols.** $n$ horizon.

**Practical use case.** A3C, PPO rollouts, TD learning.

**Source.** Sutton and Barto (2018).

#### TD-lambda return

**Equation**

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

**Plain-English explanation.** Weighted mixture of n-step returns.

**Symbols.** $\lambda$ trace parameter.

**Practical use case.** Bias-variance trade-off.

**Source.** Sutton (1988).

## 36. Exploration and Bandits

#### Epsilon-greedy

**Equation**

$$
a=\begin{cases}\mathrm{argmax}_{a} Q(s,a),&1-\epsilon\\\text{random action},&\epsilon\end{cases}
$$

**Plain-English explanation.** Usually exploits best action, sometimes explores randomly.

**Symbols.** $\epsilon$ exploration probability.

**Practical use case.** Simple RL exploration.

**Source.** Sutton and Barto (2018).

#### Boltzmann exploration

**Equation**

$$
P(a\mid s)=\frac{\exp(Q(s,a)/\tau)}{\sum_b\exp(Q(s,b)/\tau)}
$$

**Plain-English explanation.** Samples actions according to softmax Q-values.

**Symbols.** $\tau$ temperature.

**Practical use case.** Smooth exploration.

**Source.** Sutton and Barto (2018).

#### UCB1

**Equation**

$$
a_t=\mathrm{argmax}_{a}\left[\hat{\mu}_a+c\sqrt{\frac{\ln t}{N_a}}\right]
$$

**Plain-English explanation.** Chooses action by reward estimate plus uncertainty bonus.

**Symbols.** $N_a$ action count; $c$ exploration coefficient.

**Practical use case.** Bandit exploration.

**Source.** Auer et al. (2002).

#### Cumulative regret

**Equation**

$$
R_T=T\mu^*-\sum_{t=1}^{T}\mu_{a_t}
$$

**Plain-English explanation.** Measures reward lost relative to optimal action.

**Symbols.** $\mu^*$ best expected reward.

**Practical use case.** Bandit and RL evaluation.

**Source.** Lai and Robbins (1985).

#### Thompson sampling

**Equation**

$$
a_t=\mathrm{argmax}_{a} \tilde{\mu}_a,\quad \tilde{\mu}_a\sim P(\mu_a\mid D_t)
$$

**Plain-English explanation.** Samples beliefs about action value and acts greedily under the sample.

**Symbols.** $D_t$ observations so far.

**Practical use case.** Bayesian exploration.

**Source.** Thompson (1933).

#### LinUCB

**Equation**

$$
a_t=\mathrm{argmax}_{a}\left[x_{t,a}^\top\hat{\theta}_a+\alpha\sqrt{x_{t,a}^\top A_a^{-1}x_{t,a}}\right]
$$

**Plain-English explanation.** Contextual UCB for linear reward models.

**Symbols.** $x_{t,a}$ context-action features.

**Practical use case.** Tool selection and recommendation bandits.

**Source.** Li et al. (2010).

#### EXP3

**Equation**

$$
p_{t,i}=(1-\gamma)\frac{w_{t,i}}{\sum_jw_{t,j}}+\frac{\gamma}{K}
$$

**Plain-English explanation.** Adversarial bandit action distribution.

**Symbols.** $w_{t,i}$ action weight.

**Practical use case.** Robust exploration under adversarial rewards.

**Source.** Auer et al. (2002).

#### RND intrinsic reward

**Equation**

$$
r_t^{\mathrm{int}}=\|f_\theta(s_t)-f_{\mathrm{target}}(s_t)\|_2^2
$$

**Plain-English explanation.** Rewards states where predictor poorly matches fixed random target.

**Symbols.** $f_\theta$ predictor; $f_{target}$ fixed random network.

**Practical use case.** Exploration in sparse reward environments.

**Source.** Burda et al. (2019).

#### ICM curiosity reward

**Equation**

$$
r_t^{\mathrm{int}}=\|\hat{\phi}(s_{t+1})-\phi(s_{t+1})\|_2^2
$$

**Plain-English explanation.** Rewards prediction error in learned feature space.

**Symbols.** $\phi$ state feature encoder.

**Practical use case.** Intrinsic motivation.

**Source.** Pathak et al. (2017).

## 37. Planning and Tree Search

#### Value iteration

**Equation**

$$
V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V_k(s')]
$$

**Plain-English explanation.** Repeated Bellman optimality backups.

**Symbols.** $k$ iteration index.

**Practical use case.** Planning with known model.

**Source.** Bellman (1957).

#### Policy evaluation

**Equation**

$$
V^{\pi}_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)[R+\gamma V_k^{\pi}(s')]
$$

**Plain-English explanation.** Computes value for a fixed policy.

**Symbols.** $\pi$ policy.

**Practical use case.** Policy iteration.

**Source.** Howard (1960).

#### Policy improvement

**Equation**

$$
\pi_{\mathrm{new}}(s)=\mathrm{argmax}_{a}\sum_{s'}P(s'\mid s,a)[R+\gamma V^\pi(s')]
$$

**Plain-English explanation.** Updates policy greedily with respect to current value.

**Symbols.** $V^\pi$ evaluated policy value.

**Practical use case.** Policy iteration.

**Source.** Howard (1960).

#### MCTS UCB1

**Equation**

$$
a=\mathrm{argmax}_{a}\left[Q(s,a)+c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right]
$$

**Plain-English explanation.** Balances exploitation and exploration in tree search.

**Symbols.** $N(s)$ state visits; $N(s,a)$ action visits.

**Practical use case.** Monte Carlo Tree Search.

**Source.** Kocsis and Szepesvári (2006).

#### AlphaZero PUCT

**Equation**

$$
a=\mathrm{argmax}_{a}\left[Q(s,a)+c_{\mathrm{puct}}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}\right]
$$

**Plain-English explanation.** Uses policy prior to guide MCTS exploration.

**Symbols.** $P(s,a)$ prior probability.

**Practical use case.** AlphaZero-style planning and Tree-of-Thought variants.

**Source.** Silver et al. (2017).

#### Model predictive control

**Equation**

$$
a_t=\mathrm{argmax}_{a_{t:t+H}}\mathbb{E}\left[\sum_{k=0}^{H}\gamma^kr_{t+k}\right]
$$

**Plain-English explanation.** Chooses first action from best finite-horizon plan.

**Symbols.** $H$ planning horizon.

**Practical use case.** Model-based agents.

**Source.** Control theory standard.

## 38. Offline RL and Imitation

#### Behavior cloning

**Equation**

$$
\mathcal{L}_{\mathrm{BC}}=-\mathbb{E}_{(s,a)\sim D}\log\pi_\theta(a\mid s)
$$

**Plain-English explanation.** Imitates actions from logged demonstrations.

**Symbols.** $D$ offline dataset.

**Practical use case.** Tool-use imitation and agent bootstrapping.

**Source.** Pomerleau (1989).

#### Importance sampling OPE

**Equation**

$$
V^{\pi}=\mathbb{E}_{\tau\sim\mu}\left[\prod_t\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}G(\tau)\right]
$$

**Plain-English explanation.** Evaluates target policy using data from behavior policy.

**Symbols.** $\mu$ behavior policy.

**Practical use case.** Offline policy evaluation.

**Source.** Precup (2000).

#### Weighted importance sampling

**Equation**

$$
\hat{V}_{\mathrm{WIS}}=\frac{\sum_i\rho_iG_i}{\sum_i\rho_i}
$$

**Plain-English explanation.** Normalizes importance weights to reduce variance.

**Symbols.** $\rho_i$ trajectory likelihood ratio.

**Practical use case.** Offline RL evaluation.

**Source.** Precup (2000).

#### CQL penalty

**Equation**

$$
\mathcal{L}_{\mathrm{CQL}}=\alpha\left(\mathbb{E}_{s}\log\sum_a e^{Q(s,a)}-\mathbb{E}_{(s,a)\sim D}Q(s,a)\right)+\mathcal{L}_{\mathrm{Bellman}}
$$

**Plain-English explanation.** Penalizes high Q-values for unseen actions.

**Symbols.** $D$ offline dataset.

**Practical use case.** Conservative offline RL.

**Source.** Kumar et al. (2020).

#### IQL expectile loss

**Equation**

$$
\mathcal{L}_V=\mathbb{E}_{(s,a)\sim D}[L_2^\tau(Q(s,a)-V(s))]
$$

**Plain-English explanation.** Fits value to an expectile of Q-values.

**Symbols.** $L_2^\tau(u)=|\tau-\mathbf{1}[u\lt 0]|u^2$.

**Practical use case.** Implicit Q-learning from offline data.

**Source.** Kostrikov et al. (2021).

#### Advantage-weighted regression

**Equation**

$$
\mathcal{L}_{\mathrm{AWR}}=-\mathbb{E}_{(s,a)\sim D}\left[\exp\left(\frac{A(s,a)}{\beta}\right)\log\pi_\theta(a\mid s)\right]
$$

**Plain-English explanation.** Weights imitation by estimated advantage.

**Symbols.** $\beta$ temperature.

**Practical use case.** Offline RL and preference-weighted imitation.

**Source.** Peng et al. (2019).

## 39. Multi-Agent RL

#### Nash equilibrium

**Equation**

$$
\pi_i^*\in\mathrm{argmax}_{\pi_i}J_i(\pi_i,\pi_{-i}^*)
$$

**Plain-English explanation.** No agent can improve by unilaterally changing policy.

**Symbols.** $\pi_{-i}$ other agents' policies.

**Practical use case.** Game-theoretic agent analysis.

**Source.** Nash (1950).

#### Minimax value

**Equation**

$$
V^*(s)=\max_a\min_b Q(s,a,b)
$$

**Plain-English explanation.** Chooses action maximizing worst-case outcome against opponent.

**Symbols.** $b$ opponent action.

**Practical use case.** Competitive games.

**Source.** von Neumann (1928).

#### VDN

**Equation**

$$
Q_{\mathrm{tot}}(\tau,a)=\sum_{i=1}^{n}Q_i(\tau_i,a_i)
$$

**Plain-English explanation.** Factorizes team Q-value as sum of agent Q-values.

**Symbols.** $\tau_i$ local history.

**Practical use case.** Cooperative multi-agent RL.

**Source.** Sunehag et al. (2018).

#### QMIX monotonicity

**Equation**

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}\ge0
$$

**Plain-English explanation.** Constrains mixing network so local argmax actions align with global argmax.

**Symbols.** $Q_i$ individual agent Q.

**Practical use case.** Cooperative MARL.

**Source.** Rashid et al. (2018).

#### MADDPG critic

**Equation**

$$
\nabla_{\theta_i}J\approx\mathbb{E}[\nabla_{\theta_i}\mu_i(o_i)\nabla_{a_i}Q_i(x,a_1,\dots,a_N)]
$$

**Plain-English explanation.** Centralized critic trains decentralized actors.

**Symbols.** $o_i$ local observation; $x$ global state.

**Practical use case.** Multi-agent continuous control.

**Source.** Lowe et al. (2017).

#### Potential-based reward shaping

**Equation**

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
$$

**Plain-English explanation.** Adds shaping reward without changing optimal policy.

**Symbols.** $\Phi$ potential function.

**Practical use case.** Guiding agent learning safely.

**Source.** Ng et al. (1999).

## 40. LLM-Agent Specific Math

#### Tool selection policy

**Equation**

$$
\pi_\theta(\mathrm{tool}\mid \mathrm{context})=\mathrm{softmax}(f_\theta(\mathrm{context}))
$$

**Plain-English explanation.** Chooses a tool/action from context.

**Symbols.** $tool$ available external action.

**Practical use case.** Tool-using LLM agents.

**Source.** LLM agent literature; Schick et al. (2023).

#### Tool argument likelihood

**Equation**

$$
p_\theta(\mathrm{args}\mid \mathrm{tool},\mathrm{context})=\prod_t p_\theta(a_t\mid \mathrm{tool},\mathrm{context},a_{1:t-1})
$$

**Plain-English explanation.** Generates structured arguments token by token.

**Symbols.** $a_t$ argument token.

**Practical use case.** Function calling and API agents.

**Source.** Autoregressive modeling standard.

#### Expected utility of tool use

**Equation**

$$
U(\mathrm{tool})=\mathbb{E}[R\mid \mathrm{tool},\mathrm{context}]-\lambda \mathrm{Cost}(\mathrm{tool})
$$

**Plain-English explanation.** Selects tools by value minus cost.

**Symbols.** $R$ expected reward; $Cost$ latency/money/risk.

**Practical use case.** Agent routing and tool budgeting.

**Source.** Decision theory standard.

#### ReAct trajectory

**Equation**

$$
p(\tau\mid x)=\prod_t p_\theta(\mathrm{thought}_t,\mathrm{action}_t\mid x,\mathrm{obs}_{1:t-1},\mathrm{action}_{1:t-1})p(\mathrm{obs}_t\mid \mathrm{action}_t)
$$

**Plain-English explanation.** Models reasoning/action/observation loop as a trajectory.

**Symbols.** $\tau$ trajectory.

**Practical use case.** ReAct-style agents.

**Source.** Yao et al. (2023).

#### Tree of Thoughts path score

**Equation**

$$
S(\mathrm{path})=\sum_{t=1}^{T}v_\theta(\mathrm{thought}_t\mid x,\mathrm{thought}_{1:t-1})
$$

**Plain-English explanation.** Scores a reasoning path by evaluator values.

**Symbols.** $v_\theta$ thought evaluator.

**Practical use case.** Tree search over reasoning steps.

**Source.** Yao et al. (2023).

#### Self-consistency vote

**Equation**

$$
\hat{y}=\mathrm{argmax}_{y}\sum_{i=1}^{N}\mathbf{1}[y_i=y]
$$

**Plain-English explanation.** Samples multiple reasoning paths and chooses majority answer.

**Symbols.** $N$ samples.

**Practical use case.** Improving reasoning reliability.

**Source.** Wang et al. (2023).

#### Weighted self-consistency

**Equation**

$$
\hat{y}=\mathrm{argmax}_{y}\sum_i w_i\mathbf{1}[y_i=y]
$$

**Plain-English explanation.** Majority vote weighted by confidence or verifier score.

**Symbols.** $w_i$ sample weight.

**Practical use case.** Verifier-guided reasoning.

**Source.** Reasoning ensemble practice.

#### Verifier-guided decoding

**Equation**

$$
y^*=\mathrm{argmax}_{y\in\mathcal{Y}}[\log p_\theta(y\mid x)+\lambda v_\phi(x,y)]
$$

**Plain-English explanation.** Combines generator likelihood with verifier score.

**Symbols.** $v_\phi$ verifier/reward model.

**Practical use case.** Math/code reasoning reranking.

**Source.** Cobbe et al. (2021).

#### Reflexion memory update

**Equation**

$$
M_{t+1}=M_t\cup\{\rho_t\},\quad \rho_t=f_\theta(\mathrm{trajectory}_t,\mathrm{feedback}_t)
$$

**Plain-English explanation.** Adds reflection to memory after feedback.

**Symbols.** $\rho_t$ reflection text.

**Practical use case.** Self-improving agents across trials.

**Source.** Shinn et al. (2023).

#### Retry value

**Equation**

$$
\mathrm{Retry}=\mathbf{1}[P(\mathrm{success}\mid \mathrm{retry})\cdot V-C_{\mathrm{retry}}> 0]
$$

**Plain-English explanation.** Retries only when expected value exceeds cost.

**Symbols.** $V$ value of success.

**Practical use case.** Tool failure handling.

**Source.** Decision theory standard.

#### Failure probability

**Equation**

$$
P(\mathrm{fail}\mid \mathrm{context},\mathrm{tool})=\sigma(w^\top\phi(\mathrm{context},\mathrm{tool}))
$$

**Plain-English explanation.** Predicts probability of tool/action failure.

**Symbols.** $\phi$ context-tool features.

**Practical use case.** Agent safety and routing.

**Source.** Logistic regression standard.

## 41. Reward Modeling

#### Plackett-Luce ranking

**Equation**

$$
P(\pi)=\prod_{i=1}^{n}\frac{e^{s_{\pi_i}}}{\sum_{j=i}^{n}e^{s_{\pi_j}}}
$$

**Plain-English explanation.** Models probability of a full ranking from item scores.

**Symbols.** $\pi$ permutation.

**Practical use case.** Ranking multiple agent outputs.

**Source.** Plackett (1975); Luce (1959).

#### Pairwise margin reward loss

**Equation**

$$
\mathcal{L}=\max(0,m-r(x,y_w)+r(x,y_l))
$$

**Plain-English explanation.** Enforces winner reward above loser reward by margin.

**Symbols.** $m$ margin.

**Practical use case.** Reward model training.

**Source.** Ranking SVM / preference learning literature.

#### Process reward loss

**Equation**

$$
\mathcal{L}_{\mathrm{PRM}}=-\sum_t[y_t\log p_t+(1-y_t)\log(1-p_t)]
$$

**Plain-English explanation.** Classifies each reasoning step as good or bad.

**Symbols.** $y_t$ step label; $p_t$ predicted correctness.

**Practical use case.** Process supervision for reasoning agents.

**Source.** Lightman et al. (2023).

#### Outcome reward loss

**Equation**

$$
\mathcal{L}_{\mathrm{ORM}}=-(y\log p+(1-y)\log(1-p))
$$

**Plain-English explanation.** Scores final answer correctness.

**Symbols.** $y$ outcome label.

**Practical use case.** Outcome-supervised reward models.

**Source.** Cobbe et al. (2021); Lightman et al. (2023).

## 42. Memory and Context

#### Attention over memory

**Equation**

$$
c_t=\sum_i\alpha_{\mathrm{ti}}m_i,\quad \alpha_{\mathrm{ti}}=\mathrm{softmax}(q_t^\top k_i)
$$

**Plain-English explanation.** Retrieves memory items by attention weights.

**Symbols.** $m_i$ memory values; $k_i$ memory keys.

**Practical use case.** Memory-augmented agents.

**Source.** Graves et al. (2014); Weston et al. (2015).

#### Episodic memory score

**Equation**

$$
\mathrm{score}(m_i)=\lambda_r \mathrm{rel}(q,m_i)+\lambda_t \mathrm{recency}(m_i)+\lambda_s \mathrm{salience}(m_i)
$$

**Plain-English explanation.** Combines relevance, recency, and salience.

**Symbols.** $m_i$ memory item.

**Practical use case.** Long-term agent memory retrieval.

**Source.** Generative Agents; Park et al. (2023).

#### Exponential memory decay

**Equation**

$$
\mathrm{decay}(m_i)=e^{-\lambda(t-t_i)}
$$

**Plain-English explanation.** Older memories decay exponentially.

**Symbols.** $t_i$ creation time.

**Practical use case.** Memory ranking and forgetting.

**Source.** Cognitive modeling / agent practice.

#### Memory write decision

**Equation**

$$
\mathrm{write}=\mathbf{1}[\mathrm{novelty}(m)>\tau_n \land \mathrm{importance}(m)>\tau_i]
$$

**Plain-English explanation.** Stores memory only if novel and important enough.

**Symbols.** $\tau_n,\tau_i$ thresholds.

**Practical use case.** Preventing memory pollution.

**Source.** Agent memory engineering practice.

#### Retrieval-augmented agent policy

**Equation**

$$
\pi(a\mid s,M)=\pi(a\mid s,\mathrm{Retrieve}(M,s))
$$

**Plain-English explanation.** Agent action depends on retrieved memories.

**Symbols.** $M$ memory store.

**Practical use case.** RAG agents and tool agents.

**Source.** Memory-augmented RL/LLM-agent literature.

## 43. Risk, Safety, and Constrained Agents

#### Constrained MDP

**Equation**

$$
\max_\pi J_R(\pi)\quad\mathrm{s.t.}\quad J_C(\pi)\leq d
$$

**Plain-English explanation.** Maximizes reward under cost/safety constraint.

**Symbols.** $J_C$ expected cost; $d$ limit.

**Practical use case.** Safe agents and budgeted tool use.

**Source.** Altman (1999).

#### Lagrangian constrained objective

**Equation**

$$
\mathcal{L}(\pi,\lambda)=J_R(\pi)-\lambda(J_C(\pi)-d)
$$

**Plain-English explanation.** Turns constrained problem into penalized optimization.

**Symbols.** $\lambda\ge0$ Lagrange multiplier.

**Practical use case.** Safe RL and constrained tool policies.

**Source.** Altman (1999).

#### CVaR

**Equation**

$$
\mathrm{CVaR}_\alpha(X)=\mathbb{E}[X\mid X\leq \mathrm{VaR}_\alpha(X)]
$$

**Plain-English explanation.** Expected outcome in the worst alpha-tail.

**Symbols.** $VaR_\alpha$ value-at-risk quantile.

**Practical use case.** Risk-sensitive agent planning.

**Source.** Rockafellar and Uryasev (2000).

#### Human handoff threshold

**Equation**

$$
\mathrm{handoff}=\mathbf{1}[P(\mathrm{error}\mid x)> \tau_e\lor \mathrm{Risk}(x)>\tau_r]
$$

**Plain-English explanation.** Escalates when predicted error or risk is too high.

**Symbols.** $\tau_e,\tau_r$ thresholds.

**Practical use case.** Production agent safety.

**Source.** Decision theory / selective prediction practice.

#### Safety classifier

**Equation**

$$
P(\mathrm{unsafe}\mid x)=\sigma(w^\top h_x)
$$

**Plain-English explanation.** Predicts unsafe content or action probability.

**Symbols.** $h_x$ representation of input/action.

**Practical use case.** Guardrails and action filtering.

**Source.** Binary classification standard.

# Important Python/PyTorch Snippets

## Scaled dot-product attention
```python
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ v
```

## LoRA linear layer
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return x @ self.weight.T + self.scale * (x @ self.A.T @ self.B.T)
```

## BM25
```python
import math

def bm25_score(query_terms, doc_tf, doc_len, avgdl, df, N, k1=1.5, b=0.75):
    score = 0.0
    for t in query_terms:
        tf = doc_tf.get(t, 0)
        if tf == 0:
            continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
        denom = tf + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * (tf * (k1 + 1)) / denom
    return score
```

## Reciprocal Rank Fusion
```python
from collections import defaultdict

def rrf(rank_lists, k=60):
    scores = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## DPO loss
```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_w, policy_l, ref_w, ref_l, beta=0.1):
    # inputs are sequence log-probabilities
    logits = beta * ((policy_w - ref_w) - (policy_l - ref_l))
    return -F.logsigmoid(logits).mean()
```

## Q-learning update
```python
def q_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    best_next = max(Q[(s_next, ap)] for ap in actions)
    td_target = r + gamma * best_next
    Q[(s, a)] += alpha * (td_target - Q[(s, a)])
    return Q
```

# References

- Abdul-Jaleel et al. (2004). UMass at TREC 2004: Novelty and HARD.
- Ackley, Hinton, Sejnowski (1985). A Learning Algorithm for Boltzmann Machines.
- Ahmadian et al. (2024). Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback.
- Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
- Altman (1999). Constrained Markov Decision Processes.
- Auer et al. (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
- Azar et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Preferences.
- Ba, Kiros, Hinton (2016). Layer Normalization.
- Baevski and Auli (2019). Adaptive Input Representations for Neural Language Modeling.
- Bahdanau, Cho, Bengio (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Banerjee and Lavie (2005). METEOR.
- Bellman (1957). Dynamic Programming.
- Beltagy, Peters, Cohan (2020). Longformer.
- Ben Zaken, Goldberg, Ravfogel (2022). BitFit.
- Bengio et al. (2003). A Neural Probabilistic Language Model.
- Bishop (2006). Pattern Recognition and Machine Learning.
- Bojanowski et al. (2017). Enriching Word Vectors with Subword Information.
- Bradley and Terry (1952). Rank Analysis of Incomplete Block Designs.
- Bridle (1990). Probabilistic Interpretation of Feedforward Classification Network Outputs.
- Brier (1950). Verification of Forecasts Expressed in Terms of Probability.
- Burda et al. (2019). Exploration by Random Network Distillation.
- Burges et al. (2005). Learning to Rank using Gradient Descent.
- Burges et al. (2006). Learning to Rank with Nonsmooth Cost Functions.
- Cao et al. (2007). Learning to Rank: From Pairwise Approach to Listwise Approach.
- Carbonell and Goldstein (1998). The Use of MMR for Diversity-Based Reranking.
- Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
- Chen et al. (2021). Evaluating Large Language Models Trained on Code.
- Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation.
- Chen et al. (2023). Symbolic Discovery of Optimization Algorithms / Lion.
- Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling.
- Child et al. (2019). Generating Long Sequences with Sparse Transformers.
- Choromanski et al. (2021). Rethinking Attention with Performers.
- Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences.
- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.
- Cormack, Clarke, Buettcher (2009). Reciprocal Rank Fusion.
- Cover and Thomas (1991). Elements of Information Theory.
- Dao et al. (2022). FlashAttention.
- Dauphin et al. (2017). Language Modeling with Gated Convolutional Networks.
- Dettmers et al. (2022). LLM.int8().
- Dettmers et al. (2023). QLoRA.
- Devlin et al. (2019). BERT.
- Edge et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization.
- Elfwing et al. (2018). Sigmoid-Weighted Linear Units.
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
- Espeholt et al. (2018). IMPALA.
- Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization.
- Fan et al. (2018). Hierarchical Neural Story Generation.
- Fedus, Zoph, Shazeer (2021). Switch Transformers.
- Formal et al. (2021). SPLADE.
- Frantar et al. (2023). GPTQ.
- Fujimoto, van Hoof, Meger (2018). TD3.
- Gao et al. (2023). HyDE.
- Geifman and El-Yaniv (2017). Selective Classification for Deep Neural Networks.
- Glorot and Bengio (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
- Goodfellow, Bengio, Courville (2016). Deep Learning.
- Graves et al. (2014). Neural Turing Machines.
- Gu et al. (2021). Efficiently Modeling Long Sequences with Structured State Spaces.
- Gu and Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Guo et al. (2017). On Calibration of Modern Neural Networks.
- Guo et al. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization.
- Haarnoja et al. (2018). Soft Actor-Critic.
- Hamming (1950). Error Detecting and Error Correcting Codes.
- Haviv et al. (2022). Transformer Language Models without Positional Encodings Still Learn Positional Information.
- He et al. (2015). Delving Deep into Rectifiers.
- Hendrycks and Gimpel (2016). Gaussian Error Linear Units.
- Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network.
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration.
- Hong et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model.
- Howard (1960). Dynamic Programming and Markov Processes.
- Hu et al. (2022). LoRA.
- Huang et al. (2016). Deep Networks with Stochastic Depth.
- Ioffe and Szegedy (2015). Batch Normalization.
- Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
- Jaccard (1901). Distribution de la flore alpine.
- Järvelin and Kekäläinen (2002). Cumulated Gain-Based Evaluation of IR Techniques.
- Jeh and Widom (2003). Scaling Personalized Web Search.
- Jégou, Douze, Schmid (2011). Product Quantization for Nearest Neighbor Search.
- Jelinek et al. (1977). Perplexity and language modeling foundations.
- Joachims (2002). Optimizing Search Engines using Clickthrough Data.
- Johnson, Douze, Jégou (2017). Billion-scale Similarity Search with GPUs / FAISS.
- Kaelbling, Littman, Cassandra (1998). Planning and Acting in Partially Observable Stochastic Domains.
- Kalman (1960). A New Approach to Linear Filtering and Prediction Problems.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- Karpukhin et al. (2020). Dense Passage Retrieval.
- Kazemnejad et al. (2023). The Impact of Positional Encoding on Length Generalization.
- Khattab and Zaharia (2020). ColBERT.
- Kingma and Ba (2015). Adam.
- Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- Kitaev, Kaiser, Levskaya (2020). Reformer.
- Kocsis and Szepesvári (2006). Bandit Based Monte-Carlo Planning.
- Konda and Tsitsiklis (2000). Actor-Critic Algorithms.
- Kool et al. (2019). Buy 4 REINFORCE Samples, Get a Baseline for Free.
- Kostrikov et al. (2021). Offline Reinforcement Learning with Implicit Q-Learning.
- Kudo (2018). Subword Regularization / SentencePiece Unigram LM.
- Kulesza and Taskar (2012). Determinantal Point Processes for Machine Learning.
- Kullback and Leibler (1951). On Information and Sufficiency.
- Kumar et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning.
- Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention / vLLM.
- Lai and Robbins (1985). Asymptotically Efficient Adaptive Allocation Rules.
- Lavrenko and Croft (2001). Relevance-Based Language Models.
- Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Li and Liang (2021). Prefix-Tuning.
- Li et al. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation.
- Li et al. (2023). Contrastive Decoding.
- Lightman et al. (2023). Let's Verify Step by Step.
- Lin (2004). ROUGE.
- Lin et al. (2017). Focal Loss.
- Lin et al. (2023). AWQ.
- Liu (2009). Learning to Rank for Information Retrieval.
- Liu et al. (2022). P-Tuning v2.
- Liu et al. (2022). IA3 / Few-Shot Parameter-Efficient Fine-Tuning.
- Liu et al. (2024). Lost in the Middle.
- Liu et al. (2024). DoRA.
- Lloyd (1982). Least Squares Quantization in PCM.
- Loshchilov and Hutter (2017). SGDR.
- Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization.
- Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
- Lowerre (1976). The HARPY Speech Recognition System.
- Luce (1959). Individual Choice Behavior.
- Lv and Zhai (2011). Lower-Bounding Term Frequency Normalization.
- Malkov and Yashunin (2018). HNSW.
- Manning, Raghavan, Schütze (2008). Introduction to Information Retrieval.
- Meister et al. (2023). Typical Decoding for Natural Language Generation.
- Meng et al. (2024). SimPO.
- Mikolov et al. (2010). Recurrent Neural Network Based Language Model.
- Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space.
- Mnih et al. (2015). Human-Level Control through Deep Reinforcement Learning.
- Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning.
- Misra (2019). Mish.
- Naeini, Cooper, Hauskrecht (2015). Obtaining Well Calibrated Probabilities.
- Nair and Hinton (2010). Rectified Linear Units Improve Restricted Boltzmann Machines.
- Nash (1950). Equilibrium Points in N-Person Games.
- Newman (2006). Modularity and Community Structure in Networks.
- Ng, Harada, Russell (1999). Policy Invariance under Reward Transformations.
- Nogueira and Cho (2019). Passage Re-ranking with BERT.
- Nogueira et al. (2020). Document Ranking with a Pretrained Sequence-to-Sequence Model.
- Oord, Li, Vinyals (2018). Representation Learning with Contrastive Predictive Coding.
- Ouyang et al. (2022). Training Language Models to Follow Instructions with Human Feedback.
- Page et al. (1999). The PageRank Citation Ranking.
- Papineni et al. (2002). BLEU.
- Park et al. (2023). Generative Agents.
- Pascanu, Mikolov, Bengio (2013). On the Difficulty of Training Recurrent Neural Networks.
- Pathak et al. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction.
- Peng et al. (2019). Advantage-Weighted Regression.
- Peng et al. (2023). RWKV.
- Peng et al. (2023). YaRN.
- Pennington, Socher, Manning (2014). GloVe.
- Pillutla et al. (2021). MAUVE.
- Plackett (1975). The Analysis of Permutations.
- Poli et al. (2023). Hyena Hierarchy.
- Polyak (1964). Some Methods of Speeding Up the Convergence of Iteration Methods.
- Pomerleau (1989). ALVINN.
- Ponte and Croft (1998). A Language Modeling Approach to Information Retrieval.
- Precup (2000). Eligibility Traces for Off-Policy Policy Evaluation.
- Press and Wolf (2017). Using the Output Embedding to Improve Language Models.
- Press et al. (2021). ALiBi.
- Raffel et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
- Rafailov et al. (2023). Direct Preference Optimization.
- Rajpurkar et al. (2016). SQuAD.
- Ramachandran, Zoph, Le (2017). Searching for Activation Functions.
- Rashid et al. (2018). QMIX.
- Radford et al. (2018). Improving Language Understanding by Generative Pre-Training.
- Robertson (1977). The Probability Ranking Principle in IR.
- Robertson and Sparck Jones (1976). Relevance Weighting of Search Terms.
- Robertson et al. (1994). Okapi at TREC-3.
- Robertson et al. (2004). Simple BM25 Extension to Multiple Weighted Fields.
- Robbins and Monro (1951). A Stochastic Approximation Method.
- Rocchio (1971). Relevance Feedback in Information Retrieval.
- Rockafellar and Uryasev (2000). Optimization of Conditional Value-at-Risk.
- Rummery and Niranjan (1994). On-Line Q-Learning using Connectionist Systems.
- Salton et al. (1975). A Vector Space Model for Automatic Indexing.
- Salton and Buckley (1988). Term-Weighting Approaches in Automatic Text Retrieval.
- Sanh et al. (2019). DistilBERT.
- Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?
- Schick et al. (2023). Toolformer.
- Schroff, Kalenichenko, Philbin (2015). FaceNet.
- Schulman et al. (2015). Trust Region Policy Optimization.
- Schulman et al. (2016). Generalized Advantage Estimation.
- Schulman et al. (2017). Proximal Policy Optimization.
- Schuster and Nakajima (2012). Japanese and Korean Voice Search / WordPiece.
- Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
- Shannon (1948). A Mathematical Theory of Communication.
- Shao et al. (2024). DeepSeekMath.
- Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need.
- Shazeer (2020). GLU Variants Improve Transformer.
- Shazeer and Stern (2018). Adafactor.
- Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.
- Shinn et al. (2023). Reflexion.
- Silver et al. (2014). Deterministic Policy Gradient Algorithms.
- Silver et al. (2017). Mastering Chess and Shogi by Self-Play with a General RL Algorithm.
- Sparck Jones (1972). A Statistical Interpretation of Term Specificity.
- Srivastava et al. (2014). Dropout.
- Su et al. (2021). RoFormer / Rotary Position Embedding.
- Sun et al. (2022). xPos / Length Extrapolatable Transformer.
- Sun et al. (2023). Retentive Network.
- Sunehag et al. (2018). Value-Decomposition Networks.
- Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning.
- Sutton (1988). Learning to Predict by the Methods of Temporal Differences.
- Sutton and Barto (2018). Reinforcement Learning: An Introduction.
- Sutton et al. (1999). Policy Gradient Methods.
- Szegedy et al. (2016). Rethinking the Inception Architecture.
- Thompson (1933). On the Likelihood that One Unknown Probability Exceeds Another.
- Touvron et al. (2023). LLaMA.
- van Hasselt (2010). Double Q-learning.
- van Hasselt et al. (2016). Deep RL with Double Q-learning.
- van Seijen et al. (2009). A Theoretical and Empirical Analysis of Expected Sarsa.
- Vaswani et al. (2017). Attention Is All You Need.
- Veličković et al. (2018). Graph Attention Networks.
- von Neumann (1928). Zur Theorie der Gesellschaftsspiele.
- Voorhees (1999). The TREC-8 Question Answering Track Report.
- Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
- Wang et al. (2020). Linformer.
- Wang et al. (2022). DeepNet / DeepNorm.
- Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning.
- Watkins (1989). Learning from Delayed Rewards.
- Wei et al. (2022). Emergent Abilities of Large Language Models.
- Weston, Chopra, Bordes (2015). Memory Networks.
- Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist RL.
- Williams et al. (2009). Roofline.
- Wu and He (2018). Group Normalization.
- Wu et al. (2016). Google's Neural Machine Translation System.
- Xia et al. (2008). Listwise Approach to Learning to Rank.
- Xiao et al. (2023). SmoothQuant.
- Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks.
- Xiong et al. (2020). On Layer Normalization in the Transformer Architecture.
- Yao et al. (2023). ReAct.
- Yao et al. (2023). Tree of Thoughts.
- Yu et al. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models.
- Zhai and Lafferty (2001). A Study of Smoothing Methods for Language Models Applied to Ad Hoc IR.
- Zhang and Sennrich (2019). RMSNorm.
- Zhang et al. (2020). BERTScore.
- Zhang et al. (2023). AdaLoRA.
- Ziegler et al. (2019). Fine-Tuning Language Models from Human Preferences.
- Zoph et al. (2022). Designing Effective Sparse Expert Models.
