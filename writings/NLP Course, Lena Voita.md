# NLP Course, Lena Voita

## Course 1, Word Embeddings

**One-hot Vectors:**

1. 数据量大，每个单词占用 N 位，N为单词表的词数。
2. **不表意**，cat 距离 dog 可能和 table 相同。

**Distributional Semantics:**

达意：在不同语境中表现的意思*相似*。

![image-20210702153258805](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-context-meaning.png)

上图中，意义相近的词，行（向量）相似。词嵌入过程中，把语境信息嵌入单词表示/向量中。

> Words which frequently appear in similar contexts have similar meaning.

***Count*-based Methods:** 根据全局预料数据，将上下文信息*手动*添加到词向量中。

1. Co-Occurence Counts 共现次数

   ![image-20210702152809394](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-count-based-1.png)

1. PPMI (Positive Pointwise Mutual Information) 正点互信息

   ![image-20210702152853734](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-count-based-2.png)

2. LSA (Latent Semantic Analysis): Understanding *Documents* 潜在语义分析，用于文档理解、

   ![image-20210702152917390](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-count-based-3.png)

***Prediction(window)*-Based Method:** **Word2Vec**

![](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-word2vec-1.jpeg)

![](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-word2vec-2.jpeg)

**Skip-Gram** and **CBOW**

- [x] Negative Sampling. 为何使用sigmoid？为何计算 1-sigmoid(P)？负采样的理解？

  **负采样：**对每一个（central - context）词对，选取若干个*负样本*，即不在 central word 上下文出现的单词。更新 embedding 权重时，只更新 central word、context word 以及 negative sample 的词向量。

  **1-sigmoid(P)：**

  ​	log(σ(P(context|central))) → 词对*出现*的概率，正样本

  ​	log(1-σ(P(negative_context|central))) → 词对*不出现*的概率，负样本

- [x]  U 与 V 对应模型中的哪些部分？分别对于SG和CBOW模型。
  1. 在 CBOW 实现中，没有显式表示 U 与 V，只是用 embedding 保存并更新词向量。
  2. 在 Skip-Gram 实现中，U 与 V 分别为两个 embedding。

Implementation.

- [x] embedding 的意义？

  不使用 W 和 W' 存储 input vector 和 output vector，使用两个全连接层组成网络，使用 embedding 存储词向量，梯度更新全连接层的权重及 embedding 中特定词向量。

- [ ] CBOW 实现中，为何样本标签为单词 index？

```python
total_loss += loss_fun(log_probs, torch.tensor([word_to_idx[target]]))
```

- [x] NLL loss (Negative Log-Likelihood)?

  *softmax + nl* = *log_softmax+nll* = *cross_entropy*(pass x directly in, without exp/softmax)

python code:

```python
def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    currentCenterWordIdx = word2Ind[currentCenterWord]
    v_c = centerWordVectors[currentCenterWordIdx, :]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        partial_loss, partial_gradCenterVec, partial_gradOutsideVecs = \
            word2vecLossAndGradient(centerWordVec=v_c,
                                    outsideWordIdx=outsideWordIdx,
                                    outsideVectors=outsideVectors,
                                    dataset=dataset)
        loss += partial_loss
        gradCenterVecs[currentCenterWordIdx, :] += partial_gradCenterVec
        gradOutsideVectors += partial_gradOutsideVecs
    
    return loss, gradCenterVecs, gradOutsideVectors
```



**window_size**

- large window: 主题相关的相似性，（狗，吠，项圈；走，跑）  ---> *semantic*
- small window: 功能相关，形式相近，（walking, running, approaching） ---> *syntactic*

**GloVe**

- [ ] To be continued...

**Evaluation of Word Embedding**

1. Intrinsic：最近邻，近义词评估；线性结构，词类推评估；跨语言相似性。

   快速，但不直观。

2. Extrinsic，通过具体的任务评估，如文本分类、指代消解等。

   需要在多个模型上训练并得出结果，较慢。

## Course 2, Text Classification

> Probability of a **sentence**.

- 二分类，target $\in \mathbb{R}^2$
- 多分类，target $\in \mathbb{R}^N, N\geq 3$，单个正确
- 多标签，target $\in \mathbb{R}^N, N\geq 3$，多个正确

**datasets:** 

*sentiment*: SST, IMDb Review, Yelp Review, Amazon Review, 

*question*: TREC, Yahoo! Answers, 

*topic*: AG's News, Sogou News, DBPedia, etc.

### General View

structure: **feature extractor** + **classifier**，类似于 CV 问题，提取特征，分类。

特征提取：传统方法、神经网络；分类器：逻辑回归、朴素贝叶斯、SVM等

![image-20210707151432490](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-structure.png)

生成模型 v.s. 判别模型

生成模型：学习联合概率分布，$p(x,y)=p(x|y)\cdot p(y)$. 输入 $x$，则，
$$
y=\mathop{\arg\max}_{k}p(x|y=k)\cdot p(y=k)
$$
即给定分类，计算输入 $x$ 时 $x,y$ 出现的概率（联合分布），选择最大者对应的分类。

判别模型：学习分类边界，输入$x$，则，
$$
y=\mathop{\arg\max}_{k}p(y=k|x)
$$
即给定输入 $x$，计算分类概率，并选择最大者。

### 传统方法

#### 朴素贝叶斯（生成模型）

根据输入不断调整 $P(y=k|x)$，并根据贝叶斯定理计算 $P(x|y=k) \cdot P(y=k)$。
$$
P(y=k|x) \cdot P(x) = P(xy) = P(x|y=k) \cdot P(y=k) \\
P(y=k|x) = \frac{P(x|y=k) \cdot P(y=k)}{P(x)} \\
略去 P(x), \rightarrow y^*=\mathop{\arg\max}_{k}p(x|y=k)\cdot p(y=k)
$$
$x$ 为文档，$y$ 为类别。

$P(y=k)$: 先验概率，接受数据之前

$P(y=k|x)$: 后验概率，接受数据之后

估计准则：最大*后验*

先验概率 $P(y=k)$ 的确定：基于数据集中*标签数量*。

各个*词*的后验概率 $P(x|y=k)$ 的确定，根据**朴素（Naive）**贝叶斯假设，

1. 词袋假设，词序无关
2. 条件独立假设，给定分类，各特征独立

$$
P(x|y=k)=P(x_1,x_2, \ldots,x_n|y=k)=\prod ^n _{t=1} P(x_t | y=k) \\
P(x_i | y=k) = \frac{N(x_i,y=k)}{\sum^{|V|}_{t=1}N(x_t,y=k)}=\frac{x_i在k类别中出现的次数}{k类别中的总词数}
$$

$P(x_i | y=k)$ 的情况：

判定 $P(x|y=k)$ 的概率，即在 $k$ 类中，出现文档 $x$ 的概率由 $x$ 中所有词/特征概率相乘得到，一旦 $P(x_i | y=k)=0$，则使 $P(x|y=k)=0$，不合理！通过添加微小量解决，
$$
P(x_i | y=k) = \frac{\color{red} \delta \color{black} + N(x_i,y=k)}{\sum^{|V|}_{t=1}\color{red} \delta \color{black} + N(x_t,y=k)}
$$
$\delta$ 的选取可通过*交叉验证*得到。

> 手动定义如何使用特征/手动定义特征，没有*学习*过程

**模型预测：**

![image-20210707162333774](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-bayes-prediction.png)

对不同分类计算联合分布概率，即先验概率乘以各个词的后验概率，选择联合概率分布大的分类。

在应用中，使用概率 log 的和，而非概率积。概率的表示使用CBOW，只用次数，无需 embedding。

- [ ] **特点/优缺点**

#### 逻辑回归/最大熵分类器（判别模型）

**Pipeline**

1. $h=(\color{red}f_0=1,\color{black}f_1,f_2,\ldots,f_n) \in \mathbb{R}^{n+1}$ 将文本转换为特征向量；

2. *每个类别*有权重向量 $w^{class=i} =(\color{red}w^{(i)}_0=b^{(i)},\color{black}w^{(i)}_1,w^{(i)}_2, \ldots, w^{(i)}_n) \in \mathbb{R}^{n+1}$；

3. 特征向量和权重*内积* $w^{(k)}h$；

4. *softmax*，
   $$
   P(class=k|h)=\frac{e^{w^{(k)}h}}{\sum^K_{i=1}e^{w^{(i)}h}}
   $$
   **implementation:** 所有 $e^{w^{(i)}h}$ 先减去其中最大值，然后 softmax。

![image-20210707212121488](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-logistic-regression.png)

**训练**

训练权重使 $w^*$ 有，
$$
w^*=\mathop{\arg \max}_w \sum^N_{i=1}logP(y=y^i|x^i)
$$
即，*最大（指数）似然估计*，等价于*最小化交叉熵*，（目标分布 $\boldsymbol{p^*}=(0,\ldots,0,1,0,\ldots)$ 和估计分布 $\boldsymbol{p}=(p_1,\ldots,p_K),p_i=p(i|x)$ 之间），
$$
Loss(p^*,p)=-p^*log(p)=-\sum^K_{i=1}p^*_ilog(p_i)=-log(p_k)=-log(p(k|x)),~where~ p^*_k=1
$$
![image-20210707213908489](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-logistic-regression-loss.png)

> $\color{red} \bold{？？？}$ 计算损失时，只计算第 k 类 loss，因为我们并不知道其他类的标签是否*确切*的错误。
>
> At the same time, we minimize sum of the probabilities of incorrect classes: since sum of all probabilities is constant, by increasing one probability we decrease sum of all the rest.
>
> - [ ] 损失计算的对象？其他权重是否更新？

#### 贝叶斯和逻辑回归的对比

| **贝叶斯**              | **逻辑回归**                            |
| ----------------------- | --------------------------------------- |
| $\checkmark$ *非常*简单 | *$\checkmark$ 挺*简单                   |
| $\checkmark$ *非常*快   | $\checkmark$ 可解释                     |
| $\checkmark$ 可解释     | $\checkmark$ 没有*特征条件独立*这一假设 |
| 假设*特征条件独立*      | *没那么*快                              |
| 手动定义特征            | 手动定义特征                            |

### 神经网络

> 神经网络*学习*特征（网络输入 embedding，输出句子的特征），而不是手动设定。

![image-20210707220935349](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-nn.png)

NN 之后的分类器，即为一个逻辑回归，以一个*全连接*实现。

![image-20210707221448816](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-nn-final-fc-layer.png)

右上图，$\color{orange}w_1,\color{red}w_2,\color{blue}w_3$ 为全连接层中相应的权重，分别指向不同类空间中心。

#### BoE（Bag of Embedding)

直接使用（相加求*和*或求*权重和*） embedding 向量，无神经网络。做 baseline，适用于少量数据时。

#### RNN/LSTM/etc

> Recurrent networks are a natural way to process text in a sense that, similar to humans, they **read** a sequence of tokens *one by one* and process the information. Hopefully, at each step the network will **remember** everything it has read before.

**RNN 单元**：之前的状态向量 & 当前的词向量 $\rightarrow$ 新的状态向量，

![](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-rnn-cell.png)

**Vanilla RNN**：

![image-20210708111139537](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-vanilla-rnn.png)
$$
h_t=\tanh (h_{t-1}W_h+x_tW_t)
$$
**文本分类中的 RNN**：

> We need a model that can produce a **fixed-sized** vector for inputs of **different** lengths.

1. **one-layer**，使用最后一个状态向量。

2. **multiple-layers**，堆叠多层 RNN，$i$ 层使用 $i-1$ 层的状态向量作为输入；

   低层：短语等；

   高层：主题等。

3. **bi-directional**，使用两个RNN，分别以从头至尾和从尾至头的方向输入词向量。

![image-20210708112343023](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-simple-rnn.png)

![image-20210708112429391](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-multi-layer-rnn.png)

![image-20210708112524845](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-text-classification-bidirectional-rnn.png)

**Implementation**:

- [x] TO BE DONE. DONE, in cs224n homework 4.

#### CNN

## Course 3, Language Modeling

**definition:**

> **Language Models** (LMs) estimate the *probability* of different linguistic units: symbols, tokens, token sequences.

### General Framework

#### Text Probability

**句子的概率：**

$sentence=[y_1,y_2,\ldots,y_n]$，则句子的概率为，
$$
\begin{align}
P(y_1,y_2,\ldots,y_n) &= P(y_1) \cdot P(y_2|y_1）\cdot P(y_3|y_1,y_2) ~~ \cdots ~~ P(y_n|y_1,y_2,\ldots,y_{n-1}) \\ &= \prod^n_{t=1} P(y_t|y_{<t})
\end{align}
$$
即得到*从左到右*语言模型，包括 N-gram 和 Neural 模型。

#### Generate a Text Using a LM

当前（部分）句子 $\rightarrow$ 词的概率 $\rightarrow$ 采样（不同的采样策略） $\rightarrow$ 循环。

![image-20210713193948320](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-generate-a-text.png)

### N-Gram

由$P(y_1,y_2,\ldots,y_n) = \prod^n_{t=1} P(y_t|y_{<t})$，*基于数量*定义 $P(y_n|y_1,y_2,\ldots,y_{n-1})$。
$$
P(y_t|y_1,y_2,\ldots,y_{t-1})=\frac{N(y_1,\cdots,y_{t-1},y_t)}{N(y_1,\cdots,y_{t-1})}
$$
采用马尔可夫性质（独立性假设），

> The probability of a word only depends on a **fixed** number of previous words.

则，
$$
\begin{align}
P(y_t|y_1,y_2,\ldots,y_{t-1}) &= P(y_t|y_{t-n+1},\cdots,y_{t-1}) \\
&= P(y_t|y_{t-2},y_{t-1}),~~ n=3 \rightarrow trigram ~~ model\\
&= P(y_t|y_{t-1}),~~ n=2 \rightarrow bigram ~~ model\\
&= P(y_t),~~ n=1\rightarrow unigram ~~ model
\end{align}
$$
![image-20210713195257433](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-n-gram-model.png)

#### Smoothing

$$
P(\textsf{ mat }|\textsf{ I saw a cat on a }) = P(\textsf{ mat }|\textsf{ cat on a })=\frac{N(\textsf{ cat on a mat })}{N(\textsf{ cat on a })}=\frac{\color{red}0?}{\color{blue}0?}
$$

1. 分母为 $\color{blue}0 \color{black} \rightarrow$ 

   1. *回退*
      $$
      \textsf{cat on a} \rightarrow \textsf{on a} \rightarrow \textsf{a} \rightarrow unigram(\approx P(\textsf{mat}))
      $$

   2. *线性插值*
      $$
      \hat{P}(\textsf{mat|cat on a}) \approx \lambda_3 P(\textsf{mat|cat on a}) + \lambda_2 P(\textsf{mat|on a}) \\
      + \lambda_1 P(\textsf{mat|a}) + \lambda_0 P(\textsf{mat}),~~ \sum_i \lambda_i =1
      $$

2. 分子为 $\color{red}0 \color{black} \rightarrow$ *拉普拉斯平滑（微小量）*。
   $$
   \hat P(\textsf{ mat }|\textsf{ cat on a })=\frac{\delta +N(\textsf{ cat on a mat })}{\delta \cdot |V| +N(\textsf{ cat on a })}
   $$

- [ ] 其他， *Kneser-Ney* Smoothing

#### Generation

*shortcoming:* 仅使用临近的若干词作为生成新词的上下文，无一致性、连贯性。

### Neural LM

#### Pipeline

不同于 N-gram *基于数量*定义 $P(y_n|y_1,y_2,\ldots,y_{n-1})$，使用神经网络*预测*概率。

相当与分类器，输入为 $prefix$，输出为概率分布 $p \in \mathbb{R}^{|V|}$，即目标词。

![image-20210713213410560](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-neural-LMs.png)

$h \in \mathbb{R}^d \rightarrow p \in \mathbb{R}^{|V|}$：经过变换矩阵 $M \in \mathbb{R}^{d \times |V|}$，则 $M$ 可以视为*输出*词嵌入矩阵，

![image-20210713213937743](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-neural-LMs-2.png)

### Generation Strategies

Goal：

1. *Coherence*：语义连贯
2. *Diversity*：多样性，可以生成多样化的结果

#### 标准采样

直接用输出概率分布采样。

#### Temperature Sampling

*temperature:* $softmax$ 函数中的参数，
$$
\frac{e^{h^\top w}}{\sum_{w_i\in V} e^{h^\top w_i}} \rightarrow \frac{e^{\frac{h^\top w}{\tau}}}{\sum_{w_i\in V} e^{\frac{h^\top w_i}{\tau}}}
$$
温度采样，即使用不同温度系数的 $softmax$ 后再进行标准采样。

$\tau$ 越大，概率分布越均匀，方差越小；$\tau$ 越小，概率分布越参差，方差越大。即改变指数的取值范围，影响差异性。

![image-20210713220205593](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-temperature-1.png)

![image-20210713220304202](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-temperature-2.png)

![image-20210713220505254](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-temperature-effect.png)

#### Top-K & Top-p 采样

Top-k，顾名思义，一般比变温采样更有效。

*K 的选择：* 固定 K 的值效果不好，

![image-20210713220941671](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-top-k.png)

Top-p，选择前 $p\%$ 的分布，

![image-20210713221112198](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/nlp-course-language-modeling-top-p.png)

### Evaluation (Intrinsic)

#### Cross-Entropy & Perplexity(confusion，困惑度)

*对数似然：*将每一步的概率 $p(y_t|y_{<t})$ 的（以 2 为底）对数求和，
$$
L(y_{1:M})=L(y_1,y_2,\cdots,y_M)=\sum_{t=1}^M log_2 p(y_t|y_{<t})
$$
*交叉熵：* **负**（自然）对数似然。

*困惑度：* 对数似然的*指数形式*，
$$
Perplexity(y_{1:M})=2^{-\frac{1}{M}L(y_{1:M})}
$$

- 最佳困惑度：1

- 最差困惑度：|V|，即每次预测每个词概率都为 $\frac{1}{|V|}$，
  $$
  Perplexity(y_{1:M})=2^{-\frac{1}{M}\cdot M\cdot log_2 \frac{1}{|V|}}=|V|
  $$

### 权重捆绑（Weight Tying）

全连接权重 $M\in \mathbb{R}^{d \times |V|}$ 使用 $Embedding$ 的权重，减小模型大小。

