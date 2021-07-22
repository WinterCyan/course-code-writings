# *BLEU* calculation

**standard unigram precision**,

Candidate: $Candi$

References: $Ref1, Ref2, \cdots$
$$
Precision = \frac{Length~~of~~list[w_i~~if~~(w_i \in Candi\cap Refs)~~for~~i\in[1,Length_{Candi}]]}{Length_{Candi}}
$$
*example 1*,

- Candidate: the the the the the the the.

- Reference 1: The cat is on the mat.

- Reference 2: There is a cat on the mat.

  every <u>*the*</u> in $Candi$ do appear in $Ref1$ or $Ref2$, so $Length_{list}=7$.

  the length of $Candi$ is $7$, so the final $Precision$ is $\frac{7}{7}=1$.

**modified unigram precision**,

![Screenshot 2021-07-15 141442](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/blog-BLEU-calculation.png)

| word          | it   | is   | a    | guide | to   | action | which | ensures | that | the  | miltary | always | obeys | commands | of   | party | SUM  |
| ------------- | ---- | ---- | ---- | ----- | ---- | ------ | ----- | ------- | ---- | ---- | ------- | ------ | ----- | -------- | ---- | ----- | ---- |
| count         | 1    | 1    | 1    | 1     | 1    | 1      | 1     | 1       | 1    | 3    | 1       | 1      | 1     | 1        | 1    | 1     | 18   |
| Max_Ref_Count | 1    | 1    | 1    | 1     | 1    | 1      | 1     | 1       | 2    | 4    | 1       | 1      | 0     | 1        | 1    | 1     |      |
| Count_clip    | 1    | 1    | 1    | 1     | 1    | 1      | 1     | 1       | 1    | 3    | 1       | 1      | 0     | 1        | 1    | 1     | 17   |

1. get $CountInCandi_{w_i}$ for every word $w_i \in Candi$,

2. get $MaxRefCount_{w_i}$ for every word $w_i \in Candi$,
   $$
   MaxRefCount_{w_i} = max\{CountInRef1_{w_i},CountInRef2_{w_i},CountInRef3_{w_i},\cdots\}
   $$

3. get $CountClip_{w_i}$ for every word $w_i \in Candi$,
   $$
   CountClip_{w_i}=min\{ CountInCandi_{w_i},MaxRefCount_{w_i} \}
   $$

4. final, presion,
   $$
   Precision = \frac{\sum_{w_i} CountClip_{w_i}}{Length_{Candi}}
   $$

**modified n-gram precision**,

e.g., bi-gram.

![Screenshot 2021-07-15 141442](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/blog-BLEU-calculation-2.png)

in formulation, for a whole test corpus,
$$
p_n = \frac{\sum \limits_{C\in\{Candidates\}}\sum \limits_{n-gram\in C}CountClip_{n-gram}}{\sum \limits_{C^{'} \in\{Candidates\}} \sum \limits_{n-gram^{'}\in C^{'}}Count_{n-gram^{'}}}
$$
and, for $n=1,2,3,4,~~ p_1>p_2>p_3>p_4$, use *geometric mean* on $p_n$.
$$
p_{mean} = \sqrt[4]{p_1\cdot p_2 \cdot p_3 \cdot p_4}
$$
**Question 1**, for now, *modified n-gram precision* has penalty for *too long* candidates. for *too short* candidates, it has no penalty.

- Candidate: of the

- Reference 1: It is a guide to action that ensures that the military will forever heed Party commands.

- Reference 2: It is the guiding principle which guarantees the military forces always being under the command of the Party.

- Reference 3: It is the practical guide for the army always to heed the directions of the party.

  $p_1=\frac{2}{2}, ~~ p_2 = \frac{1}{1}$

**Question 2**, traditional *recall* is **not** a good measure to enforce *proper length*,

- Candidate 1: I always invariably perpetually do.
- Candidate 2: I always do. $\checkmark$
- Reference 1: I always do.
- Reference 2: I invariably do.
- Reference 3: I perpetually do.

**brevity penalty**,

$r$, sum of *best match* lengths for candidate in corpus.

$c$, sum of candidates in corpus.

then, the *brevity penalty*,
$$
BP=\begin{cases}
1 & if~~c>r \\
e^{1-\frac{r}{c}} & if~~c \leq r
\end{cases}
$$
then, 
$$
\boldsymbol{B}LEU = BP \cdot e^{\sum\limits_{n=1}^N w_n log p_n} \\
log \boldsymbol{B}LEU=\mathrm{min}(1-\frac{r}{c},0)+ \sum\limits_{n=1}^N w_n \mathrm{log} p_n,~~w_n=\frac{1}{N}
$$
**remarks**,

1. 倾向于整体特性，对单句翻译质量的评价可能不准，不适于比较单独的语句。
2. 取值范围 $[0,1]$。
3. 句子的参考翻译越多，得分越高。

