from nltk import WordPunctTokenizer as Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
from sklearn.model_selection import train_test_split


data = pd.read_json("./arxivData.json")
print('loaded data.')
# print(data.sample(n=5))
lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'], axis=1).tolist()

tokenizer = Tokenizer()
lines = [' '.join(tokenizer.tokenize(line.lower())) for line in lines]
# assert sorted(lines, key=len)[0] == 'differential contrastive divergence ; this paper has been retracted .'
# assert sorted(lines, key=len)[2] == 'p = np ; we claim to resolve the p =? np problem via a formal argument for p = np .'

UNK, EOS = '_UNK_', '_EOS_'


def count_n_grams(lines, n):
    counts = defaultdict(Counter)
    for line in lines:
        # e.g. line: sequential short - text classification with recurrent and convolutional neural networks
        tokens = line.split()
        tokens = [UNK] * (n - 1) + tokens + [EOS]
        # tokens.append(EOS)
        # for i in range(n-1):
        #     tokens.insert(0, UNK)  # insert N-1 UNKs at index-0
        # tokens: ['_UNK_', '_UNK_', 'sequential', 'short', '-', 'text', ..., 'networks', '_EOS_']
        for i in range(n-1, len(tokens)):
            prefix = tuple(tokens[i - n + 1:i])  # for every word in sequence, get prefix (empty/short/long)
            c = Counter()
            c[tokens[i]] += 1
            counts[prefix] += c

    return counts


dummy_lines = sorted(lines, key=len)[:100]
# dummy_counts = count_n_grams(dummy_lines, 3)
# assert set(map(len, dummy_counts.keys())) == {2}, "please only count {n-1}-grams"
# assert len(dummy_counts[('_UNK_', '_UNK_')]) == 78
# assert dummy_counts['_UNK_', 'a']['note'] == 3
# assert dummy_counts['p', '=']['np'] == 2
# assert dummy_counts['author', '.']['_EOS_'] == 1


class NGramLM:
    def __init__(self, lines, n):
        print('initializing {}-gram LM, corpus lines: {} ...'.format(n, len(lines)))
        assert n >= 1
        self.n = n
        counts = count_n_grams(lines, self.n)
        self.probs = defaultdict(Counter)
        for prefix in counts:
            num_counter = counts[prefix]
            prefix_total_num = sum(num_counter.values())
            prob_counter = Counter()
            for word in num_counter.keys():
                prob_counter[word] = num_counter[word]/prefix_total_num
            self.probs[prefix] += prob_counter  # return a counter here

    def get_possible_next_tokens(self, prefix):
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]  # clamp too long prefix
        prefix = [UNK] * (self.n - 1 - len(prefix)) + prefix  # padding too short prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix, next_token):
        return self.get_possible_next_tokens(prefix).get(next_token, 0)


# dummy_lm = NGramLM(dummy_lines, n=3)
# p_initial = dummy_lm.get_possible_next_tokens('')
# assert np.allclose(p_initial['learning'], 0.02)
# assert np.allclose(p_initial['a'], 0.13)
# assert np.allclose(p_initial.get('meow', 0), 0)
# assert np.allclose(sum(p_initial.values()), 1)
#
# p_a = dummy_lm.get_possible_next_tokens('a')
# assert np.allclose(p_a['machine'], 0.15384615)
# assert np.allclose(p_a['note'], 0.23076923)
# assert np.allclose(p_a.get('the', 0), 0)
# assert np.allclose(sum(p_a.values()), 1)
#
# assert np.allclose(dummy_lm.get_possible_next_tokens('a note')['on'], 1)
# assert dummy_lm.get_possible_next_tokens('a machine') == dummy_lm.get_possible_next_tokens("there have always been ghosts in a machine"), "your 3-gram model should only depend on 2 previous words"

# lm = NGramLM(lines, n=3)
# model_file = open("model.pickle", "wb")
# pickle.dump(lm, model_file)
# model_file.close()
# model_file = open("model.pickle", "rb")
# lm = pickle.load(model_file)
# print('loaded model.')


def get_next_token(lm: NGramLM, prefix, temperature=1.0):
    assert temperature >= 0.0
    next_tokens_and_prob = lm.get_possible_next_tokens(prefix=prefix)
    token_list = list(next_tokens_and_prob.keys())
    token_idx_list = np.arange(len(next_tokens_and_prob.keys()))
    if temperature > 0.0:
        token_prob_list = [v ** (1.0/temperature) for v in next_tokens_and_prob.values()]
        prob_sum = sum(token_prob_list)
        token_prob_list = [v/prob_sum for v in token_prob_list]
        token_idx = np.random.choice(token_idx_list, p=token_prob_list)
    else:
        token_prob_list = [v for v in next_tokens_and_prob.values()]
        token_idx = token_prob_list.index(max(token_prob_list))
    return token_list[token_idx]


# print('first test...')
# test_freqs = Counter([get_next_token(lm, 'there have') for _ in range(10000)])
# assert 250 < test_freqs['not'] < 450
# assert 8500 < test_freqs['been'] < 9500
# assert 1 < test_freqs['lately'] < 200
#
# print('second test...')
# test_freqs = Counter([get_next_token(lm, 'deep', temperature=1.0) for _ in range(10000)])
# assert 1500 < test_freqs['learning'] < 3000
#
# print('third test...')
# test_freqs = Counter([get_next_token(lm, 'deep', temperature=0.5) for _ in range(10000)])
# assert 8000 < test_freqs['learning'] < 9000
#
# print('fourth test...')
# test_freqs = Counter([get_next_token(lm, 'deep', temperature=0.0) for _ in range(10000)])
# assert test_freqs['learning'] == 10000
#
# print("Looks nice!")
#
# prefix = 'artificial'
# for i in range(100):
#     prefix += ' ' + get_next_token(lm, prefix)
#     if prefix.endswith(EOS) or len(lm.get_possible_next_tokens(prefix)) == 0:
#         break
# print(prefix)
#
# prefix = 'bridging the'
# for i in range(100):
#     prefix += ' ' + get_next_token(lm, prefix)
#     if prefix.endswith(EOS) or len(lm.get_possible_next_tokens(prefix)) == 0:
#         break
# print(prefix)


def perplexity(lm: NGramLM, lines, min_logprob=np.log(10 ** -50.0)):
    prob = 0.0
    M = 0
    for line in lines:
        # calculate the perplexity
        tokens = line.split()
        tokens = [UNK] * (lm.n - 1) + tokens + [EOS]
        for i in range(lm.n - 1, len(tokens)):
            token = tokens[i]
            prefix = ' '.join(tokens[i-lm.n+1: i])
            prob += min_logprob if lm.get_next_token_prob(prefix, token) == 0 else np.log(lm.get_next_token_prob(prefix, token))
            M += 1
    return np.exp((-1.0/M) * prob)


# lm1 = NGramLM(dummy_lines, n=1)
# lm3 = NGramLM(dummy_lines, n=3)
# lm10 = NGramLM(dummy_lines, n=10)
#
# ppx1 = perplexity(lm1, dummy_lines)
# ppx3 = perplexity(lm3, dummy_lines)
# ppx10 = perplexity(lm10, dummy_lines)
# ppx_missing = perplexity(lm3, ['the jabberwock , with eyes of flame , '])  # thanks, L. Carrol
#
# print("Perplexities: ppx1=%.3f ppx3=%.3f ppx10=%.3f" % (ppx1, ppx3, ppx10))
#
# assert all(0 < ppx < 500 for ppx in (ppx1, ppx3, ppx10)), "perplexity should be nonnegative and reasonably small"
# assert ppx1 > ppx3 > ppx10, "higher N models should overfit and "
# assert np.isfinite(ppx_missing) and ppx_missing > 10 ** 6, "missing words should have large but finite perplexity. " \
#     " Make sure you use min_logprob right"
# assert np.allclose([ppx1, ppx3, ppx10], (318.2132342216302, 1.5199996213739575, 1.1838145037901249))


train_lines, test_lines = train_test_split(lines, test_size=0.75, random_state=42)
# for n in (1, 2, 3):
#     lm = NGramLM(n=n, lines=train_lines)
#     ppx = perplexity(lm, test_lines)
#     print("N = %i, Perplexity = %.5f" % (n, ppx))
lm = NGramLM(n=1, lines=train_lines)
print('calculating ppx...')
ppx = perplexity(lm, test_lines)
print("N = %i, Perplexity = %.5f" % (1, ppx))


class SmoothNGramLM(NGramLM):
    def __init__(self, lines, n, delta=1.0):
        print('initializing {}-gram LM, corpus lines: {} ...'.format(n, len(lines)))
        assert n >= 1
        self.n = n
        self.delta = delta
        counts = count_n_grams(lines, self.n)
        self.vocab = set(token for token_counts in counts.values() for token in token_counts)  # all words do appear in values()
        self.probs = defaultdict(Counter)
        for prefix in counts:
            num_counter = counts[prefix]
            prefix_total_num = sum(num_counter.values())
            prob_counter = Counter()
            for word in num_counter.keys():
                prob_counter[word] = (self.delta + num_counter[word])/(prefix_total_num + self.delta*(len(num_counter.values())))
            self.probs[prefix] += prob_counter

    def get_possible_next_tokens(self, prefix):
        token_probs = super().get_possible_next_tokens(prefix)
        missing_prob_total = 1.0 - sum(token_probs.values())
        missing_prob = missing_prob_total / max(1, len(self.vocab) - len(token_probs))
        return {token: token_probs.get(token, missing_prob) for token in self.vocab}  # return a dict here

    def get_next_token_prob(self, prefix, next_token):
        token_probs = super().get_possible_next_tokens(prefix)
        if next_token in token_probs:
            return token_probs[next_token]
        else:
            miss_prob_total = 1.0 - sum(token_probs.values())
            miss_prob_total = max(0, miss_prob_total)
            return miss_prob_total / max(1, len(self.vocab) - len(token_probs))


for n in (1, 2, 3):
    dummy_lm = SmoothNGramLM(dummy_lines, n=n)
    assert np.allclose(sum([dummy_lm.get_next_token_prob('a', w_i) for w_i in dummy_lm.vocab]), 1), "I told you not to break anything! :)"

# for n in (1, 2, 3):
#     lm = LaplaceLanguageModel(train_lines, n=n, delta=0.1)
#     ppx = perplexity(lm, test_lines)
#     print("N = %i, Perplexity = %.5f" % (n, ppx))
