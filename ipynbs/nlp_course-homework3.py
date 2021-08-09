import collections

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from sklearn.model_selection import train_test_split
from random import sample
from tqdm import trange

DEVICE = 'cuda:0'
# DEVICE = 'cpu'

BOS, EOS = ' ', '\n'
data = pd.read_json("arxivData.json")
lines = data.apply(lambda row: (row['title']+' ; ' + row['summary'])[:512], axis=1).apply(lambda line: BOS + line.replace(EOS, ' ') + EOS).tolist()
# print(lines[0])
tokens = sorted(set(''.join(lines)))
n_tokens = len(tokens)
# assert 100 < n_tokens < 150
# assert BOS in tokens, EOS in tokens

token_to_id = {token: id for id, token in enumerate(tokens)}
# assert len(tokens) == len(token_to_id), "dictionaries must have same size"
# for i in range(n_tokens):
#     assert token_to_id[tokens[i]] == i, "token identifier must be it's position in tokens list"


def to_matrix(lines, max_len=None, pad=token_to_id[EOS], dtype=np.int64):
    max_len = max_len or max(map(len, lines))
    matrix = np.full([len(lines), max_len], pad, dtype=dtype)
    for i in range(len(lines)):
        row = list(map(token_to_id.get, lines[i][:max_len]))
        matrix[i, :len(row)] = row

    return matrix


dummy_lines = [
    ' abc\n',
    ' abacaba\n',
    ' abc1234567890\n',
]
# matrix = torch.tensor(to_matrix(dummy_lines), dtype=torch.int64)
# embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=16)
# print(matrix)
# print(embedding(matrix))
# padding = nn.ZeroPad2d((5, 0, 0, 0))
# pprint(padding(matrix), width=200)
# pprint(padding(embedding(matrix)))


class FixedWindowLM(nn.Module):
    def __init__(self, n_tokens=n_tokens, emb_size=16, hid_size=64, kernel_size=6):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size)
        self.padding = nn.ZeroPad2d((5, 0, 0, 0))
        self.conv = nn.Conv1d(in_channels=emb_size, out_channels=hid_size, kernel_size=kernel_size, padding=0, bias=False)
        self.fc = nn.Linear(in_features=hid_size, out_features=n_tokens)

    def __call__(self, input_matrix):
        padded = self.padding(input_matrix)
        embedded = self.embedding(padded).permute(0, 2, 1)
        conved = self.conv(embedded).permute(0, 2, 1)
        output = self.fc(conved)

        return output  # shape: (batch_size, max_len, n_token)

    def get_possible_next_tokens(self, prefix=BOS, temperature=1.0, max_len=100):
        prefix_idx = torch.as_tensor(to_matrix([prefix]), dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self(prefix_idx)[0, -1], dim=-1).cpu().numpy()
        return dict(zip(tokens, probs))


# dummy_model = FixedWindowLM().to(DEVICE)
#
# dummy_input_ix = torch.as_tensor(to_matrix(dummy_lines), device=DEVICE)
# dummy_logits = dummy_model(dummy_input_ix)
#
# print('Weights:', tuple(name for name, w in dummy_model.named_parameters()))
# assert isinstance(dummy_logits, torch.Tensor)
# assert dummy_logits.shape == (len(dummy_lines), max(map(len, dummy_lines)), n_tokens), "please check output shape"
# assert np.all(np.isfinite(dummy_logits.data.cpu().numpy())), "inf/nan encountered"
# assert not np.allclose(dummy_logits.data.cpu().numpy().sum(-1), 1), "please predict linear outputs, don't use softmax (maybe you've just got unlucky)"
#
# # test for lookahead
# dummy_input_ix_2 = torch.as_tensor(to_matrix([line[:3] + 'e' * (len(line) - 3) for line in dummy_lines]), device=DEVICE)
# dummy_logits_2 = dummy_model(dummy_input_ix_2)

# assert torch.allclose(dummy_logits[:, :3], dummy_logits_2[:, :3]), "your model's predictions depend on FUTURE tokens. " \
#     " Make sure you don't allow any layers to look ahead of current token." \
#     " You can also get this error if your model is not deterministic (e.g. dropout). Disable it for this test."


# IMPORTANT TRICK: compute mask for padding
def compute_mask(input_ix, eos_ix=token_to_id[EOS]):
    """ compute a boolean mask that equals "1" until first EOS (including that EOS) """
    mask = F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True).to(DEVICE)
    return mask


# print('matrix:\n', dummy_input_ix.cpu().numpy())
# print('mask:', compute_mask(dummy_input_ix).to(torch.int32).cpu().numpy())
# print('lengths:', compute_mask(dummy_input_ix).sum(-1).cpu().numpy())


def compute_loss(model, input_matrix):
    # calculate NLL (negative-log-likelihood)
    input_matrix = torch.as_tensor(input_matrix, dtype=torch.int64, device=DEVICE)
    pred = model(input_matrix[:, :-1])  # (batch_size, max_len-1, n_tokens)
    target = input_matrix[:, 1:]  # (batch_size, max_len-1)
    prob = F.log_softmax(pred, dim=-1)
    mask = compute_mask(input_matrix[:, 1:]).to(DEVICE)
    pred_gather_masked = torch.gather(prob, dim=-1, index=target.unsqueeze(dim=-1)).squeeze(-1) * mask  # find target-corresponding pred, gather them up
    loss_batch = -pred_gather_masked.sum()
    total_loss = loss_batch/input_matrix.shape[0]

    return total_loss


# loss_1 = compute_loss(dummy_model, to_matrix(dummy_lines, max_len=15))
# loss_2 = compute_loss(dummy_model, to_matrix(dummy_lines, max_len=16))
# assert (np.ndim(loss_1) == 0) and (0 < loss_1 < 100), "loss must be a positive scalar"
# assert torch.allclose(loss_1, loss_2), 'do not include  AFTER first EOS into loss. '\
#     'Hint: use compute_mask. Beware +/-1 errors. And be careful when averaging!'


def score_lines(model, dev_lines, batch_size):
    """ compute the average loss over entire dataset """
    dev_loss_num, dev_loss_len = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(dev_lines), batch_size):
            batch_idx = to_matrix(dev_lines[i:i+batch_size])
            dev_loss_num += compute_loss(model, batch_idx).item() * len(batch_idx)
            dev_loss_len += len(batch_idx)

    return dev_loss_num/dev_loss_len


def generate(model, prefix=BOS, temperature=1.0, max_len=100):
    with torch.no_grad():
        while True:
            token_probs = model.get_possible_next_tokens(prefix)
            tokens, probs = zip(*token_probs.items())
            if temperature == 0.0:
                next_token = tokens[np.argmax(probs)]
            else:
                probs = np.array([p**(1.0/temperature) for p in probs])
                probs /= sum(probs)
                next_token = np.random.choice(tokens, p=probs)
            prefix += next_token
            if next_token == EOS or len(prefix) > max_len:
                break
    return prefix


#######################
# CNN
#######################

# train_lines, dev_lines = train_test_split(lines, test_size=0.25, random_state=42)
# batch_size = 256
# EPOCH = 30
# batch_num = len(train_lines)//batch_size
# print("batch num: ", batch_num)
# score_dev_every = 500
# print_every_epoch = 5
# model = FixedWindowLM().to(device=DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#
# print("sample before training: ", generate(model, "Bridging"))
#
# train_history, dev_history = [], []
# for epoch in range(EPOCH):
#     for i in trange(5000):
#         batch = to_matrix(sample(train_lines, batch_size))
#         loss_i = compute_loss(model, batch)
#         optimizer.zero_grad()
#         loss_i.backward()
#         optimizer.step()
#         if i == 0:
#             train_history.append((epoch, loss_i.item()))
#
#     print("Generated examples (tam=0.5):")
#     for _ in range(3):
#         print(generate(model, temperature=0.5))
#     print('scoring dev...')
#     dev_history.append((epoch, score_lines(model, dev_lines, batch_size)))
#     print('#%i dev loss: %.3f' % dev_history[-1])
#
# plt.plot(*zip(*train_history), label='train', color='red')
# plt.plot(*zip(*dev_history), label='dev', color='blue')
# plt.savefig("result")
#
# assert np.mean(train_history[:10], axis=0)[1] > np.mean(train_history[-10:], axis=0)[1], "The model didn't converge."
# print("Final dev loss:", dev_history[-1][-1])
#
# for i in range(10):
#     print(generate(model, temperature=0.5))


#################################
# RNN/LSTM/GRU
#################################

class RNNLM(nn.Module):
    def __init__(self, n_tokens=n_tokens, emb_size=16, hid_size=64):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hid_size, batch_first=True)
        self.linear = nn.Linear(in_features=hid_size, out_features=n_tokens)

    def __call__(self, input_matrix):
        """
        :param input_matrix: (batch_size, seq_len)
        :return: (batch_size, seq_len, n_tokens)
        """
        embeded = self.embedding(input_matrix)  # (batch_size, emb_size, seq_len)
        hidden_state, cell_state = self.lstm(embeded)  # (batch_size, seq_len, hid_size)
        output = self.linear(hidden_state)  # (batch_size, seq_len, n_tokens)
        return output

    def get_possible_next_tokens(self, prefix=BOS, temperature=1.0, max_len=100):
        prefix_idx = torch.as_tensor(to_matrix([prefix]), dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self(prefix_idx)[0, -1], dim=-1).cpu().numpy()
        return dict(zip(tokens, probs))


# dummy_model = RNNLM().to(DEVICE)
# dummy_input_ix = torch.as_tensor(to_matrix(dummy_lines), device=DEVICE)
# dummy_logits = dummy_model(dummy_input_ix)
# assert isinstance(dummy_logits, torch.Tensor)
# assert dummy_logits.shape == (len(dummy_lines), max(map(len, dummy_lines)), n_tokens), "please check output shape"
# assert not np.allclose(dummy_logits.cpu().data.numpy().sum(-1), 1), "please predict linear outputs, don't use softmax (maybe you've just got unlucky)"
# print('Weights:', tuple(name for name, w in dummy_model.named_parameters()))
#
# dummy_input_ix_2 = torch.as_tensor(to_matrix([line[:3] + 'e' * (len(line) - 3) for line in dummy_lines]), device=DEVICE)
# dummy_logits_2 = dummy_model(dummy_input_ix_2)
#
# assert torch.allclose(dummy_logits[:, :3], dummy_logits_2[:, :3]), "your model's predictions depend on FUTURE tokens. " \
#     " Make sure you don't allow any layers to look ahead of current token." \
#     " You can also get this error if your model is not deterministic (e.g. dropout). Disable it for this test."

train_lines, dev_lines = train_test_split(lines, test_size=0.25, random_state=42)
batch_size = 64
EPOCH = 30
score_dev_every = 250
train_history, dev_history = [], []
model = RNNLM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dev_history.append((0, score_lines(model, dev_lines, batch_size)))
print("sample before training: ", generate(model, "Bridging"))

for epoch in range(EPOCH):
    for i in trange(len(train_history), 5000):
        batch = to_matrix(sample(train_lines, batch_size))
        loss_i = compute_loss(model, batch)
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()
        if i == 0:
            train_history.append((i, loss_i.item()))

    print("Generated examples (tam=0.5):")
    for _ in range(3):
        print(generate(model, temperature=0.5))
    print('scoring dev...')
    dev_history.append((epoch, score_lines(model, dev_lines, batch_size)))
    print('#%i dev loss: %.3f' % dev_history[-1])

plt.plot(*zip(*train_history), label='train', color='red')
plt.plot(*zip(*dev_history), label='dev', color='blue')
plt.savefig("result-LSTM")

assert np.mean(train_history[:10], axis=0)[1] > np.mean(train_history[-10:], axis=0)[1], "The model didn't converge."
print("Final dev loss:", dev_history[-1][-1])
for i in range(10):
    print(generate(model, temperature=0.5))


def generate_nucleus(model, prefix=BOS, nucleus=0.9, max_len=100):
    """
    Remove the non-sense words without sacrificing diversity.
    Nucleus Sampling: sample top-N% fraction from prob mass.
    """
    while True:
        tokens_probs: dict = model.get_possible_next_tokens(prefix)
        sorted_token_prob = {k: v for k, v in sorted(tokens_probs.items(), key=lambda item: item[1], reverse=True)}
        sorted_p = [p for p in sorted_token_prob.values()]
        padded_sorted_p = np.insert(sorted_p, 0, 0.0)
        sorted_token = [t for t in sorted_token_prob.keys()]
        cumsum_p = np.cumsum(padded_sorted_p)
        mask = (cumsum_p < nucleus)[:-1]
        masked_prob = sorted_p * mask
        norm_prob = masked_prob / sum(masked_prob)
        next_token = np.random.choice(sorted_token, p=norm_prob)

        prefix += next_token
        if next_token == EOS or len(prefix) > max_len:
            break
    return prefix


for i in range(10):
    print(generate_nucleus(model, nucleus=0.5))


