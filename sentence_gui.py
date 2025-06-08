import sys
import re
import pickle
import torch
import fitz  # PyMuPDF
import docx

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import language_tool_python


# ------------------ MODEL COMPONENTS------------------

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        self._init_vocab()

    def _init_vocab(self):
        self.word2idx = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX,
            '< SOS >': self.SOS_IDX,
            '<EOS>': self.EOS_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def tokenize(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'([.!?,:;()"\'])', r' \1 ', sentence)
        return [word for word in sentence.split() if word.strip()]

    def sentence_to_indices(self, sentence):
        words = self.tokenize(sentence)
        return [self.word2idx.get(w, self.UNK_IDX) for w in words]

    def indices_to_sentence(self, indices):
        words = [self.idx2word.get(i, '<UNK>') for i in indices if i not in [self.PAD_IDX, self.SOS_IDX, self.EOS_IDX]]
        return ' '.join(words)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                  bidirectional=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = torch.nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.out(output.squeeze(1))
        return prediction, hidden, cell, attn_weights


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)
        self.hidden_dim = 512
        self.num_layers = 2

    def forward(self, src):
        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = hidden.view(self.num_layers, 2, 1, self.hidden_dim)[:, 0, :, :]
        cell = cell.view(self.num_layers, 2, 1, self.hidden_dim)[:, 0, :, :]
        return encoder_outputs, hidden, cell