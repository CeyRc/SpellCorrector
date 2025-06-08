# vocab_module.py
import re
import pandas as pd
from collections import Counter

MIN_COUNT = 1

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()

        # Özel tokenler
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        # Özel token indeksleri
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        self._init_vocab()

    def _init_vocab(self):
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def add_sentence(self, sentence):
        words = self.tokenize(sentence)
        for word in words:
            self.word_count[word] += 1

    def tokenize(self, sentence):
        if pd.isna(sentence) or not isinstance(sentence, str):
            return []
        sentence = sentence.lower().strip()
        sentence = re.sub(r'([.!?,:;])', r' \1 ', sentence)
        words = sentence.split()
        return [word for word in words if word.strip()]

    def build_vocab(self, min_count=MIN_COUNT):
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_count and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"📚 Vocabulary boyutu: {len(self.word2idx):,}")

    def sentence_to_indices(self, sentence, add_sos_eos=False):
        words = self.tokenize(sentence)
        indices = []

        if add_sos_eos:
            indices.append(self.SOS_IDX)

        for word in words:
            indices.append(self.word2idx.get(word, self.UNK_IDX))

        if add_sos_eos:
            indices.append(self.EOS_IDX)

        return indices

    def indices_to_sentence(self, indices):
        words = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx not in [self.PAD_IDX, self.SOS_IDX]:
                words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return ' '.join(words)
