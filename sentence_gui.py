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
    


# ------------------ UYGULAMA GUI ------------------

class SentenceCorrectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentence Correction")
        self.setGeometry(100, 100, 900, 700)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.init_model()
        self.initUI()
        self.tool = language_tool_python.LanguageTool('en-US')

    def init_model(self):
        try:
            with open('vocabulary_200k.pkl', 'rb') as f:
                self.vocab = pickle.load(f)

            checkpoint = torch.load('spell_corrector_200k_best.pth', map_location=self.device, weights_only=True)


            self.model = Seq2SeqModel(len(self.vocab.word2idx)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✅ Model başarıyla yüklendi!")
            self.model_loaded = True
        except Exception as e:
            print("❌ Model yüklenemedi! Lütfen model dosyalarını kontrol edin.")
            print("Hata detayı:", e)
            self.model_loaded = False

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(20)

        # Logo ve başlık için dikey layout - görüntüdeki gibi
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(10)

        # Logo - 200x200 boyutta ve merkezde
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        try:
            pixmap = QPixmap("sentencelogo.jpg")
            if not pixmap.isNull():
                # Logo boyutunu 200x200 yap
                scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                # Logo bulunamazsa büyük emoji kullan
                logo_label.setText("🛡️📝")
                logo_label.setFont(QFont("Segoe UI", 64))
        except:
            # Hata durumunda büyük emoji kullan
            logo_label.setText("🛡️📝")
            logo_label.setFont(QFont("Segoe UI", 64))

        header_layout.addWidget(logo_label)

        layout.addLayout(header_layout)

        # "Enter Sentences:" etiketi
        word_label = QLabel("Enter Sentences:")
        word_label.setFont(QFont("Segoe UI", 12))
        word_label.setStyleSheet("color: #34495e; margin-top: 20px;")
        layout.addWidget(word_label)

        # Input alanı
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter or load your sentence here...")
        self.input_text.setFont(QFont("Consolas", 11))
        self.input_text.setMaximumHeight(100)  # Yüksekliği sınırla
        layout.addWidget(self.input_text)

        # Butonlar
        button_layout = QHBoxLayout()

        self.correct_button = QPushButton("✅ Correct Sentence")
        self.correct_button.clicked.connect(self.correct_text)
        self.correct_button.setStyleSheet(
            "background-color:#27ae60;color:white;padding:10px;border:none;border-radius:6px")
        button_layout.addWidget(self.correct_button)

        clear_button = QPushButton("🗑️ Clear Text")
        clear_button.clicked.connect(self.clear_text)
        clear_button.setStyleSheet("background-color:#e74c3c;color:white;padding:10px;border:none;border-radius:6px")
        button_layout.addWidget(clear_button)

        load_pdf_btn = QPushButton("📄 Upload PDF")
        load_pdf_btn.clicked.connect(self.load_pdf)
        load_pdf_btn.setStyleSheet("background-color:#3498db;color:white;padding:10px;border:none;border-radius:6px")
        button_layout.addWidget(load_pdf_btn)

        load_word_btn = QPushButton("📄 Upload Word")
        load_word_btn.clicked.connect(self.load_word)
        load_word_btn.setStyleSheet("background-color:#8e44ad;color:white;padding:10px;border:none;border-radius:6px")
        button_layout.addWidget(load_word_btn)

        layout.addLayout(button_layout)


        # "Suggestions:" etiketi
        result_label = QLabel("Suggestions:")
        result_label.setFont(QFont("Segoe UI", 12))
        result_label.setStyleSheet("color: #34495e; margin-top: 20px;")
        layout.addWidget(result_label)

        # Output alanı - beyaz arka plan
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 11))
        self.output_text.setPlaceholderText("Corrected text will appear here...")
        layout.addWidget(self.output_text)

        self.setLayout(layout)

        # Beyaz arka plan için stylesheet ekle
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: "Segoe UI", sans-serif;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton {
                padding: 10px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
        """)

        back_button = QPushButton("⬅️ Back to Main Menu")
        back_button.clicked.connect(self.go_back_to_main)
        back_button.setStyleSheet("background-color:#95a5a6;color:white;padding:10px;border:none;border-radius:6px")
        layout.addWidget(back_button)



    def correct_text(self):
        raw = self.input_text.toPlainText().strip()
        if not raw:
            return

        # Model yüklü değilse sadece language_tool kullan
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            print("⚠️ Model yüklü değil! Sadece temel dilbilgisi kontrolü yapılıyor...")
            matches = self.tool.check(raw)
            corrected = language_tool_python.utils.correct(raw, matches)
            self.output_text.setText(corrected)
            return

        print("🔄 AI model ile cümle düzeltiliyor...")
        # Model yüklüyse normal işlemi yap
        sentences = re.split(r'(?<=[.!?])\s+', raw)
        corrected = []
        for sentence in sentences:
            result = self.correct_sentence(sentence)
            corrected.append(result)
        final = ' '.join(corrected)

        self.output_text.setText(final)
        print("✅ Cümle düzeltme işlemi tamamlandı!")

    def correct_sentence(self, sentence, max_length=50):
        if not sentence or len(sentence.strip().split()) <= 2:
            # Kısa girişlerde seq2seq yerine language_tool kullan
            matches = self.tool.check(sentence)
            return language_tool_python.utils.correct(sentence, matches)

        indices = self.vocab.sentence_to_indices(sentence)
        if not indices:
            return sentence

        src_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        encoder_outputs, hidden, cell = self.model(src_tensor)
        input_token = torch.tensor([[self.vocab.SOS_IDX]], dtype=torch.long).to(self.device)

        output_indices = []
        seen_tokens = set()
        repeat_count = 0
        last_token = None

        for _ in range(max_length):
            prediction, hidden, cell, _ = self.model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = prediction.argmax(1).item()

            if top1 == self.vocab.EOS_IDX:
                break
            if top1 == last_token:
                repeat_count += 1
                if repeat_count >= 5:
                    break
            else:
                repeat_count = 0
            last_token = top1

            output_indices.append(top1)
            input_token = torch.tensor([[top1]], dtype=torch.long).to(self.device)

        result = self.vocab.indices_to_sentence(output_indices)
        result = result.capitalize()
        result = re.sub(r'\s+([.!?,])', r'\1', result)

        # Eğer sonuç çok saçma şekilde uzunsa, yine language_tool ile düzelt
        if len(result.split()) > 30:
            matches = self.tool.check(sentence)
            return language_tool_python.utils.correct(sentence, matches)

        return result


    def clear_text(self):
        """Input ve output alanlarını temizle"""
        self.input_text.clear()
        self.output_text.clear()
        print("🗑️ Metin alanları temizlendi!")

    def load_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if path:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            self.input_text.setText(text.strip())

    def load_word(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Word File", "", "Word Files (*.docx)")
        if path:
            doc = docx.Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])
            self.input_text.setText(text.strip())

    def go_back_to_main(self):
        import subprocess
        import os
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        subprocess.Popen(["python", "arayuz2.py"], env=env)
        self.close()


# ------------------ MAIN ------------------


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentenceCorrectorApp()
    window.show()
    sys.exit(app.exec_())
