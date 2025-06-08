import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFrame, QSizePolicy)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer

import json
from collections import Counter

# Model dosyasının yolu
MODEL_PATH = 'spell_corrector_word_model.json'
MAX_SUGGESTIONS = 5  # öneri sayısı
IMAGE_PATH = 'wordlogo.jpg'
SUGGESTION_DELAY = 300  # ms cinsinden gecikme

def calculate_distances(word1, word2):
    """İki kelime arasındaki Levenshtein mesafesini hesaplar."""
    m = len(word1)
    n = len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def find_closest_words_with_freq(input_word, correct_words, word_counts, n=MAX_SUGGESTIONS):
    """Verilen kelimeye en yakın N adet doğru kelimeyi frekans bilgisiyle bulur."""
    distances = []
    for correct_word in correct_words:
        distance = calculate_distances(input_word, correct_word)
        distances.append((distance, correct_word))

    # Mesafeye göre sırala, ardından frekansa göre (tersine) sırala
    ranked_suggestions = sorted(distances, key=lambda item: (item[0], -word_counts.get(item[1], 0)))

    return [word for dist, word in ranked_suggestions[:n]]

def load_model(load_path):
    """Kaydedilmiş modelden doğru kelime listesini ve frekanslarını yükler."""
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        return set(model_data['correct_words']), Counter(model_data['word_counts'])
    except FileNotFoundError:
        print(f"Error: Model file not found: {load_path}")
        return set(), Counter()
    except Exception as e:
        print(f"Error: An error occurred while loading the model: {e}")
        return set(), Counter()

class SpellCorrectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spell Checker")
        self.setGeometry(100, 100, 700, 600)
        self.correct_words, self.word_counts = load_model(MODEL_PATH)
        self.suggestion_buttons = []
        self.suggestion_timer = QTimer()
        self.suggestion_timer.timeout.connect(self.update_suggestions)
        self.initUI()
        self.setStyleSheet("""
            QWidget {
                background-color: white; /* Arka planı beyaz yaptık */
                font-family: "Segoe UI", sans-serif;
                font-size: 16px;
            }
            QLabel {
                font-size: 18px;
                color: #333;
                margin-bottom: 5px;
            }
            QLineEdit {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 16px;
                margin-bottom: 15px;
            }
            QPushButton#check_button {
                background-color: #8e44ad; /* Mor renk */
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 6px;
                font-size: 16px;
                margin-bottom: 20px;
            }
            QPushButton#check_button:hover {
                background-color: #7d3c98;
            }
            QFrame#suggestions_frame {
                background-color: #e0e0e0;
                border-radius: 6px;
                margin-top: 15px;
                padding: 15px;
            }
            QPushButton#suggestion_button {
                background-color: #fff;
                color: #333;
                border: 1px solid #ccc;
                padding: 10px 20px;
                border-radius: 6px;
                margin-right: 10px;
                font-size: 16px;
            }
            QPushButton#suggestion_button:hover {
                background-color: #f8f8f8;
            }
            QLabel#logo_label {
                margin-bottom: 20px;
            }
        """)

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setSpacing(20)

        # Logo Label'ı oluştur ve resmi yükle
        self.logo_label = QLabel()
        self.logo_label.setObjectName("logo_label")
        pixmap = QPixmap(IMAGE_PATH)
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        self.logo_label.setPixmap(scaled_pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.logo_label)

        input_layout = QHBoxLayout()
        self.label = QLabel("Enter Word:")
        self.entry = QLineEdit()
        self.entry.textChanged.connect(self.start_suggestion_timer)
        input_layout.addWidget(self.label)
        input_layout.addWidget(self.entry)
        main_layout.addLayout(input_layout)

        self.suggestions_label = QLabel("Suggestions:")
        main_layout.addWidget(self.suggestions_label)

        self.suggestions_frame = QFrame()
        self.suggestions_frame.setObjectName("suggestions_frame")
        self.suggestions_layout = QHBoxLayout()
        self.suggestions_layout.setSpacing(10)
        self.suggestions_frame.setLayout(self.suggestions_layout)
        main_layout.addWidget(self.suggestions_frame)

        main_layout.addStretch(1)

        self.setLayout(main_layout)

        # BACK BUTONU
        back_button = QPushButton("⬅️ Back to Main Menu")
        back_button.setStyleSheet("background-color:#95a5a6;color:white;padding:10px;border:none;border-radius:6px")
        back_button.clicked.connect(self.go_back_to_main)
        main_layout.addWidget(back_button)


    def start_suggestion_timer(self):
        """Kullanıcı yazmayı bıraktıktan sonra önerileri göstermek için timer'ı başlatır."""
        self.suggestion_timer.start(SUGGESTION_DELAY)

    def update_suggestions(self):
        """Gecikmenin ardından önerileri günceller."""
        self.suggestion_timer.stop()
        self.show_suggestions()

    def show_suggestions(self):
        input_word = self.entry.text().strip().lower()
        self._clear_suggestions()

        if not self.correct_words:
            no_model_label = QLabel("Model could not be loaded.")
            self.suggestions_layout.addWidget(no_model_label)
            return

        if not input_word:
            return  # Kullanıcı hiçbir şey yazmadıysa öneri gösterme

        suggestions = find_closest_words_with_freq(input_word, self.correct_words, self.word_counts)
        if suggestions:
            # Eğer girilen kelime doğruysa, öneri listesine ekle
            if input_word in self.correct_words and input_word not in suggestions:
                suggestions.insert(0, input_word)
                suggestions = suggestions[:MAX_SUGGESTIONS] # Maksimum öneri sayısını koru

            for suggestion in suggestions:
                button = QPushButton(suggestion)
                button.setObjectName("suggestion_button")
                button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                button.clicked.connect(lambda checked, word=suggestion: self.select_suggestion(word))
                self.suggestions_layout.addWidget(button)
                self.suggestion_buttons.append(button)
        else:
            no_suggestion_label = QLabel(f"No suggestions found for '{input_word}'.")
            self.suggestions_layout.addWidget(no_suggestion_label)

    def _clear_suggestions(self):
        for button in self.suggestion_buttons:
            button.deleteLater()
        self.suggestion_buttons = []

        for i in reversed(range(self.suggestions_layout.count())):
            widget = self.suggestions_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def select_suggestion(self, selected_word):
        self.entry.setText(selected_word)
        self._clear_suggestions() # Öneri seçildikten sonra temizle

    def go_back_to_main(self):
        import subprocess
        import os
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        subprocess.Popen(["python", "arayuz2.py"], env=env)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpellCorrectorApp()
    window.show()
    sys.exit(app.exec_())