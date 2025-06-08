import pandas as pd
import re

def clean_text(text):
    text = str(text)
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # URL temizle
    text = re.sub(r'\@w+|\#', '', text)  # @mention ve hashtag kaldır
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)  # Sadece harf, rakam ve noktalama tut
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
    return text

# Dosyayı oku
df = pd.read_csv("only_english_sentences.csv")

# Cümleleri temizle
df['sentence'] = df['sentence'].apply(clean_text)

# Temizlenmiş dosyayı kaydet
df.to_csv("cleaned_only_english_sentences.csv", index=False)
