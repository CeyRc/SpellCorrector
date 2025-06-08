import pandas as pd

# CSV dosyasının yolunu belirt
dosya_yolu = 'final_spell_dataset.csv'

# CSV dosyasını oku (ilk 200000 satırı al)
df = pd.read_csv(dosya_yolu, nrows=200000)

# Yeni, küçültülmüş CSV dosyasını kaydet
df.to_csv('sentence_200k.csv', index=False)

print("İlk 200000 satır başarıyla alındı ve yeni dosya oluşturuldu.")