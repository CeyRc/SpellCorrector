import pandas as pd
import random

# Dosyayı yükle
file_path = 'cleaned_only_english_sentences.csv'  # Kendi dosya yolunuzu buraya ekleyin
data = pd.read_csv(file_path)


# Hata oluşturma yardımcı fonksiyonları
def add_spelling_error(word):
    """Kelime üzerinde yazım hatası oluşturur."""
    if len(word) <= 1:
        return word  # Çok kısa kelimelerde hata yapmayalım

    error_type = random.choice(['replace', 'delete', 'insert', 'transpose'])
    if error_type == 'replace':  # Harf değiştirme
        idx = random.randint(0, len(word) - 1)
        replacement = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:idx] + replacement + word[idx + 1:]
    elif error_type == 'delete':  # Harf eksiltme
        idx = random.randint(0, len(word) - 1)
        word = word[:idx] + word[idx + 1:]
    elif error_type == 'insert':  # Harf ekleme
        idx = random.randint(0, len(word))
        addition = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:idx] + addition + word[idx:]
    elif error_type == 'transpose':  # Harf yer değiştirme
        if len(word) > 1:
            idx = random.randint(0, len(word) - 2)
            word = (word[:idx] + word[idx + 1] + word[idx] +
                    word[idx + 2:])
    return word


def add_typo_based_on_keyboard(word):
    """Kelimeye klavye komşuluğuna bağlı hata ekler."""
    keyboard_neighbors = {
        'q': 'w', 'w': 'qe', 'e': 'wr', 'r': 'et', 't': 'ry', 'y': 'tu', 'u': 'yi', 'i': 'uo',
        'o': 'ip', 'p': 'o', 'a': 's', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh', 'h': 'gj',
        'j': 'hk', 'k': 'jl', 'l': 'k', 'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn',
        'n': 'bm', 'm': 'n'
    }
    if len(word) <= 1:
        return word
    idx = random.randint(0, len(word) - 1)
    if word[idx] in keyboard_neighbors:
        typo_char = random.choice(keyboard_neighbors[word[idx]])
        word = word[:idx] + typo_char + word[idx + 1:]
    return word


def add_phonetic_error(word):
    """Kelimeye fonetik hata ekler."""
    phonetic_pairs = {'their': 'there', 'there': 'their', 'write': 'right', 'right': 'write',
                      'bare': 'bear', 'bear': 'bare'}
    return phonetic_pairs.get(word.lower(), word)


def split_word(word):
    """Kelimeyi böler."""
    if len(word) > 4:
        split_point = random.randint(1, len(word) - 2)
        return word[:split_point] + ' ' + word[split_point:]
    return word


def create_noisy_word(word):
    """Bir kelimeye rastgele hata ekler."""
    if random.random() < 0.2:
        return add_typo_based_on_keyboard(word)
    elif random.random() < 0.2:
        return add_phonetic_error(word)
    elif random.random() < 0.1:
        return split_word(word)
    else:
        return add_spelling_error(word)


def add_grammatical_error(sentence):
    """Cümle üzerinde gramer hatası oluşturur."""
    words = sentence.split()
    error_type = random.choice(['tense', 'auxiliary', 'preposition', 'article', 'agreement'])

    if error_type == 'tense' and 'is' in words:  # Zaman uyumsuzluğu
        idx = words.index('is')
        words[idx] = 'was'
    elif error_type == 'auxiliary' and 'can' in words:  # Yardımcı fiil hatası
        idx = words.index('can')
        words.insert(idx + 1, 'be')
    elif error_type == 'preposition' and 'at' in words:  # Preposition hatası
        idx = words.index('at')
        words[idx] = 'in'
    elif error_type == 'article':  # Article eksikliği ya da fazlalığı
        if 'a' in words:
            words.remove('a')
        else:
            words.insert(0, 'a')
    elif error_type == 'agreement' and 'is' in words:  # Subject-verb uyumsuzluğu
        idx = words.index('is')
        words[idx] = 'are'
    return ' '.join(words)


def add_additional_grammatical_error(sentence):
    """Cümleye ekstra gramer hatası ekler."""
    words = sentence.split()
    error_type = random.choice(['negation', 'word_order', 'plurality'])

    if error_type == 'negation':  # Olumsuzluk ekleme ya da fazlalığı
        if 'not' in words:
            words.remove('not')
        else:
            words.insert(random.randint(0, len(words)), 'not')
    elif error_type == 'word_order':  # Kelime sırası bozulması
        if len(words) > 2:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
    elif error_type == 'plurality':  # Tekil-çoğul uyumsuzluğu
        if 'is' in words:
            idx = words.index('is')
            words[idx] = 'are'
        elif 'are' in words:
            idx = words.index('are')
            words[idx] = 'is'
    return ' '.join(words)


def create_diverse_noisy_sentence(sentence):
    """
    Gelişmiş kelime ve gramer hatalarını uygulayarak bir cümleyi bozar.
    """
    words = sentence.split()

    # Kelime bazlı hataları daha çeşitlendirilmiş bir şekilde uygula
    noisy_words = [create_noisy_word(word) if random.random() < 0.4 else word for word in words]
    noisy_sentence = ' '.join(noisy_words)

    # Gramer hatalarını iki aşamada uygula
    if random.random() < 0.5:
        noisy_sentence = add_grammatical_error(noisy_sentence)
    if random.random() < 0.5:
        noisy_sentence = add_additional_grammatical_error(noisy_sentence)

    return noisy_sentence


# Yeni sütunları oluştur
data['correct_sentence'] = data['sentence']
data['noisy_sentence'] = data['sentence'].apply(create_diverse_noisy_sentence)

# Sütunları yeniden sıralama
output_data = data[['noisy_sentence', 'correct_sentence']]

# Yeni dosyayı kaydet
output_data.to_csv('noisy_correct_sentences.csv', index=False)
print("Dosya başarıyla oluşturuldu: noisy_correct_sentences.csv")

# Dosyayı kaydet
#output_path = 'noisy_sentences.csv'  # Çıkış dosya adı
#data.to_csv(output_path, index=False)

#print(f"Noisy sentences dataset saved to: {output_path}")
