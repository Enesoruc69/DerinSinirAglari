import numpy as np
import pickle
import os

# VERİ YOLLARINI TANIMLAMA
data_path = 'cifar-10-batches-py'

# VERİYİ YÜKLEME
# Eğitim verilerini birleştirme
X_train_list = []
y_train_list = []

for i in range(1, 6):
    file_path = os.path.join(data_path, f'data_batch_{i}')
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X_train_list.append(batch[b'data'])
        y_train_list.extend(batch[b'labels'])

X_train = np.concatenate(X_train_list)
y_train = np.array(y_train_list)

# Test verisini yükleme
test_file = os.path.join(data_path, 'test_batch')
with open(test_file, 'rb') as f:
    test_batch = pickle.load(f, encoding='bytes')
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

X_test = X_test[:100]
y_test = y_test[:100]

# KULLANICI ETKİLEŞİMİ
print("\n--- CIFAR-10 k-NN Ödevi  ---")
secim = input("Mesafe tipi seçin (L1 veya L2): ").strip().upper()
k = int(input("k değerini giriniz (Örn: 3, 5): "))

#  TAHMİNLEME DÖNGÜSÜ
tahminler = []

print(f"\n{len(X_test)} test örneği için hesaplama yapılıyor, lütfen bekleyin...")

for i in range(len(X_test)):
    # Test görüntüsü ile tüm eğitim seti arasındaki fark
    diff = X_train - X_test[i]

    if secim == 'L1':
        # L1 Mesafesi: Mutlak farkların toplamı
        distances = np.sum(np.abs(diff), axis=1)
    else:
        # L2 Mesafesi: Karelerin toplamının karekökü
        distances = np.sqrt(np.sum(np.square(diff), axis=1))

    # En yakın k adet komşuyu bul
    yakin_komsular = np.argsort(distances)[:k]
    k_etiketleri = y_train[yakin_komsular]

    # En çok tekrar eden sınıfı seç
    en_cok_tekrar_eden = np.bincount(k_etiketleri).argmax()
    tahminler.append(en_cok_tekrar_eden)

# SONUÇLAR
dogruluk = np.mean(tahminler == y_test) * 100
print("-" * 30)
print(f"Sonuç: %{dogruluk:.2f} doğruluk payı elde edildi.")
print("-" * 30)