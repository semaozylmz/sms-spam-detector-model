# SMS Spam Tespit Modeli (SMS Spam Detection Model)

Bu proje, Scikit-learn kütüphanesi kullanılarak geliştirilmiş, İngilizce SMS metinlerini **spam** veya **ham** (spam olmayan) olarak sınıflandıran bir makine öğrenmesi modelidir.

> Bu model, [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) veri seti üzerinde eğitilmiştir.

## ✨ Temel Özellikler

- **Yüksek Doğruluk:** Test verisi üzerinde **~%97** doğruluk oranına sahiptir.
- **Kullanıcı Odaklı:** Normal mesajları spam olarak etiketleme oranı sıfırdır (**0 False Positive**). Bu sayede önemli bir mesajın spam kutusuna düşme riskini en aza indirir.
- **Hafif ve Hızlı:** Model dosyaları sadece birkaç megabayttır ve tahmin işlemi milisaniyeler içinde gerçekleşir. Gerçek zamanlı uygulamalar için uygundur.
- **Basit Entegrasyon:** Herhangi bir Python projesine kolayca entegre edilebilir.

## 📂 Proje Yapısı

```
.
├── spam_model.pkl          # Eğitilmiş Naive Bayes modeli
├── vectorizer.pkl          # Eğitilmiş TF-IDF vektörleştiricisi
├── spam.csv 
├── Spam_SMS_Collection.ipynb # Modelin geliştirildiği ve eğitildiği Jupyter Notebook
└── README.md               
```

## 🚀 Kurulum ve Başlangıç

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

**1. Projeyi Klonlayın:**
```bash
git clone [https://github.com/](https://github.com/)[KULLANICI_ADIN]/[REPO_ADIN].git
cd [REPO_ADIN]
```

**2. Gerekli Kütüphaneleri Yükleyin:**
Proje bağımlılıklarını `requirements.txt` dosyasını kullanarak yüklemeniz önerilir.
```bash
pip install -r requirements.txt
```
Eğer `requirements.txt` dosyanız yoksa, aşağıdaki kütüphaneleri manuel olarak yükleyebilirsiniz:
```bash
pip install pandas numpy scikit-learn nltk
```

**3. NLTK Stopwords İndirmesi:**
Modelin metin temizleme adımı için NLTK'nın `stopwords` listesine ihtiyacı vardır. Aşağıdaki komutları bir Python yorumlayıcısında çalıştırarak listeyi indirin:
```python
import nltk
nltk.download('stopwords')
```

## 💻 Kullanım Örneği

Modeli kendi projenizde kullanmak çok kolaydır. Aşağıdaki `example.py` dosyasındaki gibi, kaydedilmiş `vectorizer.pkl` ve `spam_model.pkl` dosyalarını yükleyerek tahmin yapabilirsiniz.

```python
# example.py
import pickle

def predict_spam(message: str) -> str:
    """
    Verilen tek bir mesaj için spam tahmini yapar.
    """
    try:
        # Kaydedilmiş vektörleştiriciyi ve modeli yükle
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return "Hata: Model dosyaları ('vectorizer.pkl', 'spam_model.pkl') bulunamadı."

    # Gelen mesajı vektörleştirici ile dönüştür (bir liste içinde olmalı)
    message_tfidf = vectorizer.transform([message])
    
    # Tahmini yap
    prediction = model.predict(message_tfidf)
    
    return prediction[0].upper()

# --- Test ---
spam_mesaj = "Congratulations! You've won a $1000 Walmart gift card. Go to [http://example.com](http://example.com) to claim now."
normal_mesaj = "Hi mom, I'll be home late for dinner tonight. Can you save me some food?"

print(f"Mesaj: '{spam_mesaj}'")
print(f"  -> Tahmin: {predict_spam(spam_mesaj)}\n")

print(f"Mesaj: '{normal_mesaj}'")
print(f"  -> Tahmin: {predict_spam(normal_mesaj)}")
```

## 📊 Model Performansı

Model, test verisinin %20'si üzerinde değerlendirilmiştir. Elde edilen sonuçlar:

| Metrik | Değer |
| :--- | :--- |
| **Doğruluk (Accuracy)** | %96.77 |
| **Normal Mesaj Başarısı (Ham Precision)** | %96 |
| **Spam Mesaj Başarısı (Spam Precision)** | %100 |
| **Normal Mesaj → Spam (False Positives)** | **0** |


## ⚠️ Sınırlamalar

- **Dil:** Model sadece **İngilizce** metinler için tasarlanmıştır.
- **Format:** En iyi performansı SMS gibi kısa metinlerde gösterir. Uzun e-postalar veya farklı formattaki metinler için performansı düşebilir.
- **Kullanım Amacı:** Modelin, mesajları otomatik olarak silmek yerine, kullanıcıya "spam olabilir" şeklinde bir uyarı göstermesi önerilir.

## 📄 Lisans

Bu proje **Apache 2.0** Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.
