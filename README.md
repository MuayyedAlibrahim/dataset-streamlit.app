🩺 Diyabet Tahmin Uygulaması

Bu Streamlit tabanlı uygulama, kullanıcıdan alınan sağlık verileriyle diyabet riski tahmini yapar.  
Makine öğrenmesi modeli olarak Random Forest kullanılmıştır.

📌 Özellikler

- Veri görselleştirme (diyabet dağılımı, yaş, cinsiyet, BMI, HbA1c, kan şekeri vb.)
- Random Forest modeli ile tahmin
- Kullanıcıdan veri girişi alarak anlık tahmin
- Özellik önemliliği görselleştirmesi
- Türkçe dil desteği
  
🧠 Kullanılan Teknolojiler

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

🗂️ Dosyalar

- `app.py`: Uygulamanın ana kod dosyası
- `diabetes_prediction_dataset.csv`: Eğitim ve tahmin için kullanılan veri seti
- `requirements.txt`: Gerekli Python kütüphaneleri

🚀 Uygulama Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
````

2. Uygulamayı çalıştırın:

```bash
streamlit run app.py
```

🌍 Canlı Yayın

Uygulamanın canlı halini görmek için şu bağlantıya göz atabilirsiniz (yayına aldıktan sonra):

https://diyabet-tahmin.streamlit.app

📬 İletişim

Herhangi bir soru veya öneri için:muayyedalibrahim@gmail.com
