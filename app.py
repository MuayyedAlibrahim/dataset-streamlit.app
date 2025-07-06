import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sayfa yapılandırması
st.set_page_config(page_title="Diyabet Tahmini", page_icon="🩺", layout="wide")

# Başlık ve açıklama
st.title("Diyabet Tahmin Uygulaması")
st.markdown("Bu uygulama, çeşitli sağlık parametrelerine dayanarak diyabet riski tahmini yapar.")

# Veri setini yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("data/diabetes_prediction_dataset.csv")
    # Sütun isimlerini Türkçe'ye çevirme
    df.rename(columns={
        'gender': 'cinsiyet',
        'age': 'yaş',
        'hypertension': 'hipertansiyon',
        'heart_disease': 'kalp_hastalığı',
        'smoking_history': 'sigara_geçmişi',
        'bmi': 'vücut_kitle_indeksi',
        'HbA1c_level': 'HbA1c_düzeyi',
        'blood_glucose_level': 'kan_şekeri_düzeyi',
        'diabetes': 'diyabet'
    }, inplace=True)
    
    # Veri tipi dönüşümleri
    df['yaş'] = df['yaş'].astype(int)
    
    # Kategorik değişkenleri sayısallaştırma
    df['cinsiyet'] = df['cinsiyet'].map({'Male': 1, 'Female': 0})
    df['sigara_geçmişi'] = LabelEncoder().fit_transform(df['sigara_geçmişi'])
    
    return df

# Veriyi yükle
df = load_data()

# Sidebar - Uygulama Seçenekleri
st.sidebar.title("Menü")
app_mode = st.sidebar.selectbox("Seçenek", ["Veri Görselleştirme", "Diyabet Tahmini"])

# Veri Görselleştirme Sayfası
if app_mode == "Veri Görselleştirme":
    st.header("Veri Görselleştirme")
    
    # Diyabet dağılımı
    st.subheader("Diyabet Dağılımı")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='diyabet', data=df, ax=ax)
        ax.set_title("Diyabet Dağılımı")
        ax.set_xlabel("Diyabet (0: Yok, 1: Var)")
        ax.set_ylabel("Kişi Sayısı")
        st.pyplot(fig)
    
    with col2:
        # Diyabet yüzdesi
        diyabet_yuzde = df['diyabet'].value_counts(normalize=True) * 100
        st.write(f"Diyabet Oranı: %{diyabet_yuzde[1]:.2f}")
        st.write(f"Sağlıklı Oranı: %{diyabet_yuzde[0]:.2f}")
    
    # Yaş ve Cinsiyet dağılımı
    col1, col2 = st.columns(2)
    
    with col1:
        # Yaş dağılımı
        st.subheader("Yaş Dağılımı")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['yaş'], bins=20, kde=True, ax=ax)
        ax.set_title("Yaş Dağılımı")
        ax.set_xlabel("Yaş")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        
        # Yaş istatistikleri
        st.write(f"Ortalama Yaş: {df['yaş'].mean():.2f}")
        st.write(f"Medyan Yaş: {df['yaş'].median()}")
        st.write(f"Minimum Yaş: {df['yaş'].min()}")
        st.write(f"Maksimum Yaş: {df['yaş'].max()}")
    
    with col2:
        # Cinsiyet dağılımı
        st.subheader("Cinsiyet Dağılımı")
        cinsiyet_mapping = {0: 'Kadın', 1: 'Erkek'}
        df_temp = df.copy()
        df_temp['cinsiyet_str'] = df_temp['cinsiyet'].map(cinsiyet_mapping)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='cinsiyet_str', data=df_temp, ax=ax)
        ax.set_title("Cinsiyet Dağılımı")
        ax.set_xlabel("Cinsiyet")
        ax.set_ylabel("Kişi Sayısı")
        st.pyplot(fig)
        
        # Cinsiyet yüzdesi
        cinsiyet_yuzde = df['cinsiyet'].value_counts(normalize=True) * 100
        st.write(f"Kadın Oranı: %{cinsiyet_yuzde[0]:.2f}")
        st.write(f"Erkek Oranı: %{cinsiyet_yuzde[1]:.2f}")
    
    # BMI ve HbA1c dağılımı
    col1, col2 = st.columns(2)
    
    with col1:
        # Vücut kitle indeksi dağılımı
        st.subheader("Vücut Kitle İndeksi Dağılımı")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['vücut_kitle_indeksi'], bins=20, kde=True, ax=ax)
        ax.set_title("Vücut Kitle İndeksi Dağılımı")
        ax.set_xlabel("Vücut Kitle İndeksi (BMI)")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        
        # BMI istatistikleri
        st.write(f"Ortalama BMI: {df['vücut_kitle_indeksi'].mean():.2f}")
        st.write(f"Medyan BMI: {df['vücut_kitle_indeksi'].median():.2f}")
        st.write(f"Minimum BMI: {df['vücut_kitle_indeksi'].min():.2f}")
        st.write(f"Maksimum BMI: {df['vücut_kitle_indeksi'].max():.2f}")
    
    with col2:
        # HbA1c düzeyi dağılımı
        st.subheader("HbA1c Düzeyi Dağılımı")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['HbA1c_düzeyi'], bins=20, kde=True, ax=ax)
        ax.set_title("HbA1c Düzeyi Dağılımı")
        ax.set_xlabel("HbA1c Düzeyi")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        
        # HbA1c istatistikleri
        st.write(f"Ortalama HbA1c: {df['HbA1c_düzeyi'].mean():.2f}")
        st.write(f"Medyan HbA1c: {df['HbA1c_düzeyi'].median():.2f}")
        st.write(f"Minimum HbA1c: {df['HbA1c_düzeyi'].min():.2f}")
        st.write(f"Maksimum HbA1c: {df['HbA1c_düzeyi'].max():.2f}")
    
    # Kan şekeri düzeyi ve Korelasyon matrisi
    col1, col2 = st.columns(2)
    
    with col1:
        # Kan şekeri düzeyi dağılımı
        st.subheader("Kan Şekeri Düzeyi Dağılımı")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['kan_şekeri_düzeyi'], bins=20, kde=True, ax=ax)
        ax.set_title("Kan Şekeri Düzeyi Dağılımı")
        ax.set_xlabel("Kan Şekeri Düzeyi")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        
        # Kan şekeri istatistikleri
        st.write(f"Ortalama Kan Şekeri: {df['kan_şekeri_düzeyi'].mean():.2f}")
        st.write(f"Medyan Kan Şekeri: {df['kan_şekeri_düzeyi'].median():.2f}")
        st.write(f"Minimum Kan Şekeri: {df['kan_şekeri_düzeyi'].min()}")
        st.write(f"Maksimum Kan Şekeri: {df['kan_şekeri_düzeyi'].max()}")
    
    with col2:
        # Korelasyon matrisi
        st.subheader("Korelasyon Matrisi")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Korelasyon Matrisi")
        st.pyplot(fig)

# Diyabet Tahmini Sayfası
elif app_mode == "Diyabet Tahmini":
    st.header("Diyabet Risk Tahmini")
    st.write("Lütfen aşağıdaki bilgileri girerek diyabet risk tahminini görüntüleyin.")
    
    # Model eğitimi
    @st.cache_resource
    def train_model():
        # Veri hazırlama
        X = df.drop('diyabet', axis=1)
        y = df['diyabet']
        
        # Eğitim ve test verilerini ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model oluşturma ve eğitme
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Model performansı
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    # Modeli eğit
    model, accuracy = train_model()
    st.write(f"Model Doğruluk Oranı: %{accuracy*100:.2f}")
    
    # Kullanıcı girişi için form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cinsiyet = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"])
            yas = st.number_input("Yaş", min_value=0, max_value=120, value=30)
            hipertansiyon = st.selectbox("Hipertansiyon", options=["Hayır", "Evet"])
            kalp_hastaligi = st.selectbox("Kalp Hastalığı", options=["Hayır", "Evet"])
            sigara_gecmisi = st.selectbox("Sigara Geçmişi", options=["Bilgi Yok", "Şu an İçiyor", "Eski İçici", "Hiç İçmemiş", "Diğer"])
        
        with col2:
            vucut_kitle_indeksi = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            hba1c_duzeyi = st.number_input("HbA1c Düzeyi", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
            kan_sekeri_duzeyi = st.number_input("Kan Şekeri Düzeyi", min_value=80, max_value=300, value=120)
        
        submit_button = st.form_submit_button(label="Tahmin Et")
    
    # Tahmin işlemi
    if submit_button:
        # Girişleri dönüştürme
        cinsiyet_value = 1 if cinsiyet == "Erkek" else 0
        hipertansiyon_value = 1 if hipertansiyon == "Evet" else 0
        kalp_hastaligi_value = 1 if kalp_hastaligi == "Evet" else 0
        
        # Sigara geçmişi dönüşümü
        sigara_mapping = {"Bilgi Yok": 0, "Şu an İçiyor": 1, "Eski İçici": 2, "Hiç İçmemiş": 4, "Diğer": 3}
        sigara_value = sigara_mapping[sigara_gecmisi]
        
        # Tahmin için veri hazırlama
        input_data = np.array([[cinsiyet_value, yas, hipertansiyon_value, kalp_hastaligi_value, 
                              sigara_value, vucut_kitle_indeksi, hba1c_duzeyi, kan_sekeri_duzeyi]])
        
        # Tahmin yapma
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] * 100
        
        # Sonuçları gösterme
        st.subheader("Tahmin Sonucu")
        if prediction == 1:
            st.error(f"Diyabet Riski: YÜKSEK (Olasılık: %{prediction_proba:.2f})")
            st.write("Bu sonuç bir tahmindir ve kesin teşhis için mutlaka bir doktora başvurmalısınız.")
        else:
            st.success(f"Diyabet Riski: DÜŞÜK (Olasılık: %{prediction_proba:.2f})")
            st.write("Bu sonuç bir tahmindir. Düzenli sağlık kontrolleri yaptırmayı unutmayın.")
        
        # Özellik önemliliği
        st.subheader("Özellik Önemliliği")
        feature_importance = pd.DataFrame({
            'Özellik': ['Cinsiyet', 'Yaş', 'Hipertansiyon', 'Kalp Hastalığı', 'Sigara Geçmişi', 
                      'Vücut Kitle İndeksi', 'HbA1c Düzeyi', 'Kan Şekeri Düzeyi'],
            'Önem': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Önem', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Önem', y='Özellik', data=feature_importance, ax=ax)
        ax.set_title("Özellik Önemliliği")
        st.pyplot(fig)
