import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Netflix di Google Play Store",
    page_icon="🎬",
    layout="wide"
)


# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("netflix_sentiment.csv")
# membersihkan nilai kosong
df['clean_review'] = df['clean_review'].fillna('').astype(str)
df['content'] = df['content'].fillna('').astype(str)

# ==========================
# STYLE
# ==========================
st.markdown("""
<style>

body{
    background-color:#0f0f0f;
}

.main-title{
    text-align:center;
    color:#E50914;
    font-size:45px;
    font-weight:800;
}

.sub-title{
    text-align:center;
    color:#777;
    font-size:20px;
}


.card{
    background-color:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
}


.step{
    background:#111;
    color:white;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
}


</style>
""", unsafe_allow_html=True)



# ==========================
# HEADER
# ==========================

st.markdown(
    '<div class="main-title">🎬 Analisis Sentimen Netflix</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Exploratory Data Visualization - Google Play Store Reviews</div>',
    unsafe_allow_html=True
)


st.divider()



# ==========================
# IDENTITAS
# ==========================

with st.container():

    col1,col2 = st.columns([1,3])

    with col1:
       st.image("foto_saya.jpg", width=220)

    with col2:

        st.markdown("""
        ### 👤 Identitas Pembuat

        **Nama:** Rambu Marshanda Tamu Ina Paa  
        
        **NIM:** 20254920004  
        
        **Mata Kuliah:** Exploratory Data Visualization

        **Objek Penelitian:** Ulasan Pengguna Aplikasi Netflix pada Google Play Store
        """)



st.divider()



# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🎬 Netflix Dashboard")

st.sidebar.write(
f"""
### Dataset

Jumlah data:
**{len(df)} ulasan**

Sumber:
Google Play Store
"""
)



# ==========================
# SENTIMENT COUNT
# ==========================

sentiment = df['sentiment'].value_counts()


positif = sentiment.get("Positif",0)
netral = sentiment.get("Netral",0)
negatif = sentiment.get("Negatif",0)



st.header("📌 Ringkasan Sentimen")


c1,c2,c3 = st.columns(3)


c1.metric(
"😊 Positif",
positif,
"+ Sentimen baik"
)

c2.metric(
"😐 Netral",
netral,
"Mayoritas"
)

c3.metric(
"😡 Negatif",
negatif,
"Keluhan"
)



st.divider()



# ==========================
# TAHAP ANALISIS
# ==========================

st.header("🔎 Tahapan Analisis")


steps = [

"1️⃣ Pengumpulan Data → Mengambil ulasan pengguna Netflix dari Google Play Store",

"2️⃣ Cleaning Data → Menghapus simbol, angka, tanda baca, dan karakter tidak penting",

"3️⃣ Case Folding → Mengubah seluruh teks menjadi huruf kecil",

"4️⃣ Tokenization → Memecah kalimat menjadi kata",

"5️⃣ Stopword Removal → Menghapus kata umum yang tidak memiliki makna",

"6️⃣ Stemming → Mengubah kata menjadi bentuk dasar",

"7️⃣ Sentiment Analysis → Mengelompokkan ulasan menjadi positif, netral, negatif"

]


for s in steps:

    st.markdown(
    f'<div class="step">{s}</div>',
    unsafe_allow_html=True
    )



st.divider()



# ==========================
# TAB
# ==========================

tab1,tab2,tab3,tab4 = st.tabs(
[
"📄 Data",
"☁️ WordCloud",
"📊 Visualisasi",
"💡 Kesimpulan"
]
)



# ==========================
# DATA
# ==========================

with tab1:


    st.subheader("Data Mentah")

    st.dataframe(
        df[['content']].head(15),
        use_container_width=True
    )


    st.subheader("Data Setelah Preprocessing")


    st.dataframe(
        df[['content','clean_review']].head(15),
        use_container_width=True
    )




# ==========================
# WORDCLOUD
# ==========================

with tab2:

    st.subheader("☁️ WordCloud Review Netflix")

    text = " ".join(
        str(x)
        for x in df["clean_review"]
        if pd.notna(x)
    )

    if text.strip():

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(wc)
        ax.axis("off")

        st.pyplot(fig)

    else:
        st.warning("Data clean_review kosong.")

    st.info("""
    Kata dominan menunjukkan pengguna banyak membahas:
    Netflix, film, aplikasi, menonton, dan layanan.
    """)

# ==========================
# VISUALISASI
# ==========================

with tab3:


    st.subheader("📊 Distribusi Sentimen")


    col1,col2 = st.columns(2)



    with col1:

        fig,ax = plt.subplots()

        ax.pie(
            sentiment.values,
            labels=sentiment.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
        "Persentase Sentimen"
        )

        st.pyplot(fig)




    with col2:


        fig,ax = plt.subplots()


        ax.bar(
        sentiment.index,
        sentiment.values
        )


        ax.set_title(
        "Jumlah Review"
        )

        ax.set_ylabel(
        "Jumlah"
        )

        st.pyplot(fig)



# ==========================
# KESIMPULAN
# ==========================


with tab4:


    st.subheader("💡 Kesimpulan Analisis")


    st.success(
f"""
Berdasarkan {len(df)} ulasan pengguna Netflix:

😊 Positif : {positif} ulasan  
😐 Netral : {netral} ulasan  
😡 Negatif : {negatif} ulasan


Mayoritas pengguna memberikan sentimen netral.
Hal ini menunjukkan bahwa pengguna lebih banyak
memberikan komentar berupa pengalaman penggunaan aplikasi.

Keluhan negatif umumnya berkaitan dengan login,
pembayaran, dan kendala teknis aplikasi.
"""
)



st.caption(
"© 2026 | Exploratory Data Visualization | Netflix Sentiment Analysis"
)