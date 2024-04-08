import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import pickle
import altair as alt

st.sidebar.title("Language of New York City")

with st.sidebar:
    selected_option = option_menu(
        "Pilih opsi:",
        ["💻 Top 10 Bahasa Umum", "🔍 Top 10 Bahasa Langka", "🌐 Distribusi Jumlah Global Speakers", "📈 Hubungan Antara Jumlah Global_Speakers dan Size", "🤖 Clustering"]
    )

st.header("DEPLOYMENT")
st.title("TOP 10 LANGUAGE Of NEW YORK CITY and Classifier")

url = 'https://raw.githubusercontent.com/sagitasantia/Checkpoint/main/Data%20Language%20(1).csv'
df = pd.read_csv(url, index_col=[0])

if selected_option == "💻 Top 10 Bahasa Umum":
    st.write(df)
    st.subheader('Top 10 bahasa yang paling sering digunakan')
    top_10_languages = df['Language'].value_counts().head(10)
    top_10_languages_df = pd.DataFrame({
        'Language': top_10_languages.index,
        'Count': top_10_languages.values
    })
    st.bar_chart(top_10_languages_df.set_index('Language'))
    st.title("Kesimpulan")
    st.write("Bahasa yang paling banyak digunakan dari sepuluh yang disajikan adalah Language Lenape (Munsee) dengan jumlah penggunaan sebanyak 8 kali.")


elif selected_option == "🔍 Top 10 Bahasa Langka":
    st.write(df)
    st.subheader('10 bahasa yang jarang digunakan atau yang langka')
    bottom_10_languages = df['Language'].value_counts().tail(10)
    bottom_10_languages_df = pd.DataFrame({
        'Language': bottom_10_languages.index,
        'Count': bottom_10_languages.values
    })
    st.bar_chart(bottom_10_languages_df.set_index('Language'))
    st.title("Kesimpulan")
    st.write("Semua bahasa-bahasa ini dikenal sebagai bahasa yang memiliki jumlah penutur yang sangat sedikit, dan beberapa di antaranya kemungkinan sudah punah atau sangat terancam punah.")

elif selected_option == "🌐 Distribusi Jumlah Global Speakers":
    st.subheader('Distribusi Jumlah Penutur Global')
    num_bins = st.number_input('Number of Bins', min_value=5, max_value=50, value=20)
    histogram = alt.Chart(df).mark_bar(color='blue', opacity=0.7, stroke='black').encode(
        alt.X('Global Speakers', bin=alt.Bin(maxbins=num_bins), title='Global Speakers'),
        alt.Y('count()', title='Frequency')
    ).properties(
        width=600,
        height=400,
        title='Distribution of Global Speakers'
    )
    st.write(histogram)
    st.title("Kesimpulan")
    st.write("Grup dengan 0 - 1.000.000 pembicara global menunjukkan frekuensi tertinggi, yang berarti ada lebih banyak kelompok bahasa dengan jumlah pembicara di rentang ini dibandingkan dengan rentang jumlah pembicara yang lain.")


elif selected_option == "📈 Hubungan Antara Jumlah Global_Speakers dan Size":
    st.subheader('Hubungan antara Jumlah Penutur Global dan Ukuran Bahasa')
    scatter_plot_alt = alt.Chart(df).mark_circle(size=60).encode(
        x='Global Speakers',
        y='Size',
        tooltip=['Language', 'Global Speakers', 'Size']
    ).properties(
        width=800,
        height=500
    ).interactive()
    st.altair_chart(scatter_plot_alt, use_container_width=True)
    st.title("Kesimpulan")
    st.write("Bahasa dengan jumlah penutur global yang lebih rendah cenderung dikategorikan dalam grup 'Small' dan 'Smallest'. Ini terlihat dari penyebaran titik-titik yang banyak berada di bagian kiri chart. Seiring meningkatnya jumlah penutur global, kategori ukuran bahasa juga cenderung meningkat, dengan beberapa bahasa masuk dalam kategori 'Medium' dan 'Large'. Terdapat beberapa bahasa dengan jumlah penutur yang sangat besar yang masuk dalam kategori 'Largest'. Data menunjukkan variasi yang lebar dalam setiap kategori ukuran, yang mengindikasikan bahwa tidak semua bahasa dengan jumlah penutur serupa memiliki ukuran yang sama.")

elif selected_option == "🤖 Clustering":
    st.subheader('Making Classifier')
    st.title('')
    file_path = 'kmeans_model.pkl'
    with open(file_path, 'rb') as f:
        clf = joblib.load(f)
    url = 'https://raw.githubusercontent.com/sagitasantia/Checkpoint/main/Data%20Cleanedd.csv'
    df1 = pd.read_csv(url, index_col=[0])
    st.write(df1)
    df_float = df1.select_dtypes(include=['float64'])
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_float)
    df1['Cluster'] = kmeans.labels_
    scatter_plot = alt.Chart(df1).mark_circle().encode(
        x='Global Speakers',
        y='Size',
        color='Cluster:N',
        tooltip=['Global Speakers', 'Size', 'Cluster']
    ).properties(
        width=800,
        height=500
    ).interactive()
    st.altair_chart(scatter_plot, use_container_width=True)
    st.write(df1)
    st.title("Masukkan data untuk prediksi")
    global_speakers_input = st.number_input('Input Global Speakers', min_value=0.0, value=5000.0)
    size_input = st.number_input('Input Size', min_value=0.0, value=500.0)
    status_input = st.number_input('Input Status', min_value=0.0, value=500.0)
    global_speakers_category_input = st.number_input('Input Global Speakers Category', min_value=0.0, value=500.0)
    submit_button = st.button('Submit')
    if submit_button:
        model = pickle.load(open("kmeans_model.pkl", "rb"))
        data = pd.DataFrame({
            'Global Speakers': [global_speakers_input],
            'Size': [size_input],
            'Status': [status_input],
            'Global Speakers Category': [global_speakers_category_input]
        })
        prediksi = model.predict(data)
        kategori = {0: "Langka", 1: "Langka", 2: "Umum", 3: "Umum", 4: "Umum", 5: "Umum", 6: "Umum"}
        st.subheader('Hasil Prediksi:')
        kategori_bahasa = kategori.get(prediksi[0], "Tidak Dikenal")
        st.write(f"Prediksi Kluster: {prediksi[0]}, Kategori: {kategori_bahasa}")
