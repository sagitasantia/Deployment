import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import pickle
import altair as alt

# Set judul di sidebar
st.sidebar.title("Daftar isi Language of New York City")
# Icon untuk setiap opsi di sidebar
icon_dict = {
    "Top 10 Bahasa": "💻 Top 10 bahasa Umum",
    "Bahasa Langka": "🔍 Top 10 bahasa Langka",
    "Distribusi Jumlah Penutur Global": "🌐 Distribusi Jumlah Global Speakers",
    "Hubungan Antara Jumlah Penutur Global dan Ukuran Bahasa": "📈 Hubungan Antara Jumlah Global_Speakers dan Size",
    "Classifier": "🤖 Clustering"
}

# Tampilkan daftar isi di sidebar
st.sidebar.markdown("### Daftar isi")
selected_option = st.sidebar.radio(
    "",
    ["Top 10 Bahasa", "Bahasa Langka", "Distribusi Jumlah Penutur Global", "Hubungan Antara Jumlah Penutur Global dan Ukuran Bahasa", "Classifier"],
    format_func=icon_dict.get
)

st.header("DEPLOYMENT")
st.title("TOP 10 LANGUAGE Of NEW YORK CITY and Classifier")

url = 'https://raw.githubusercontent.com/sagitasantia/Checkpoint/main/Data%20Language%20(1).csv'
df = pd.read_csv(url, index_col=[0])

if selected_option == "Top 10 Bahasa":
    st.write(df)
    st.subheader('Top 10 bahasa yang paling sering digunakan')
    top_10_languages = df['Language'].value_counts().head(10)
    top_10_languages_df = pd.DataFrame({
        'Language': top_10_languages.index,
        'Count': top_10_languages.values
    })
    st.bar_chart(top_10_languages_df.set_index('Language'))

elif selected_option == "Bahasa Langka":
    st.write(df)
    st.subheader('10 bahasa yang jarang digunakan atau yang langka')
    bottom_10_languages = df['Language'].value_counts().tail(10)
    bottom_10_languages_df = pd.DataFrame({
        'Language': bottom_10_languages.index,
        'Count': bottom_10_languages.values
    })
    st.bar_chart(bottom_10_languages_df.set_index('Language'))

elif selected_option == "Distribusi Jumlah Penutur Global":
    st.subheader('Distribusi Jumlah Penutur Global')
    num_bins = st.slider('Number of Bins', min_value=5, max_value=50, value=20)
    histogram = alt.Chart(df).mark_bar(color='blue', opacity=0.7, stroke='black').encode(
        alt.X('Global Speakers', bin=alt.Bin(maxbins=num_bins), title='Global Speakers'),
        alt.Y('count()', title='Frequency')
    ).properties(
        width=600,
        height=400,
        title='Distribution of Global Speakers'
    )
    st.write(histogram)

elif selected_option == "Hubungan Antara Jumlah Penutur Global dan Ukuran Bahasa":
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

elif selected_option == "Classifier":
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
    df1['Cluster'] = kmeans.labels_
    st.write(df1)
    global_speakers_input = st.number_input('Input Global Speakers')
    size_input = st.number_input('Input Size')
    status_input = st.number_input('Input Status')
    global_speakers_category_input = st.number_input('Input Global Speakers Category')
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
