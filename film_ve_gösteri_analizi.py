import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Safely parse lists from strings
def safe_eval_list(value):
    try:
        return eval(value) if isinstance(value, str) else value
    except:
        return []

# Load the data
df1 = pd.read_csv('final_data.csv')
df1['genres'] = df1['genres'].apply(safe_eval_list)

df2 = pd.read_csv("2.sistem1.csv")
df2['genres'] = df2['genres'].apply(safe_eval_list)
df2['actors'] = df2['actors'].apply(safe_eval_list)



st.markdown(
    """
    <style>
    /* Ana arka plan ve başlık ayarları */
    [data-testid="stAppViewContainer"] {
        background-color: #000000; /* Ana arka plan siyah */
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0); /* Başlık çubuğu siyah */
    }
    [data-testid="stToolbar"] {
        right: 3rem;
        color: #FF0000 !important; /* Deploy yazısı kırmızı */
    }
    
    /* Kenar çubuğu ayarları */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #000000; /* Kenar çubuğu arka plan siyah */
        color: #FF0000; /* Kenar çubuğu yazı rengi kırmızı */
    }
    [data-testid="stSidebarNav"] {
        color: #FF0000 !important; /* Kenar çubuğu yazı rengi kırmızı */
    }
    [data-testid="stSidebarNav"] * {
        color: #FF0000 !important; /* Kenar çubuğundaki tüm yazılar kırmızı */
    }
    [data-testid="stSidebarNav"] a {
        color: #FF0000 !important; /* Kenar çubuğundaki bağlantılar kırmızı */
    }
    [data-testid="stSidebarNav"] svg {
        fill: #FF0000 !important; /* Kenar çubuğundaki simgeler kırmızı */
    }
    [data-testid="stSidebar"] button[aria-expanded="true"] > div:first-child,
    [data-testid="stSidebar"] button[aria-expanded="false"] > div:first-child {
        color: #FF0000 !important; /* Kenar çubuğu ok kırmızı */
    }

    /* Üç nokta menü */
    [data-testid="collapsedControl"] {
        color: #FF0000 !important; /* Üç nokta menü kırmızı */
    }

    /* Diğer stil ayarları */
    .css-1vencpc {
        color: #FF0000 !important; /* "Deploy" yazısını kırmızı yap */
    }
    .stRadio > div {
        color: #FF0000 !important; /* Kenar çubuğundaki radyo düğmesi yazıları kırmızı */
    }
    .stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #FF0000 !important; /* Radyo düğmesindeki yazılar kırmızı */
    }
    .stRadio div[role="radiogroup"] input[type="radio"]:checked + div > div {
        background-color: white !important; /* Seçili radyo düğmesinin noktası kırmızı */
    }
    .custom-text {
        font-size: 24px; /* Yazı boyutu */
        color: red;      /* Yazı rengi */
    }
    h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader, .stCaption, .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: red !important; /* Başlık ve etiket renkleri */
    }
    .stSlider > div > div > div {
        background-color: #FFFFFF;  /* Slider beyaz */
    }
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #FF0000;  /* Slider düğmesi kırmızı */
    }
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        color: #FFFFFF !important; /* Giriş alanı yazıları beyaz */
    }
    .stMarkdown p, .stTable th, .stTable td {
        color: red !important; /* Diğer yazı renkleri kırmızı */
    }

    /* Tablo ayarları */
    table {
        width: 100%;
        color: #212121;
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
    }
    th, td {
        padding: 10px;
        border: 1px solid #E0E0E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Home Function (Introduction)
def home():
    st.write("""
    <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); min-width: 100%; min-height: 100%;">
        <img src="https://media.tenor.com/NerN41mjgV0AAAAC/netflix-intro.gif" alt="Animated GIF" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    """, unsafe_allow_html=True)

# Recommendation System 1 Function
def tavsiye_sistemi_1():
    st.title('Tavsiye Sistemi 1: Film ve Gösteri Önerileri')
    

    # Dropdown options
    type_options = ['All'] + sorted(df1['type'].dropna().unique().tolist())
    genre_options = ['All'] + sorted(set([genre for sublist in df1['genres'].dropna().tolist() for genre in sublist]))
    year_options = ['All'] + sorted(df1['release_year'].dropna().unique().tolist())
    director_options = ['All'] + sorted(df1['directors'].dropna().unique().tolist())

    # User interface
    content_type = st.selectbox("Tür (Film/Gösteri):", type_options)
    genre = st.selectbox("Kategori:", genre_options)
    year = st.selectbox("Yayın Yılı:", year_options)
    director = st.selectbox("Yönetmen:", director_options)
    score = st.slider("IMDb Puanı:", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    # Recommendation Function
    def get_recommendations(content_type, genre, year, director, score):
        filtered_df = df1[
            ((df1['type'] == content_type) if content_type != 'All' else True) &
            (df1['genres'].apply(lambda genres: genre in genres if isinstance(genres, list) else False) if genre != 'All' else True) &
            ((df1['release_year'] == year) if year != 'All' else True) &
            ((df1['directors'] == director) if director != 'All' else True) &
            (df1['imdb_score'] >= score)
        ]
        return ["Uygun film veya show bulunamadı."] if filtered_df.empty else filtered_df[['title', 'description', 'genres', 'imdb_score']]

    # Show Recommendations
    recommendations = get_recommendations(content_type, genre, year, director, score)
    st.write("Seçilen kriterlere göre öneriler:")
    st.write(recommendations)

# Recommendation System 2 Function
def tavsiye_sistemi_2():
    st.title('Tavsiye Sistemi 2: Film ve Gösteri Önerileri')
    # Function to safely join list or handle missing lists
    def safe_join(value):
        return ' '.join(value) if isinstance(value, list) else ''

    # Add overview column
    df2['overview'] = (
        df2["title"].astype(str)
        + " "
        + df2["description"].astype(str)
        + " "
        + df2["genres"].apply(lambda x: " ".join(x) if isinstance(x, list) else '')
        + " "
        + df2["director"].astype(str)
        + " "
        + df2["actors"].apply(lambda x: " ".join(x) if isinstance(x, list) else '')
        + " "
        + df2["production_countries"].astype(str)
    ).str.lower() \
     .str.replace("\n", " ") \
     .str.replace("-", "") \
     .str.translate(str.maketrans("", "", string.punctuation))

    # Count Vectorizer
    count = CountVectorizer(stop_words='english', ngram_range=(1, 5))
    count_matrix = count.fit_transform(df2['overview'])

    # Cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Index series
    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

    # Recommendation function
    def get_recommendations(title, cosine_sim=cosine_sim, top_k=5):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores if i[0] != idx]

        return (
            df2.iloc[movie_indices]
            .sort_values(["imdb_votes", "imdb_score"], ascending=False)
            [["title", "description", "genres", "imdb_score"]]
            .reset_index(drop=True)
            .head(top_k)
        )

    # Streamlit UI
    st.write("Bir film/gösteri seçin ve benzer öneriler alın:")

    # Dropdown for film/show selection
    film_selected = st.selectbox("Bir Film/Gösteri Seç", sorted(df2["title"].unique()))

    # Recommendation count slider
    recommendation_count = st.slider("Öneri Sayısı", min_value=1, max_value=10, value=5)

    # Recommendation button
    if st.button("Önerileri Al"):
        recommendations = get_recommendations(film_selected, top_k=recommendation_count)
        st.write(f"{film_selected} için tavsiye edilen filmler:")
        st.table(recommendations)
        


# Main Function to Switch Pages
def ana_sayfa():
    st.sidebar.title("Tavsiye Sistemleri")

    # Create buttons as clickable images
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = 'Home'
    if st.sidebar.button("🎬 Tavsiye Sistemi 1"):
        st.session_state.page = 'Tavsiye Sistemi 1'
    if st.sidebar.button("🍿 Tavsiye Sistemi 2"):
        st.session_state.page = 'Tavsiye Sistemi 2'

    # Default page to 'Home' if not set
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # Render the page based on selection
    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Tavsiye Sistemi 1':
        tavsiye_sistemi_1()
    elif st.session_state.page == 'Tavsiye Sistemi 2':
        tavsiye_sistemi_2()

if __name__ == "__main__":
    ana_sayfa()
