import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")


# Function to extract genres from JSON string
def extract_genres(genre_str):
    genres = ast.literal_eval(genre_str)
    return ", ".join([g['name'] for g in genres])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres']].dropna()
    df['genres'] = df['genres'].apply(extract_genres)
    return df

# Recommend movies based on content similarity
def recommend_movie(title, num=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# Load and process data
df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Streamlit UI
st.title("🎥 Movie Recommendation System")
st.markdown("Get movie suggestions based on your favorite!")

user_input = st.text_input("Enter a Movie Name (e.g. Avatar)")

if user_input:
    if user_input in df['title'].values:
        results = recommend_movie(user_input)
        st.subheader("🎯 Top Recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.markdown(f"**Genres:** {row['genres']}")
            st.markdown("---")
    else:
        st.warning("Movie not found! Please check the spelling.")
