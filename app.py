import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("imdb_top_1000.csv")

movies = movies[['Series_Title','Genre','Overview','Director','Star1','Star2','Star3']]

movies.fillna('', inplace=True)

# Combine features
def combine_features(row):
    return row['Genre'] + " " + row['Overview'] + " " + row['Director'] + " " + row['Star1'] + " " + row['Star2'] + " " + row['Star3']

movies["combined_features"] = movies.apply(combine_features, axis=1)

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(movies["combined_features"])

# Similarity
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_title):

    movie_index = movies[movies['Series_Title'] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))

    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for movie in sorted_movies[1:6]:
        recommendations.append(movies.iloc[movie[0]]['Series_Title'])

    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = movies['Series_Title'].values

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    results = recommend(selected_movie)

    st.subheader("Recommended Movies:")

    for movie in results:

        st.write(movie)

