import pickle
import streamlit as st
from surprise import SVD

# Load data from the file
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Check the columns of the movies DataFrame
st.write("Columns in movies DataFrame:", movies.columns.tolist())

def recommend_movies(user_id):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

    if len(unrated_movies) == 0:
        return []

    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]

    return top_recommendations

# Streamlit app
st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1)

if st.button("Recommend Movies"):
    if user_id:
        recommendations = recommend_movies(user_id)
        if recommendations:
            st.write(f"Top 10 movie recommendations for User {user_id}:")
            for recommendation in recommendations:
                movie_data = movies[movies['movieId'] == recommendation.iid]

                if not movie_data.empty:
                    movie_data = movie_data.iloc[0]
                    movie_title = movie_data['title']
                    
                    # Check if the poster and description columns exist
                    movie_poster = movie_data['poster'] if 'poster' in movie_data else "No Poster Available"
                    movie_description = movie_data['description'] if 'description' in movie_data else "No Description Available"

                    # Display movie details
                    st.image(movie_poster, caption=movie_title, width=200)
                    st.write(f"**Title:** {movie_title}")
                    st.write(f"**Estimated Rating:** {recommendation.est:.2f}")
                    st.write(f"**Description:** {movie_description}")
                    st.write("---")
                else:
                    st.write("Movie data not found.")
        else:
            st.write("No unrated movies available for this user.")
    else:
        st.write("Please enter a valid User ID.")
