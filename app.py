import streamlit as st
import pickle
import pandas as pd
import requests
similarity=pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    idx = movies[movies['title'].str.lower() == movie.lower()].index[0]
    distance = similarity[idx]
    movies_list=sorted(list(enumerate(distance)), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies=[]
    for j in movies_list:
        movie_id=j[0]
        recommended_movies.append(movies.iloc[j[0]].title)
    return recommended_movies


movies_dict=pickle.load(open('movie_dict.pkl', 'rb'))
movies= pd.DataFrame(movies_dict)
st.title('Movie Recommendation System')
selected_movie_name = st.selectbox("Enter a movie name ",movies['title'].values)
if st.button('Recommend'):
   re= recommend(selected_movie_name)
   for i in re:
       st.write(i)

