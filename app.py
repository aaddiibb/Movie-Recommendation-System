from pathlib import Path
import pickle
import json
import os
import re
from urllib.parse import quote_plus, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


BASE_DIR = Path(__file__).resolve().parent
KNOWN_GENRES = [
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "history",
    "horror",
    "music",
    "mystery",
    "romance",
    "science fiction",
    "tv movie",
    "thriller",
    "war",
    "western",
]

st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-top: #0e1117;
                --bg-bottom: #161b22;
                --panel: #111827;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --accent: #f59e0b;
                --accent-soft: #fbbf24;
            }

            .stApp {
                background:
                    radial-gradient(circle at 15% 20%, rgba(245, 158, 11, 0.16) 0%, rgba(245, 158, 11, 0) 35%),
                    radial-gradient(circle at 85% 0%, rgba(239, 68, 68, 0.14) 0%, rgba(239, 68, 68, 0) 30%),
                    linear-gradient(160deg, var(--bg-top), var(--bg-bottom));
                color: var(--text-main);
            }

            .stApp,
            .stMarkdown,
            .stMarkdown p,
            .stText,
            .stCaption,
            .st-emotion-cache-10trblm,
            .st-emotion-cache-16idsys {
                color: var(--text-main);
            }

            [data-testid="stWidgetLabel"] p {
                color: var(--text-main) !important;
                font-weight: 600;
            }

            [data-testid="stBaseButton-secondary"] {
                background-color: #1f2937;
                border: 1px solid rgba(251, 191, 36, 0.35);
                color: #f9fafb;
            }

            [data-testid="stBaseButton-secondary"]:hover {
                border-color: rgba(251, 191, 36, 0.65);
                color: #ffffff;
            }

            div[data-baseweb="select"] > div {
                background-color: #111827;
                border-color: rgba(156, 163, 175, 0.45);
                color: #f3f4f6;
            }

            div[data-baseweb="select"] input {
                color: #f3f4f6 !important;
            }

            [data-testid="stSlider"] [data-testid="stTickBarMin"],
            [data-testid="stSlider"] [data-testid="stTickBarMax"] {
                background: rgba(243, 244, 246, 0.25);
            }

            [data-testid="stSlider"] .st-c9,
            [data-testid="stSlider"] .st-ca {
                color: #e5e7eb;
            }

            .hero {
                padding: 0.75rem 0.95rem;
                border-radius: 14px;
                border: 1px solid rgba(251, 191, 36, 0.3);
                background: linear-gradient(120deg, rgba(17,24,39,0.85), rgba(31,41,55,0.8));
                margin-bottom: 0.45rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 1.55rem;
                letter-spacing: 0.3px;
                color: var(--text-main);
            }

            .hero p {
                margin: 0.2rem 0 0;
                color: var(--text-muted);
                font-size: 0.92rem;
            }

            .section-label {
                margin: 0.15rem 0 0.2rem;
                font-weight: 700;
                font-size: 0.98rem;
                color: var(--accent-soft);
            }

            .selected-caption {
                font-size: 0.82rem;
                color: var(--text-main);
                line-height: 1.15;
                margin-top: 0.2rem;
            }

            .selected-note {
                font-size: 0.72rem;
                color: var(--text-muted);
                line-height: 1.15;
                margin-top: 0.2rem;
            }

            .poster-caption {
                font-size: 0.86rem;
                color: var(--text-main);
                margin-top: 0.25rem;
                margin-bottom: 0.35rem;
                line-height: 1.2;
                min-height: 2.1rem;
            }

            .poster-meta {
                color: var(--text-muted);
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _poster_from_tmdb(title: str, api_key: str) -> str | None:
    params = urlencode(
        {
            "api_key": api_key,
            "query": title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }
    )
    url = f"https://api.themoviedb.org/3/search/movie?{params}"

    try:
        with urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return None

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        return None


def _normalize_title(title: str) -> str:
    lowered = title.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


@st.cache_data(show_spinner=False)
def _poster_from_imdb(title: str) -> str | None:
    query = quote_plus(title.strip().lower())
    first = query[0] if query else "a"
    url = f"https://v2.sg.media-imdb.com/suggestion/{first}/{query}.json"

    try:
        with urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("d", [])
        if not results:
            return None

        target = _normalize_title(title)
        best_item = None
        for item in results:
            candidate_title = _normalize_title(str(item.get("l", "")))
            if candidate_title == target and isinstance(item.get("i"), dict):
                best_item = item
                break

        if best_item is None:
            for item in results:
                if isinstance(item.get("i"), dict):
                    best_item = item
                    break

        if not best_item:
            return None

        image_block = best_item.get("i", {})
        image_url = image_block.get("imageUrl")
        if not image_url:
            return None
        return str(image_url)
    except Exception:
        return None


def _placeholder_poster(title: str) -> str:
    encoded = quote_plus(title[:36])
    return f"https://placehold.co/300x450/111827/F59E0B?text={encoded}"


def _get_poster_url(title: str) -> str:
    imdb_poster = _poster_from_imdb(title)
    if imdb_poster:
        return imdb_poster

    api_key = os.getenv("TMDB_API_KEY", "")

    if api_key:
        found = _poster_from_tmdb(title, str(api_key))
        if found:
            return found
    return _placeholder_poster(title)


def _load_movies_dataframe() -> pd.DataFrame:
    """Load movies from available artifacts in order of preference."""
    movie_dict_path = BASE_DIR / "movie_dict.pkl"
    movies_path = BASE_DIR / "movies.pkl"

    if movie_dict_path.exists():
        movies_dict = pickle.load(open(movie_dict_path, "rb"))
        return pd.DataFrame(movies_dict)

    if movies_path.exists():
        obj = pickle.load(open(movies_path, "rb"))
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame(obj)

    raise FileNotFoundError(
        "No usable movie artifact found. Expected movie_dict.pkl or movies.pkl"
    )


def _infer_genres(text: str) -> list[str]:
    normalized = f" {str(text).lower()} "
    genres = [genre for genre in KNOWN_GENRES if f" {genre} " in normalized]
    return genres


@st.cache_resource
def _load_resources() -> tuple[pd.DataFrame, object, object, object, object]:
    """Load data, build feature matrices, and validate any precomputed similarity."""
    movies_df = _load_movies_dataframe().reset_index(drop=True)

    similarity_path = BASE_DIR / "similarity.pkl"
    similarity = None
    content_matrix = None

    if similarity_path.exists():
        candidate = pickle.load(open(similarity_path, "rb"))
        expected_shape = (len(movies_df), len(movies_df))
        if getattr(candidate, "shape", None) == expected_shape:
            similarity = candidate

    text_column = None
    if "cleaned_text" in movies_df.columns:
        text_column = "cleaned_text"
    elif "combined" in movies_df.columns:
        text_column = "combined"

    if text_column is None:
        raise ValueError(
            "Dataset does not contain text features. Expected 'cleaned_text' or 'combined'."
        )

    content_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    content_matrix = content_vectorizer.fit_transform(movies_df[text_column].fillna(""))

    title_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    title_matrix = title_vectorizer.fit_transform(movies_df["title"].fillna("").str.lower())

    inferred_genres = movies_df[text_column].fillna("").apply(_infer_genres)
    genre_binarizer = MultiLabelBinarizer(classes=KNOWN_GENRES)
    genre_matrix = genre_binarizer.fit_transform(inferred_genres)

    return movies_df, similarity, content_matrix, title_matrix, genre_matrix


def recommend(
    movie: str,
    movies_df: pd.DataFrame,
    similarity,
    content_matrix,
    title_matrix,
    genre_matrix,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    matches = movies_df[movies_df["title"].str.lower() == movie.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]

    if similarity is not None:
        content_scores = similarity[idx]
    else:
        content_scores = cosine_similarity(content_matrix[idx], content_matrix).flatten()

    title_scores = cosine_similarity(title_matrix[idx], title_matrix).flatten()

    if np.any(genre_matrix[idx]):
        genre_scores = cosine_similarity(genre_matrix[idx].reshape(1, -1), genre_matrix).flatten()
    else:
        genre_scores = np.zeros(len(movies_df), dtype=float)

    # Genres get explicit weight so thriller/mystery style matches are prioritized.
    # Title similarity is only a small tiebreaker when there is already meaningful overlap.
    scores = (
        0.55 * content_scores
        + 0.35 * genre_scores
        + 0.10 * title_scores * ((content_scores > 0.08) | (genre_scores > 0))
    )

    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    recommendations = [
        (str(movies_df.iloc[i].title), float(score))
        for i, score in ranked
        if i != idx
    ][:top_k]
    return recommendations


_inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>CineMatch</h1>
        <p>Pick any film and get visually rich, content-based recommendations like a mini streaming shelf.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    movies, similarity_matrix, content_features, title_features, genre_features = _load_resources()
except Exception as exc:
    st.error(f"Failed to load recommendation data: {exc}")
    st.stop()

control_col, info_col = st.columns([4.2, 0.8])

with control_col:
    selected_movie_name = st.selectbox("Choose a movie", movies["title"].values)
    top_k = st.slider("Number of suggestions", min_value=3, max_value=10, value=6)
    recommend_clicked = st.button("🎥 Show Recommendations", width="stretch")

with info_col:
    st.markdown('<div class="section-label">Selected Movie</div>', unsafe_allow_html=True)
    st.image(_get_poster_url(selected_movie_name), width=95)
    st.markdown(
        f'<div class="selected-caption">{selected_movie_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="selected-note">IMDb first, TMDB fallback.</div>',
        unsafe_allow_html=True,
    )

if recommend_clicked:
    with st.spinner("Curating your movie shelf..."):
        results = recommend(
            selected_movie_name,
            movies,
            similarity_matrix,
            content_features,
            title_features,
            genre_features,
            top_k=top_k,
        )
    if not results:
        st.warning("No recommendations found for the selected movie.")
    else:
        st.markdown('<div class="section-label">You Might Also Like</div>', unsafe_allow_html=True)
        cols_per_row = 5
        for row_start in range(0, len(results), cols_per_row):
            row = results[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, (title, score) in enumerate(row):
                with cols[i]:
                    st.image(_get_poster_url(title), width=120)
                    st.markdown(f'<div class="poster-caption">{title}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="poster-meta">Similarity score: {score:.3f}</div>',
                        unsafe_allow_html=True,
                    )

