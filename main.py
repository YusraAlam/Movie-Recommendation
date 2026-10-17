
import os
import pickle
import ast
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# ENVIRONMENT
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE = "https://www.omdbapi.com/"


# FILE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None


# RESPONSE MODELS
class MovieCard(BaseModel):
    imdb_id: Optional[str] = None
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class MovieDetails(BaseModel):
    imdb_id: Optional[str] = None
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    movie: Optional[MovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[MovieCard]


# HELPERS
def norm_title(title: str) -> str:
    return str(title).strip().lower()


def parse_genres(value: Any) -> List[str]:
    """Read genres stored as lists, JSON strings, or Python-style strings."""
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()

        if not text or text.lower() in {"nan", "none", "null"}:
            return []

        try:
            items = json.loads(text)
        except (ValueError, TypeError):
            try:
                items = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [x.strip() for x in text.split(",") if x.strip()]

    if not isinstance(items, list):
        return []

    result = []

    for item in items:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append(str(name).strip())
        elif isinstance(item, str):
            if item.strip():
                result.append(item.strip())

    return result


def parse_rating(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() in {"", "N/A"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def require_api_key():
    if not OMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OMDB_API_KEY is missing. Add it to your .env file."
        )


async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    require_api_key()

    request_params = {
        **params,
        "apikey": OMDB_API_KEY
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(OMDB_BASE, params=request_params)
            response.raise_for_status()
            data = response.json()

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OMDb connection error: {type(exc).__name__}"
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OMDb HTTP error: {exc.response.status_code}"
        )

    if data.get("Response") == "False":
        raise HTTPException(
            status_code=404,
            detail=data.get("Error", "Movie not found")
        )

    return data


def movie_card(data: dict) -> MovieCard:
    return MovieCard(
        imdb_id=data.get("imdbID"),
        title=data.get("Title") or "",
        poster_url=(
            data.get("Poster")
            if data.get("Poster") not in (None, "", "N/A")
            else None
        ),
        release_date=data.get("Released") or data.get("Year"),
        vote_average=parse_rating(data.get("imdbRating"))
    )


async def omdb_search_movies(
    query: str,
    page: int = 1
) -> Dict[str, Any]:
    return await omdb_get({
        "s": query,
        "type": "movie",
        "page": page
    })


async def omdb_search_first(query: str) -> Optional[dict]:
    try:
        data = await omdb_search_movies(query)
        results = data.get("Search", [])
        return results[0] if results else None
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


async def omdb_movie_details(
    movie_id: Optional[str] = None,
    title: Optional[str] = None
) -> MovieDetails:
    params = {"plot": "full"}

    if movie_id:
        params["i"] = movie_id
    elif title:
        params["t"] = title
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide an IMDb ID or movie title."
        )

    data = await omdb_get(params)

    genres = [
        {"name": genre.strip()}
        for genre in data.get("Genre", "").split(",")
        if genre.strip() and genre.strip() != "N/A"
    ]

    return MovieDetails(
        imdb_id=data.get("imdbID"),
        title=data.get("Title") or "",
        overview=(
            data.get("Plot")
            if data.get("Plot") != "N/A"
            else None
        ),
        release_date=data.get("Released") or data.get("Year"),
        poster_url=(
            data.get("Poster")
            if data.get("Poster") not in (None, "", "N/A")
            else None
        ),
        genres=genres
    )


# TF-IDF HELPERS
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    if not hasattr(indices, "items"):
        raise RuntimeError(
            "indices.pkl must be a dictionary or pandas Series."
        )

    return {
        norm_title(key): int(value)
        for key, value in indices.items()
    }


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map is not initialized."
        )

    key = norm_title(title)

    if key not in TITLE_TO_IDX:
        raise HTTPException(
            status_code=404,
            detail=f"Movie not found in local dataset: {title}"
        )

    return TITLE_TO_IDX[key]


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources are not loaded."
        )

    idx = get_local_idx_by_title(query_title)

    if idx < 0 or idx >= tfidf_matrix.shape[0]:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF index is outside the matrix."
        )

    query_vector = tfidf_matrix[idx]
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    order = np.argsort(-scores)

    recommendations = []

    for i in order:
        i = int(i)

        if i == idx or i >= len(df):
            continue

        title = str(df.iloc[i]["title"])

        if not title or title.lower() == "nan":
            continue

        recommendations.append((title, float(scores[i])))

        if len(recommendations) >= top_n:
            break

    return recommendations


# LOCAL GENRE RECOMMENDATIONS
def local_genre_recommendations(
    selected_title: str,
    limit: int = 12
) -> List[MovieCard]:
    if df is None or "title" not in df.columns:
        return []

    selected_rows = df[
        df["title"].astype(str).str.lower()
        == selected_title.strip().lower()
    ]

    if selected_rows.empty or "genres" not in df.columns:
        return []

    selected_genres = set(
        parse_genres(selected_rows.iloc[0]["genres"])
    )

    if not selected_genres:
        return []

    recommendations = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()

        if not title or title.lower() == selected_title.strip().lower():
            continue

        genres = set(parse_genres(row.get("genres")))

        if not selected_genres.intersection(genres):
            continue

        recommendations.append(
            MovieCard(
                title=title,
                vote_average=parse_rating(row.get("vote_average"))
            )
        )

        if len(recommendations) >= limit:
            break

    return recommendations


# LOAD LOCAL PICKLES
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    required_files = [
        DF_PATH,
        INDICES_PATH,
        TFIDF_MATRIX_PATH,
        TFIDF_PATH
    ]

    missing = [path for path in required_files if not os.path.isfile(path)]

    if missing:
        raise RuntimeError(
            "Missing pickle files: " + ", ".join(missing)
        )

    with open(DF_PATH, "rb") as file:
        df = pickle.load(file)

    with open(INDICES_PATH, "rb") as file:
        indices_obj = pickle.load(file)

    with open(TFIDF_MATRIX_PATH, "rb") as file:
        tfidf_matrix = pickle.load(file)

    with open(TFIDF_PATH, "rb") as file:
        tfidf_obj = pickle.load(file)

    if not isinstance(df, pd.DataFrame) or "title" not in df.columns:
        raise RuntimeError(
            "df.pkl must contain a DataFrame with a title column."
        )

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if tfidf_matrix.shape[0] != len(df):
        raise RuntimeError(
            "The TF-IDF matrix row count must match df.pkl."
        )

    yield


app = FastAPI(
    title="OMDb Movie Recommender API",
    version="4.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ROUTES
@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "OMDb",
        "dataset_loaded": df is not None
    }


# HOME FEED
# OMDb does not provide TMDB-style popular/trending feeds.
# Use search terms to populate the home page.
@app.get("/home", response_model=List[MovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(10, ge=1, le=10)
):
    search_terms = {
        "popular": "Avengers",
        "top_rated": "The Godfather",
        "upcoming": "Dune",
        "now_playing": "Batman",
        "trending": "Spider-Man"
    }

    if category not in search_terms:
        raise HTTPException(
            status_code=400,
            detail="Invalid category."
        )

    data = await omdb_search_movies(search_terms[category])

    return [
        movie_card(item)
        for item in data.get("Search", [])[:limit]
    ]


# MOVIE SEARCH
@app.get("/tmdb/search")
async def movie_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10)
):
    data = await omdb_search_movies(query, page)

    return {
        "results": [
            {
                "imdbID": item.get("imdbID"),
                "id": item.get("imdbID"),
                "title": item.get("Title"),
                "original_title": item.get("Title"),
                "poster_path": (
                    item.get("Poster")
                    if item.get("Poster") != "N/A"
                    else None
                ),
                "poster_url": (
                    item.get("Poster")
                    if item.get("Poster") != "N/A"
                    else None
                ),
                "release_date": item.get("Year")
            }
            for item in data.get("Search", [])
        ],
        "total_results": int(data.get("totalResults", 0)),
        "page": page
    }


# MOVIE DETAILS BY IMDb ID
@app.get("/movie/id/{movie_id}", response_model=MovieDetails)
async def movie_details_route(movie_id: str):
    return await omdb_movie_details(movie_id=movie_id)


# LOCAL GENRE RECOMMENDATIONS
@app.get("/recommend/genre", response_model=List[MovieCard])
async def recommend_genre(
    title: str = Query(..., min_length=1),
    limit: int = Query(12, ge=1, le=30)
):
    return local_genre_recommendations(title, limit)


# TF-IDF RECOMMENDATIONS
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50)
):
    recommendations = tfidf_recommend_titles(title, top_n)

    return [
        {"title": title, "score": score}
        for title, score in recommendations
    ]


# DETAILS + TF-IDF + GENRE RECOMMENDATIONS
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30)
):
    search_result = await omdb_search_first(query)

    if not search_result:
        raise HTTPException(
            status_code=404,
            detail=f"No movie found for query: {query}"
        )

    imdb_id = search_result.get("imdbID")

    if imdb_id:
        details = await omdb_movie_details(movie_id=imdb_id)
    else:
        details = await omdb_movie_details(title=query)

    # Recommendations from the local TF-IDF model
    tfidf_items = []

    try:
        recs = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n
        )
    except HTTPException:
        recs = []

    # OMDb search supplies posters for recommended titles.
    # A failed poster lookup does not stop recommendations.
    for title, score in recs:
        try:
            result = await omdb_search_first(title)
            card = movie_card(result) if result else None
        except HTTPException:
            card = None

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                movie=card
            )
        )

    genre_recs = local_genre_recommendations(
        details.title,
        limit=genre_limit
    )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs
    )
