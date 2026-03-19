# Recommendation System Project

This project demonstrates a simple recommendation system built with Python using user-based collaborative filtering and cosine similarity. It was created for a University of the Cumberlands assignment on intelligent systems and recommender systems.

## Project structure

- `app.py` - command-line interface
- `recommender.py` - recommendation logic
- `data/movies.csv` - movie metadata
- `data/ratings.csv` - sample user ratings
- `requirements.txt` - Python dependencies

## How the system works

1. Loads user ratings and movie metadata.
2. Creates a user-item matrix with users as rows and movies as columns.
3. Fills missing values with zero for similarity computation.
4. Calculates user-to-user cosine similarity.
5. Predicts ratings for unseen movies using weighted neighbor ratings.
6. Returns the top 5 movie recommendations.
7. Uses a popularity-based fallback for a cold-start or unknown user.

## How to run

```bash
python -m venv .venv
```

### Windows
```bash
.venv\\Scripts\\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

## What was modified from the base idea

This project was inspired by standard collaborative filtering examples commonly taught in recommender systems tutorials and machine learning coursework. The solution was customized by:

- organizing the logic into reusable Python modules,
- adding a command-line interface,
- excluding items that the target user already rated,
- returning top-N ranked recommendations,
- adding a popularity-based fallback for cold-start cases, and
- simplifying the dataset so the recommendation flow is easy to understand and demonstrate.

## GitHub repository

Replace this placeholder before submission:

`https://github.com/your-username/recommendation-system-project`

## Academic note

The written report should cite the MovieLens dataset, recommender systems literature, and any tutorial, documentation, or package references used during development.
