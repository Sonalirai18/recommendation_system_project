from pathlib import Path

from recommender import MovieRecommender


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    recommender = MovieRecommender(
        ratings_path=base_dir / "data" / "ratings.csv",
        movies_path=base_dir / "data" / "movies.csv",
    )

    print("\nMovie Recommendation System")
    print("-" * 28)
    print("Available user IDs:", ", ".join(str(x) for x in sorted(recommender.user_item.index.tolist())))
    raw_user_id = input("Enter a user ID to generate recommendations: ").strip()

    try:
        user_id = int(raw_user_id)
    except ValueError:
        print("Invalid input. Please enter a numeric user ID.")
        return

    recommendations = recommender.get_top_n_recommendations(user_id=user_id, n=5)

    print(f"\nTop recommendations for user {user_id}:")
    for idx, rec in enumerate(recommendations, start=1):
        print(f"{idx}. {rec.title} | predicted score: {rec.predicted_score} | genres: {rec.genres}")


if __name__ == "__main__":
    main()
