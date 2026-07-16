# =====================================================
# SIMPLE CONTENT-BASED MOVIE RECOMMENDATION SYSTEM
# Uses Cosine Similarity to find movies similar to
# what the user already liked.
# =====================================================

import math

# ---- STEP 1: Our small movie "database" ----
# Each movie has a name and a genre feature vector.
# Genre order is fixed: [Action, Comedy, Drama, SciFi, Romance]
GENRES = ["Action", "Comedy", "Drama", "SciFi", "Romance"]

movies = {
    "Inception":      [1, 0, 0, 1, 0],
    "Interstellar":    [1, 0, 0, 1, 0],
    "The Notebook":    [0, 0, 1, 0, 1],
    "Titanic":         [0, 0, 1, 0, 1],
    "The Hangover":    [0, 1, 0, 0, 0],
    "Superbad":        [0, 1, 0, 0, 0],
    "Mad Max":         [1, 0, 0, 0, 0],
    "La La Land":      [0, 1, 0, 0, 1],
}


def dot_product(vec_a, vec_b):
    """Multiplies matching elements of two vectors and sums the results."""
    total = 0
    for a, b in zip(vec_a, vec_b):
        total += a * b
    return total


def magnitude(vec):
    """Calculates the length of a vector: sqrt(sum of squares)."""
    return math.sqrt(sum(x ** 2 for x in vec))


def cosine_similarity(vec_a, vec_b):
    """
    Returns a similarity score between 0 and 1.
    1 = identical taste direction, 0 = completely unrelated.
    """
    dot = dot_product(vec_a, vec_b)
    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)

    if mag_a == 0 or mag_b == 0:
        return 0   # avoid division by zero if a vector is all zeros

    return dot / (mag_a * mag_b)


def build_user_profile(liked_movies):
    """
    Builds a 'taste profile' vector by averaging the feature
    vectors of all movies the user liked.
    """
    num_genres = len(GENRES)
    profile = [0] * num_genres

    for movie in liked_movies:
        vector = movies[movie]
        for i in range(num_genres):
            profile[i] += vector[i]

    # Average out (divide by number of liked movies)
    profile = [value / len(liked_movies) for value in profile]
    return profile


def recommend(liked_movies, top_n=3):
    """
    Given a list of movies the user liked, recommends the
    top_n most similar UNWATCHED movies.
    """
    user_profile = build_user_profile(liked_movies)

    scores = []
    for title, vector in movies.items():
        if title in liked_movies:
            continue   # don't recommend movies they've already seen

        similarity = cosine_similarity(user_profile, vector)
        scores.append((title, similarity))

    # Sort by similarity score, highest first
    scores.sort(key=lambda item: item[1], reverse=True)

    return scores[:top_n]


def main():
    print("Available movies:", ", ".join(movies.keys()))
    print()

    # Example: user liked two sci-fi action movies
    liked = ["Inception", "Mad Max"]
    print(f"User liked: {liked}")

    recommendations = recommend(liked, top_n=3)

    print("\nTop Recommendations:")
    for title, score in recommendations:
        print(f"  {title}  (similarity: {score:.2f})")


if __name__ == "__main__":
    main()