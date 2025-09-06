from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("internships.csv")

def preprocess(text):
    return str(text).lower().strip()

def recommend_internships(skills, education, sector, location, top_n=5):
    # Preprocess inputs
    skills = preprocess(skills)
    education = preprocess(education)
    sector = preprocess(sector)
    location = preprocess(location)

    # Weighted combination for dataset
    df["combined"] = (
        (df["skills"].apply(preprocess) + " ") * 3 +
        (df["education"].apply(preprocess) + " ") * 2 +
        (df["sector"].apply(preprocess) + " ") * 2 +
        (df["location"].apply(preprocess) + " ")
    )

    # Candidate profile
    candidate_profile = (
        (skills + " ") * 3 +
        (education + " ") * 2 +
        (sector + " ") * 2 +
        (location + " ")
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"].tolist() + [candidate_profile])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df["score"] = cosine_sim.flatten()

    # -------- Location filtering + fallback --------
    if location:
        location_matches = df[df["location"].str.lower() == location]
        if not location_matches.empty:
            recommendations = location_matches.sort_values(by="score", ascending=False).head(top_n)
        else:
            recommendations = df.sort_values(by="score", ascending=False).head(top_n)
    else:
        recommendations = df.sort_values(by="score", ascending=False).head(top_n)
    # ------------------------------------------------

    # Convert score to percentage
    recommendations["score"] = (recommendations["score"] * 100).round(0).astype(int)

    return recommendations[["title", "skills", "education", "sector", "location", "score"]].to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        skills = request.form.get("skills", "")
        education = request.form.get("education", "")
        sector = request.form.get("sector", "")
        location = request.form.get("location", "")

        recommendations = recommend_internships(skills, education, sector, location, top_n=3)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
