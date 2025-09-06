from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("internships.csv")

# Get unique options for dropdowns
educations = sorted(df['education'].unique())
sectors = sorted(df['sector'].unique())
locations = sorted(df['location'].unique())

def recommend_internships(skills, education, sector, location, top_n=5):
    # Combine internship details into one text column
    df["combined"] = df["skills"] + " " + df["education"] + " " + df["sector"] + " " + df["location"]

    # Candidate profile as a text string
    candidate_profile = skills + " " + education + " " + sector + " " + location

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"].tolist() + [candidate_profile])

    # Cosine similarity between candidate and all internships
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df["score"] = cosine_sim.flatten()

    # Sort internships by similarity score
    recommendations = df.sort_values(by="score", ascending=False).head(top_n)
    recommendations = recommendations[["title", "skills", "education", "sector", "location", "score"]]
    recommendations["match"] = (recommendations["score"] * 100).round(2)
    return recommendations.to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        skills = request.form.get("skills", "")
        education = request.form.get("education", "")
        sector = request.form.get("sector", "")
        location = request.form.get("location", "")
        recommendations = recommend_internships(skills, education, sector, location, top_n=5)
    return render_template("index.html", recommendations=recommendations,
                           educations=educations, sectors=sectors, locations=locations)

if __name__ == "__main__":
    app.run(debug=True)
