# AI Internship Recommender

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A web application that recommends internships based on your skills, education, sector, and location using Machine Learning (TF-IDF + Cosine Similarity).**

---

## **Demo**
Check out the live application here:  
[🔗 Render Deployment](https://ai-intern-hunt.onrender.com)

---

## **Features**
- Skill-based internship recommendations.  
- Calculates match percentage using **cosine similarity**.  
- Filters by **Education, Sector, and Location**.  
- Responsive web interface for desktop and mobile.  
- Color-coded progress bars for match percentage.  

---

## **Dataset**
Internships are stored in a CSV file (`internships.csv`) with columns:  
- `title` – Internship title  
- `skills` – Required skills  
- `education` – Required education  
- `sector` – Sector of internship  
- `location` – Location  

> You can update or expand the dataset as needed.  

---

## **Technologies Used**
- **Backend:** Python, Flask  
- **Machine Learning:** scikit-learn (TF-IDF, Cosine Similarity)  
- **Frontend:** HTML, CSS  
- **Data Handling:** Pandas  
- **Deployment:** Render  

---

## **Installation**
1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/internship-recommender.git
cd internship-recommender
