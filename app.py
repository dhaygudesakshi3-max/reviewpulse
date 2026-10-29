import os
import re
import sqlite3
from datetime import datetime

import numpy as np
from flask import Flask, request, render_template_string
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
DB_NAME = os.environ.get("DB_PATH", "reviewpulse.db")

# Raw E-Commerce Dataset (Includes missing values None/NaN and noisy HTML/symbols for Preprocessing Demo)
RAW_DATASET = [
    ("Battery drains fast and phone overheating badly during charging!! <br>", 15999, 1, 25),
    ("Screen flickering defect and battery drains fast within two hours", 18500, 1, 30),
    ("Phone overheating and battery backup is terrible, defective piece", None, 1, 18),
    ("Battery drains fast and charger overheating issue reported", 16200, 1, None),
    ("Late delivery and damaged box packaging, broken item on arrival", 14000, 1, 15),
    ("Worst courier service, late delivery and torn packaging box", 1200, 2, 12),
    ("Damaged packaging and late delivery by courier partner", None, 2, 9),
    ("Delivery was very late and outer box packaging was crushed", 1150, 2, 14),
    ("Average camera quality, decent battery backup for the price", 12999, 3, 5),
    ("Product is okay, delivery was normal and build quality is average", 999, 3, None),
    ("Decent performance for basic usage, average display quality", 11500, 3, 4),
    ("Okay product for the price, average sound and normal delivery", 1499, 3, 6),
    ("Excellent camera quality, fast charging and smooth display performance!", 24999, 5, 42),
    ("Great value for money, fast delivery and sturdy build quality", 19999, 5, 38),
    ("Superb sound quality, long battery life and fast charging support", 2499, 4, 29),
    ("Amazing display quality, smooth performance and excellent camera", None, 5, 45),
    ("Fast charging works great, excellent build quality and value", 21000, 5, 33),
    ("Battery drains fast, severe overheating and defective screen", 17500, 1, 27),
]

STOPWORDS = {"and", "the", "is", "in", "it", "for", "on", "was", "by", "to", "within", "two", "very"}
ASPECT_KEYWORDS = ["battery", "overheating", "delivery", "packaging", "camera", "charging", "quality", "fast"]


def clean_text(text):
    if not text:
        return "neutral review"
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens) if tokens else "neutral review"


def map_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                review_text TEXT NOT NULL,
                price REAL NOT NULL,
                rating INTEGER NOT NULL,
                sentiment TEXT NOT NULL,
                cluster_name TEXT NOT NULL,
                matched_rule TEXT NOT NULL,
                action TEXT NOT NULL
            )
        """)


def build_mining_pipeline():
    # 1. DATA PREPROCESSING: Missing Value Imputation
    valid_prices = [r[1] for r in RAW_DATASET if r[1] is not None]
    valid_votes = [r[3] for r in RAW_DATASET if r[3] is not None]
    median_price = float(np.median(valid_prices))
    median_votes = float(np.median(valid_votes))
    missing_imputed = sum(1 for r in RAW_DATASET if r[1] is None or r[3] is None)

    cleaned_texts, prices, ratings, votes, sentiments = [], [], [], [], []
    for text, price, rating, vote in RAW_DATASET:
        cleaned_texts.append(clean_text(text))
        prices.append(median_price if price is None else float(price))
        ratings.append(int(rating))
        votes.append(median_votes if vote is None else float(vote))
        sentiments.append(map_sentiment(int(rating)))

    # 2. DATA PREPROCESSING: Min-Max Normalization & TF-IDF Transformation
    scaler = MinMaxScaler()
    num_Normalized = scaler.fit_transform(np.column_stack([prices, votes]))

    tfidf = TfidfVectorizer(max_features=35)
    X_tfidf = tfidf.fit_transform(cleaned_texts).toarray()
    X_combined = np.hstack([X_tfidf, num_Normalized])

    # 3. CLUSTERING: K-Means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_tfidf)
    sil_score = float(silhouette_score(X_tfidf, clusters))

    # Map cluster IDs to human-readable themes based on keywords
    cluster_names = {}
    for cid in range(3):
        sample_texts = " ".join([cleaned_texts[i] for i in range(len(cleaned_texts)) if clusters[i] == cid])
        if "overheating" in sample_texts or "drains" in sample_texts:
            cluster_names[cid] = "Cluster A: Hardware & Battery Defects"
        elif "packaging" in sample_texts or "courier" in sample_texts:
            cluster_names[cid] = "Cluster B: Delivery & Packaging Issues"
        else:
            cluster_names[cid] = "Cluster C: Positive Quality & Performance"

    cluster_counts = [int(np.sum(clusters == c)) for c in range(3)]

    # 4. CLASSIFICATION: Logistic Regression Sentiment Model
    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X_combined, sentiments)
    preds = clf.predict(X_combined)
    acc = float(accuracy_score(sentiments, preds)) * 100.0
    f1 = float(f1_score(sentiments, preds, average="macro")) * 100.0

    # 5. ASSOCIATION RULE MINING: Calculate Support, Confidence, Lift
    transactions = []
    for i, txt in enumerate(cleaned_texts):
        words = set(txt.split())
        items = {f"aspect:{w}" for w in ASPECT_KEYWORDS if w in words}
        items.add(f"Sentiment:{sentiments[i]}")
        transactions.append(items)

    n_trans = len(transactions)
    candidate_pairs = [
        ("aspect:battery", "Sentiment:Negative"),
        ("aspect:overheating", "Sentiment:Negative"),
        ("aspect:packaging", "Sentiment:Negative"),
        ("aspect:delivery", "Sentiment:Negative"),
        ("aspect:camera", "Sentiment:Positive"),
        ("aspect:charging", "Sentiment:Positive"),
    ]
    rules_list = []
    for ant, con in candidate_pairs:
        ant_cnt = sum(1 for t in transactions if ant in t)
        con_cnt = sum(1 for t in transactions if con in t)
        both_cnt = sum(1 for t in transactions if ant in t and con in t)
        if ant_cnt > 0 and con_cnt > 0 and both_cnt > 0:
            sup = both_cnt / n_trans
            conf = both_cnt / ant_cnt
            lift = conf / (con_cnt / n_trans)
            rules_list.append({
                "ant": ant.replace("aspect:", ""),
                "con": con.replace("Sentiment:", "") + " Sentiment",
                "support": round(sup, 2),
                "confidence": round(conf, 2),
                "lift": round(lift, 2),
            })
    rules_list.sort(key=lambda x: x["lift"], reverse=True)

    return {
        "scaler": scaler,
        "tfidf": tfidf,
        "kmeans": kmeans,
        "cluster_names": cluster_names,
        "cluster_counts": cluster_counts,
        "clf": clf,
        "acc": round(acc, 1),
        "f1": round(f1, 1),
        "sil": round(sil_score, 3),
        "missing_imputed": missing_imputed,
        "rules": rules_list,
    }


PIPELINE = build_mining_pipeline()
init_db()

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReviewPulse AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root { --navy:#14213d; --blue:#2563eb; --light:#f3f6fb; --green:#16865b; --orange:#c77700; }
* { box-sizing:border-box; }
body { margin:0; font-family:Arial,Helvetica,sans-serif; background:var(--light); color:#1f2937; }
header { background:var(--navy); color:white; padding:24px 6%; }
header h1 { margin:0 0 7px; font-size:28px; }
header p { margin:0; color:#dbeafe; }
main { max-width:1150px; margin:24px auto; padding:0 18px 30px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:18px; }
.card { background:white; border-radius:14px; padding:20px; box-shadow:0 4px 18px #17255412; }
h2 { font-size:18px; margin:0 0 14px; color:var(--navy); }
label { display:block; font-size:13px; font-weight:bold; margin:10px 0 5px; }
input, textarea { width:100%; padding:10px; border:1px solid #cbd5e1; border-radius:8px; font-size:14px; font-family:inherit; }
button { margin-top:14px; width:100%; border:0; padding:12px; background:var(--blue); color:white; border-radius:8px; font-size:15px; font-weight:bold; cursor:pointer; }
button:hover { background:#1d4ed8; }
.metric { font-size:24px; font-weight:bold; color:var(--blue); margin-bottom:8px; }
.small { color:#64748b; font-size:13px; margin:4px 0; }
.status { padding:11px; border-radius:9px; background:#ecfdf5; color:#166534; font-weight:bold; margin-top:10px; font-size:14px; }
.warn { background:#fff7ed; color:#9a3412; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { text-align:left; padding:8px 6px; border-bottom:1px solid #e2e8f0; }
th { color:#475569; }
footer { text-align:center; color:#64748b; font-size:12px; padding:0 15px 25px; }
</style>
</head>
<body>
<header>
  <h1>📊 ReviewPulse AI</h1>
  <p>E-Commerce Review Sentiment Classification, Defect Clustering & Association Mining</p>
</header>
<main>
  <div class="grid">
    <section class="card">
      <h2>Enter Customer Review Data</h2>
      <form method="post">
        <label for="review_text">Customer Review Text</label>
        <textarea id="review_text" name="review_text" rows="3" required>{{ review_text }}</textarea>
        <label for="price">Product Price (INR)</label>
        <input id="price" name="price" type="number" min="100" max="200000" step="1" value="{{ price }}" required>
        <label for="rating">Customer Star Rating (1 to 5)</label>
        <input id="rating" name="rating" type="number" min="1" max="5" value="{{ rating }}" required>
        <label for="votes">Helpful Votes</label>
        <input id="votes" name="votes" type="number" min="0" max="5000" value="{{ votes }}" required>
        <button type="submit">Preprocess & Run Data Mining</button>
      </form>
    </section>

    <section class="card">
      <h2>Data Mining Output</h2>
      <p class="small">Predicted Sentiment (Classification)</p>
      <div class="metric">{{ sentiment }}</div>
      <p class="small">Assigned Group (K-Means Clustering)</p>
      <div class="metric" style="font-size:17px">{{ cluster_name }}</div>
      <p class="small">Matched Association Rule: <b>{{ matched_rule }}</b></p>
      <p class="small">Cleaned Text: <i>{{ cleaned_preview }}</i> | Normalized Price: <b>{{ norm_price }}</b></p>
      <div class="status {{ 'warn' if sentiment == 'Negative' else '' }}">{{ action }}</div>
    </section>

    <section class="card">
      <h2>Model Evaluation Metrics</h2>
      <p class="small">Classification Accuracy: <b>{{ acc }}%</b> (F1-Score: <b>{{ f1 }}%</b>)</p>
      <p class="small">K-Means Silhouette Score: <b>{{ sil }}</b></p>
      <p class="small">Missing Records Imputed: <b>{{ missing_imputed }}</b></p>
      <h2 style="margin-top:14px">Top Association Rules (Apriori)</h2>
      <table>
        <thead><tr><th>Rule (Antecedent → Consequent)</th><th>Sup</th><th>Conf</th><th>Lift</th></tr></thead>
        <tbody>
          {% for r in rules %}
          <tr><td>{{ r.ant }} → {{ r.con }}</td><td>{{ r.support }}</td><td>{{ r.confidence }}</td><td>{{ r.lift }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>
  </div>

  <div class="grid" style="margin-top:18px">
    <section class="card">
      <h2>K-Means Cluster Distribution Chart</h2>
      <canvas id="clusterChart" height="170"></canvas>
    </section>

    <section class="card">
      <h2>Recent Review Mining History (SQLite)</h2>
      {% if history %}
      <table>
        <thead><tr><th>Time</th><th>Sentiment</th><th>Cluster</th><th>Rule</th><th>Action</th></tr></thead>
        <tbody>
          {% for row in history %}
          <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td><td>{{ row[4] }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p class="small">No saved records yet. Submit the form to log your first review analysis.</p>
      {% endif %}
    </section>
  </div>
</main>
<footer>Mini-project demonstration • Flask + Python + Scikit-learn + SQLite + Chart.js</footer>
<script>
new Chart(document.getElementById('clusterChart'), {
  type: 'bar',
  data: {
    labels: ['Hardware/Battery Defects', 'Delivery/Packaging', 'Positive Quality'],
    datasets: [{
      label: 'Reviews per Cluster (K-Means)',
      data: {{ cluster_counts|safe }},
      backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
      borderRadius: 6
    }]
  },
  options: { responsive: true, scales: { y: { beginAtZero: true } } }
});
</script>
</body>
</html>
"""


def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT substr(created_at, 12, 8), sentiment, cluster_name, matched_rule, action
            FROM review_history ORDER BY id DESC LIMIT 5
        """).fetchall()


@app.route("/", methods=["GET", "POST"])
def index():
    review_text = "Battery drains fast and phone overheating badly during charging"
    price, rating, votes = 15999.0, 1, 20

    if request.method == "POST":
        try:
            review_text = request.form.get("review_text", review_text).strip()
            price = float(request.form.get("price", 15999))
            rating = int(request.form.get("rating", 1))
            votes = float(request.form.get("votes", 20))
            if not (1 <= rating <= 5 and price >= 0 and votes >= 0):
                raise ValueError("Invalid input range")
        except (ValueError, TypeError):
            return "Invalid input: enter valid price, rating (1-5), and non-negative votes.", 400

    # Step 1: Preprocess input
    cleaned = clean_text(review_text)
    norm_vals = PIPELINE["scaler"].transform([[price, votes]])[0]
    norm_price = round(float(norm_vals[0]), 3)
    vec_tfidf = PIPELINE["tfidf"].transform([cleaned]).toarray()
    vec_combined = np.hstack([vec_tfidf, [norm_vals]])

    # Step 2: Predict Sentiment Class & Cluster ID
    sentiment = str(PIPELINE["clf"].predict(vec_combined)[0])
    cluster_id = int(PIPELINE["kmeans"].predict(vec_tfidf)[0])
    cluster_name = PIPELINE["cluster_names"][cluster_id]

    # Step 3: Match Association Rule
    words = set(cleaned.split())
    matched_rule = "general_feedback → Neutral"
    for r in PIPELINE["rules"]:
        if r["ant"] in words:
            matched_rule = f"{r['ant']} → {r['con']} (Lift: {r['lift']})"
            break

    # Step 4: Decision Support Action
    if sentiment == "Negative" and ("battery" in words or "overheating" in words):
        action = "Alert QA Team: Hardware/Battery Defect Batch Inspection"
    elif sentiment == "Negative" and ("delivery" in words or "packaging" in words):
        action = "Alert Logistics: Courier & Packaging Audit Required"
    elif sentiment == "Negative":
        action = "Priority Customer Support Follow-up Required"
    elif sentiment == "Neutral":
        action = "Monitor Product Feedback Trend"
    else:
        action = "Verified Positive Review: Promote on Product Page"

    if request.method == "POST":
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                INSERT INTO review_history
                (created_at, review_text, price, rating, sentiment, cluster_name, matched_rule, action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(timespec="seconds"), review_text, price, rating,
                  sentiment, cluster_name, matched_rule, action))

    return render_template_string(
        PAGE,
        review_text=review_text,
        price=int(price),
        rating=rating,
        votes=int(votes),
        sentiment=sentiment,
        cluster_name=cluster_name,
        matched_rule=matched_rule,
        cleaned_preview=cleaned,
        norm_price=norm_price,
        action=action,
        acc=PIPELINE["acc"],
        f1=PIPELINE["f1"],
        sil=PIPELINE["sil"],
        missing_imputed=PIPELINE["missing_imputed"],
        rules=PIPELINE["rules"],
        cluster_counts=PIPELINE["cluster_counts"],
        history=get_history(),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)