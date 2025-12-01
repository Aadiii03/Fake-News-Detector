from flask import Flask, render_template, request
import os
import joblib

# Tell Flask where to find static & templates (matches your folder names)
app = Flask(__name__, static_folder='static', template_folder='Templates')

# ---------- LOAD MODELS CORRECTLY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# These are exactly the files you have in /models (from your screenshot)
nb_model_path = os.path.join(MODELS_DIR, "nb_model.pkl")
lr_model_path = os.path.join(MODELS_DIR, "lr_model.pkl")
vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")

nb_model = joblib.load(nb_model_path)
lr_model = joblib.load(lr_model_path)
vectorizer = joblib.load(vectorizer_path)

# ---------- ROUTES ----------

@app.route("/", methods=["GET"])
def home():
    # landing page with hero + quick form
    return render_template("home.html")


@app.route("/detector", methods=["GET", "POST"])
def detector():
    prediction = None
    score = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "")

        if news_text.strip():
            X = vectorizer.transform([news_text])

            nb_prob = nb_model.predict_proba(X)[0][1]
            lr_prob = lr_model.predict_proba(X)[0][1]
            hybrid_prob = (nb_prob + lr_prob) / 2.0

            score = round(hybrid_prob * 100, 2)
            prediction = "FAKE" if hybrid_prob > 0.5 else "REAL"

    return render_template("detector.html", prediction=prediction, score=score)


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/resources")
def resources():
    return render_template("resources.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
