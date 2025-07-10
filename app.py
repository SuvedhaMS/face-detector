from flask import Flask, render_template, send_file
import pandas as pd
import os
import pickle

app = Flask(__name__)

@app.route("/")
def dashboard():
    # Load visitor logs
    visitor_log_path = "../visitor_log.csv"
    if os.path.exists(visitor_log_path):
        df = pd.read_csv(visitor_log_path, header=None, names=["Timestamp", "Person ID", "Image"])
        df = df[df["Image"].astype(str).str.contains("logs/")]  # Only keep video-based image logs
        df = df.tail(10)
        df["Image"] = df["Image"].astype(str)
    else:
        df = pd.DataFrame(columns=["Timestamp", "Person ID", "Image"])

    # Load face database
    db_path = "../faces.pkl"
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            face_db = pickle.load(f)
        total_registered = len(face_db)
    else:
        total_registered = 0

    unique_visitors = df["Person ID"].nunique()

    return render_template("index.html",
                           total_registered=total_registered,
                           unique_visitors=unique_visitors,
                           logs=df.to_dict(orient="records"))

@app.route("/logs/<filename>")
def get_image(filename):
    return send_file(f"../logs/{filename}", mimetype="image/jpeg")

@app.route("/download_csv")
def download_csv():
    return send_file("../visitor_log.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
