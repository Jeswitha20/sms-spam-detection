# app.py
from flask import Flask, request, jsonify, render_template
import joblib

# Load Models and Vectorizer
svc_model = joblib.load("svc_model.pkl")
catboost_model = joblib.load("catboost_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form.get("sms")
    if not sms:
        return jsonify({"error": "Please enter a message"}), 400
    
    # Preprocess Input
    sms_tfidf = vectorizer.transform([sms])
    
    # Predict
    svc_pred = svc_model.predict(sms_tfidf)[0]
    catboost_pred = catboost_model.predict(sms_tfidf)[0]
    
    # Majority Voting
    final_pred = "spam" if svc_pred == 1 or catboost_pred == 1 else "ham"
    
    return jsonify({"result": final_pred})

if __name__ == '__main__':
    app.run(debug=True)
