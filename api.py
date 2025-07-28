from flask import Flask, request, jsonify
from predict import classify_clauses
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        results = classify_clauses(text)
        return jsonify({"clauses": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
