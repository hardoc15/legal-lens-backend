from flask import Flask, request, jsonify
from predict import classify_clauses
from flask_cors import CORS
from utils import format_prediction_response, validate_input_text, log_prediction_request, get_health_status

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Validate input
    is_valid, error_message = validate_input_text(text)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    try:
        results = classify_clauses(text)
        
        # Log the request for monitoring
        log_prediction_request(len(text), results)
        
        # Format response with metadata
        formatted_response = format_prediction_response(
            predictions=results,
            metadata={"input_length": len(text)}
        )
        
        return jsonify(formatted_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify(get_health_status())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
