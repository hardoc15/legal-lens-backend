# LegalLens AI — Backend

This is the backend service for **LegalLens AI**, a contract clause analysis platform that uses transformer-based NLP models to detect and classify legal risks. This backend powers the API endpoints that receive contract text, run inference using a fine-tuned LegalBERT model, and return clause-level classifications.

> 🚀 The frontend client is hosted separately at [legal-lens-frontend](https://github.com/YOUR_USERNAME/legal-lens-frontend).

---

## 🧠 Features

- REST API built with Flask
- Fast clause-level classification using a fine-tuned LegalBERT model
- Supports multi-clause input for bulk analysis
- CORS enabled for frontend integration
- Easy model loading and inference
- **Input validation and error handling**
- **Health check endpoint for monitoring**
- **Structured logging for prediction requests**
- **Comprehensive test suite**

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hardoc15/legal-lens-backend.git
cd legal-lens-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python api.py
```

The server will start on `http://localhost:7860`

## 📡 API Endpoints

### `POST /predict`
Analyze legal contract text and classify clauses.

**Request:**
```json
{
  "text": "This contract may be terminated with 30 days written notice."
}
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-08-09T18:50:00.000Z",
  "total_clauses": 1,
  "predictions": [
    {
      "label": "Termination",
      "confidence": 0.95
    }
  ],
  "metadata": {
    "input_length": 65
  }
}
```

### `GET /health`
Health check endpoint for monitoring application status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-09T18:50:00.000Z",
  "version": "1.0.0",
  "service": "legal-lens-backend"
}
```

## 🧪 Testing

Run the test suite:
```bash
python -m unittest test_api.py -v
```

## 📂 Project Structure

```
legal-lens-backend/
├── api.py              # Main Flask application
├── predict.py          # ML model inference
├── utils.py            # Utility functions
├── test_api.py         # Test suite
├── preprocessing.py    # Data preprocessing
├── train.py           # Model training
├── evaluate.py        # Model evaluation
├── requirements.txt   # Dependencies
├── model/            # Model files
│   ├── model.pt
│   └── tokenizer.pkl
└── README.md
```
