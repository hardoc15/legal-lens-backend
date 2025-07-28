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
