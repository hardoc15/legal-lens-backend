"""
Utility functions for the Legal Lens Backend
Created to demonstrate file creation capabilities
"""

import json
from datetime import datetime
from typing import List, Dict, Any


def format_prediction_response(predictions: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Format prediction results with additional metadata for API responses.
    
    Args:
        predictions: List of prediction dictionaries with 'label' and 'confidence'
        metadata: Optional metadata to include in response
    
    Returns:
        Formatted response dictionary
    """
    response = {
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "total_clauses": len(predictions),
        "status": "success"
    }
    
    if metadata:
        response["metadata"] = metadata
        
    return response


def validate_input_text(text: str) -> tuple[bool, str]:
    """
    Validate input text for legal document analysis.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    
    if len(text.strip()) == 0:
        return False, "Text cannot be empty or only whitespace"
    
    if len(text) > 50000:  # 50k character limit
        return False, "Text exceeds maximum length of 50,000 characters"
    
    return True, ""


def log_prediction_request(text_length: int, predictions: List[Dict[str, Any]]) -> None:
    """
    Log prediction request details for monitoring and debugging.
    
    Args:
        text_length: Length of input text
        predictions: Prediction results
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text_length": text_length,
        "num_predictions": len(predictions),
        "prediction_summary": {
            label: sum(1 for p in predictions if p["label"] == label)
            for label in set(p["label"] for p in predictions)
        }
    }
    
    # In a real application, this would write to a proper logging system
    print(f"[PREDICTION LOG] {json.dumps(log_entry)}")


def get_health_status() -> Dict[str, Any]:
    """
    Get application health status.
    
    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "legal-lens-backend"
    }