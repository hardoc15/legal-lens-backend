"""
Simple tests for the Legal Lens Backend
Created to demonstrate testing and file creation capabilities
"""

import json
import unittest
from unittest.mock import patch, MagicMock
from api import app
from utils import validate_input_text, format_prediction_response, get_health_status


class TestLegalLensBackend(unittest.TestCase):
    """Test suite for Legal Lens Backend"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertEqual(data['service'], 'legal-lens-backend')
    
    def test_validate_input_text_valid(self):
        """Test input validation with valid text"""
        valid_text = "This is a valid legal clause for termination."
        is_valid, error_msg = validate_input_text(valid_text)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_input_text_empty(self):
        """Test input validation with empty text"""
        is_valid, error_msg = validate_input_text("")
        self.assertFalse(is_valid)
        self.assertIn("empty", error_msg.lower())
    
    def test_validate_input_text_none(self):
        """Test input validation with None"""
        is_valid, error_msg = validate_input_text(None)
        self.assertFalse(is_valid)
        self.assertIn("non-empty string", error_msg)
    
    def test_validate_input_text_too_long(self):
        """Test input validation with text that's too long"""
        long_text = "a" * 50001  # Exceeds 50k limit
        is_valid, error_msg = validate_input_text(long_text)
        self.assertFalse(is_valid)
        self.assertIn("maximum length", error_msg)
    
    def test_format_prediction_response(self):
        """Test prediction response formatting"""
        mock_predictions = [
            {"label": "Termination", "confidence": 0.95},
            {"label": "License", "confidence": 0.87}
        ]
        
        response = format_prediction_response(mock_predictions)
        
        self.assertEqual(response['status'], 'success')
        self.assertEqual(response['total_clauses'], 2)
        self.assertEqual(response['predictions'], mock_predictions)
        self.assertIn('timestamp', response)
    
    def test_format_prediction_response_with_metadata(self):
        """Test prediction response formatting with metadata"""
        mock_predictions = [{"label": "Liability", "confidence": 0.92}]
        metadata = {"input_length": 150}
        
        response = format_prediction_response(mock_predictions, metadata)
        
        self.assertEqual(response['metadata'], metadata)
        self.assertEqual(response['total_clauses'], 1)
    
    def test_get_health_status(self):
        """Test health status function"""
        status = get_health_status()
        
        self.assertEqual(status['status'], 'healthy')
        self.assertEqual(status['service'], 'legal-lens-backend')
        self.assertEqual(status['version'], '1.0.0')
        self.assertIn('timestamp', status)
    
    def test_predict_endpoint_no_text(self):
        """Test predict endpoint with no text"""
        response = self.app.post('/predict', 
                                json={},
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_empty_text(self):
        """Test predict endpoint with empty text"""
        response = self.app.post('/predict', 
                                json={"text": ""},
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch('api.classify_clauses')
    def test_predict_endpoint_success(self, mock_classify):
        """Test predict endpoint with valid input"""
        # Mock the classification function
        mock_classify.return_value = [
            {"label": "Termination", "confidence": 0.95}
        ]
        
        response = self.app.post('/predict', 
                                json={"text": "This contract may be terminated with 30 days notice."},
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['total_clauses'], 1)
        self.assertIn('predictions', data)
        self.assertIn('timestamp', data)


if __name__ == '__main__':
    unittest.main()