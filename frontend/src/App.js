import { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post("https://legal-lens-backend.onrender.com/predict", {
        text,
      });
      setResult(res.data.prediction);  // Assuming backend returns { "prediction": "Risky" }
    } catch (err) {
      setResult("Error connecting to backend.");
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Legal Clause Risk Classifier</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows="6"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a legal clause here..."
        />
        <br />
        <button type="submit">Classify</button>
      </form>

      {loading && <p>Analyzing...</p>}
      {result && <p>Prediction: <strong>{result}</strong></p>}
    </div>
  );
}

export default App;
