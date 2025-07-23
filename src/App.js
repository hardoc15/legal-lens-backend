import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:5000/predict", { text });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="App">
      <h1>LegalLens AI</h1>
      <textarea
        rows="6"
        cols="60"
        placeholder="Paste your legal clause here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br />
      <button onClick={handleSubmit}>Classify</button>
      {result && (
        <div className="result">
          <h3>
            Prediction: {result.label === 1 ? "🟥 Risky" : "🟩 Not Risky"}
          </h3>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;
