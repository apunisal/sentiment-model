from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import hashlib

app = FastAPI(title="Sentiment Analysis API")
ort_session = ort.InferenceSession("sentiment_model.onnx")

class TextRequest(BaseModel):
    text: str

def hash_token(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % 12000

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.lower()
    words = text.split()
    tokens = []
    
    i = 0
    while i < len(words):
        if words[i] in ["not", "no", "never", "dont", "cant", "isnt", "wasnt"] and (i + 1) < len(words):
            combined_token = f"{words[i]}_{words[i+1]}"
            tokens.append(hash_token(combined_token))
            i += 2
        else:
            tokens.append(hash_token(words[i]))
            i += 1
            
    if len(tokens) > 20:
        tokens = tokens[:20]
    else:
        tokens = tokens + [0] * (20 - len(tokens))
        
    input_data = np.array([tokens], dtype=np.int64)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    
    score = float(ort_outs[0][0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    
    return {"sentiment": sentiment, "score": round(score, 4)}

@app.get("/")
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>✨ Cute Sentiment Analyzer ✨</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Nunito:wght=400;700;900&display=swap');
            body { 
                font-family: 'Nunito', sans-serif; margin: 0; padding: 0; min-height: 100vh;
                display: flex; align-items: center; justify-content: center;
                background: linear-gradient(-45deg, #a1c4fd, #c2e9fb, #ff9a9e, #fecfef);
                background-size: 400% 400%; animation: gradientBG 15s ease infinite; color: #4a4a4a;
            }
            @keyframes gradientBG {
                0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; }
            }
            .container { 
                background-color: rgba(255, 255, 255, 0.95); padding: 40px; border-radius: 25px; 
                box-shadow: 0 15px 35px rgba(0,0,0,0.15); max-width: 650px; text-align: center; position: relative; z-index: 10;
                margin: 20px;
            }
            h1 { color: #ff6b81; font-weight: 900; font-size: 2.5em; margin-top: 0; text-shadow: 2px 2px 4px rgba(255, 107, 129, 0.2); }
            .info-box { background-color: #f1f2f6; padding: 20px; border-radius: 15px; margin-bottom: 25px; font-size: 1.05em; line-height: 1.6; text-align: left; }
            textarea { 
                width: 100%; height: 100px; padding: 15px; border: 3px dashed #a4b0be; border-radius: 15px; 
                font-size: 16px; font-family: 'Nunito', sans-serif; margin-bottom: 20px; box-sizing: border-box; resize: none; transition: 0.3s;
            }
            textarea:focus { border-color: #ff6b81; outline: none; box-shadow: 0 0 15px rgba(255, 107, 129, 0.3); }
            button { 
                background: linear-gradient(to right, #ff6b81, #ff4757); color: white; border: none; padding: 15px 30px; 
                font-size: 18px; font-weight: 700; border-radius: 50px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; 
                box-shadow: 0 5px 15px rgba(255, 71, 87, 0.4);
            }
            button:hover { transform: translateY(-3px) scale(1.05); box-shadow: 0 8px 20px rgba(255, 71, 87, 0.6); }
            #result { margin-top: 25px; font-size: 22px; font-weight: 900; min-height: 35px; }
            .positive { color: #2ed573; animation: pop 0.5s ease; }
            .negative { color: #ff4757; animation: pop 0.5s ease; }
            @keyframes pop { 0% { transform: scale(0.8); opacity: 0; } 50% { transform: scale(1.1); } 100% { transform: scale(1); opacity: 1; } }
            .animal { position: absolute; z-index: 1; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); animation: float 6s ease-in-out infinite; }
            .cat { top: -80px; left: -60px; width: 140px; animation-delay: 0s; }
            .dog { bottom: -80px; right: -60px; width: 150px; animation-delay: 3s; }
            @keyframes float { 0% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-12px) rotate(3deg); } 100% { transform: translateY(0px) rotate(0deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="https://media.giphy.com/media/3o6Zt4ZQb0Peu0f1Oo/giphy.gif" alt="Animated Cat" class="animal cat">
            <img src="https://media.giphy.com/media/IgKeRXLiKepHi8aDa3/giphy.gif" alt="Cute Puppy" class="animal dog">

            <h1>✨ Vibe Check ✨</h1>
            
            <div class="info-box">
                <strong>What is a Sentiment Model? 🤔</strong><br>
                A sentiment model is an AI tool that analyzes text to determine the emotional tone behind it. It classifies sentences into categories like <strong style="color:#2ed573;">Positive</strong> or <strong style="color:#ff4757;">Negative</strong>.<br><br>
                <em>Examples:</em><br>
                "I absolutely love this new phone the battery life is amazing!" ➔ <strong style="color:#2ed573;">Positive ✨</strong><br>
                "The service was terrible and my food was cold." ➔ <strong style="color:#ff4757;">Negative 🌧️</strong>
            </div>

            <textarea id="inputText" placeholder="Type your own thoughts here... let's check the vibes!"></textarea>
            <br>
            <button onclick="analyzeText()">Analyze Vibes! 🪄</button>

            <div id="result">Waiting for your text...</div>
        </div>

        <script>
            async function analyzeText() {
                const text = document.getElementById("inputText").value;
                if (!text.trim()) {
                    document.getElementById("result").innerHTML = "Oops! You forgot to type something! 🙈";
                    return;
                }
                document.getElementById("result").innerHTML = "Checking vibes... 🐾";
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    const data = await response.json();
                    let colorClass = data.sentiment === 'Positive' ? 'positive' : 'negative';
                    let emoji = data.sentiment === 'Positive' ? '✨🐶✨' : '😿🌧️';
                    document.getElementById("result").innerHTML = 
                        `<span class="${colorClass}">${data.sentiment} Vibe! ${emoji}</span> <br><span style="font-size: 14px; color: #888; font-weight: normal;">(AI Certainty Score: ${data.score})</span>`;
                } catch (error) {
                    document.getElementById("result").innerHTML = "Uh oh, something broke! 🚨";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
