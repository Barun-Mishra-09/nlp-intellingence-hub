from fastapi import FastAPI
from pydantic import BaseModel
from src.analyzer import NLPAnalyzer

app = FastAPI()
analyzer = NLPAnalyzer()

class Input(BaseModel):
    text: str
    categories: list[str] | None = None

@app.post("/analyze")
def analyze(data: Input):
    return analyzer.analyze_all(data.text, data.categories)
