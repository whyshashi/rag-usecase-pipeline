import uvicorn
from fastapi import FastAPI
from main import run_pipeline

app = FastAPI()

@app.post("/ask")
def read_root(data:dict):
    try:
        if "query" not in data:
            return {"error": "Missing 'query' in request body"}
        user_query = data.get("query")
        final_answer = run_pipeline(user_query)
        return {"query": user_query, "answer": final_answer}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Starts the server on port 8000
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)