from fastapi import FastAPI
from db.config import lifespan

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}