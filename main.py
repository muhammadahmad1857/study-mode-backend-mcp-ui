from fastapi import FastAPI
from routes.chat import router as chat_router

app = FastAPI(title="Study Mode", version="1.0")

@app.get("/")
def Home():
    return {"message":"Welcome to Study Mode"}
# Include routes
app.include_router(chat_router)
