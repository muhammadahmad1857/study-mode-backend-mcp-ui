from fastapi import FastAPI
from routes.chat import router as chat_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Study Mode", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Or use ["*"] to allow all origins
    allow_credentials=True,     # Allow cookies/auth headers
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)
@app.get("/")
def Home():
    return {"message":"Welcome to Study Mode"}
# Include routes
app.include_router(chat_router)
