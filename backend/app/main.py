from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.chat import router as chat_router
from .routers.user_session import router as session_router

app = FastAPI(title="CalorAI - Nutrition Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router)
app.include_router(session_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
