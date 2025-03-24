from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.bert_route import router as bot_router

app = FastAPI(
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(bot_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}
