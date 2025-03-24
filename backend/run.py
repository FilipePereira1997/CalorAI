import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="192.168.1.241", port=5000, reload=True)
