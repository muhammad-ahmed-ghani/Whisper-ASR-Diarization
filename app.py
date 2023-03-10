import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:asgi_app", host="0.0.0.0", reload=True, port=8000, log_level="info")