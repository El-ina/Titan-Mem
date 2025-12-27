from fastapi import FastAPI
from api.routes import router
from storage.sessions import ensure_dirs

app = FastAPI()

app.include_router(router)

ensure_dirs()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
