from fastapi import FastAPI
from fastapi_app.controller import router
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Chart Generator")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.include_router(router)

import uvicorn

if __name__ == "__main__":
    # uvicorn.run("main:app", host="127.0.0.1", port=9090, reload=True)
    # Use below for docker deployment
    uvicorn.run("main:app", host="127.0.0.1", port=9010, reload=True)
