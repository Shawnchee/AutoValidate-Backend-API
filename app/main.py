from fastapi import FastAPI

app = FastAPI()


@app.get("/neekgirl")
async def root(hello : str):
    return {"message": hello}