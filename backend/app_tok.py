from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from tokenizers import Tokenizer
import uvicorn
from contextlib import asynccontextmanager

tokenizer = None
vocab_size = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, vocab_size
    tokenizer = Tokenizer.from_file("src_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    yield
    # (No cleanup needed)

app = FastAPI(lifespan=lifespan)

@app.get("/encode")
async def encode_text(text: str = Query(..., description="Text to encode")):
    tokens = tokenizer.encode(text).ids
    return JSONResponse(content={"tokens": tokens})

if __name__ == "__main__":
    uvicorn.run("app_tok:app", host="0.0.0.0", port=4444, reload=True)





