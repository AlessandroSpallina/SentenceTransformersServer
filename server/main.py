import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
from sentence_transformers import SentenceTransformer
import logging
import sys
import os
# import asyncio
# import concurrent.futures

logFormatter = logging.Formatter("%(asctime)s - [%(levelname)s] %(name)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(logFormatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)


logger.info(f"Loading model {os.environ['MODEL']}")
model = SentenceTransformer(os.environ['MODEL'], cache_folder='/code/server/models')
# pool = model.start_multi_process_pool()#['cpu'] * os.cpu_count() * int(0.75))

# process_pool = concurrent.futures.ProcessPoolExecutor()

logger.info("Starting fastapi server")

app = FastAPI(title="SentenceTransformersServer")

# Enable CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(description="The input to embed.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


@app.get("/")
def read_root():
    return {"status": "up"}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    logger.info(f"Received {request}")

    # model.encode() doens't support strings with only space, below a workaround
    if request.input == ' ':
        request.input = '.'

    # embeddings = model.encode_multi_process(request.input, pool)


    embeddings = model.encode(request.input).tolist()

    if type(embeddings[0]) == list:
        to_ret = [{
            'object': 'embedding',
            'embedding': embedding,
            'index': i
        } for i, embedding in enumerate(embeddings)]
    else:
        to_ret = [{
            'object': 'embedding',
            'embedding': embeddings,
            'index': 0
        }]

    return {
        'object': 'list',
        'data': to_ret,
        'model': os.environ['MODEL'],
        'usage': { # If someone is using this field we should find the right way to fill these values
            'prompt_tokens': len(request.input),
            'total_tokens': len(request.input)
        }
    }
