# SentenceTransformersServer

This project implements an API server for SentenceTransformers' embedders which aims to be a drop-in replacement for [llama-cpp-python's webserver embeddings](https://github.com/abetlen/llama-cpp-python#web-server).

## Why should I use it?
If you are using the llama-cpp-python webserver and you are experiencing poor embedding performance, now you can try another embedder without modifying your client code!


## Quickstart

Clone the project
```bash
git clone https://github.com/AlessandroSpallina/SentenceTransformersServer.git
```

Choose the model you want to use for the embedding, [here](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/) all the supported models from SentenceTransformers.

Keep in mind that the right model to pick mostly depends on your use case and your data, so try some similarity search in order to understand which model fits better your needs. For example you can start by comparing [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). You might want to check [this leaderboard](https://huggingface.co/spaces/mteb/leaderboard) too.

When you are done, rename the .env.example to .env and modify the model name accordingly. If you are behind a corporate proxy remember to uncomment the right section in the docker-compose.yml file.

Then impose your hands on the keyboard, close your eyes and 
```bash
docker compose up
```