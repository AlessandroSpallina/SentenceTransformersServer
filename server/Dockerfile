FROM python:3.10.11-slim-bullseye

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124


COPY . /code/server

WORKDIR /code/server

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8081"]