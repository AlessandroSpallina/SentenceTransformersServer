FROM python:3.10.11-slim-bullseye

# if you are behind a corporate proxy uncomment below to pass these values at container build time
# ARG HTTP_PROXY
# ARG HTTPS_PROXY
# ARG NO_PROXY
# ENV HTTP_PROXY=$HTTP_PROXY
# ENV HTTPS_PROXY=$HTTPS_PROXY
# ENV NO_PROXY=$NO_PROXY
# ENV http_proxy=$HTTP_PROXY
# ENV https_proxy=$HTTPS_PROXY
# ENV no_proxy=$NO_PROXY

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
# RUN pip install torch --index-url https://download.pytorch.org/whl/cu124


COPY . /code/server

WORKDIR /code/server

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8081"]
