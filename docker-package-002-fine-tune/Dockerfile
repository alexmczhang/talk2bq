# Use the official lightweight Python image.
# https://hub.docker.com/_/python
#FROM python:3.9.18-slim
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Needed to build hnswlib during pip install. 
# See here https://github.com/chroma-core/chroma/issues/642
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential

RUN apt-get update; apt-get install curl -y

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["talk2bq-002-fine-tune.py" ]