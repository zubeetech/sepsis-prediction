# Base Image
FROM python:3.11.3

#  Copy the requirements.txt in to the tmp folder
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --timeout 300000 -r /tmp/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 80

CMD ["uvicorn", "api:app", "--host", "0.0.0.0" "--port", "80" --reload ]