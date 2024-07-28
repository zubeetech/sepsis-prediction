# Base Image
FROM python:3.11.3

#  Copy the requirements.txt in to the tmp folder
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --timeout 300000 -r /tmp/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 4434

CMD ["streamlit", "run", "main.py", --server.port==8501]