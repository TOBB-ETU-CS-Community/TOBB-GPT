FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    &&  pip install --requirement requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]