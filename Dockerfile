FROM python:3.10-slim

WORKDIR /Sigma-Chatbot

COPY . .

RUN pip install --upgrade pip \
    &&  pip install --requirement requirements.txt 
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/chatbot/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
