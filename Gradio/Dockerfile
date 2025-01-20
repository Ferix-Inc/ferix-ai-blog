FROM python:3.12.8-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gradio" "app.py"]
