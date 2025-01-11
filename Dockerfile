FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY start.sh .
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

RUN chmod +x start.sh
CMD ["./start.sh"]
