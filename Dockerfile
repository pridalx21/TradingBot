FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit --upgrade

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=10000
ENV HOST=0.0.0.0

# Expose the port
EXPOSE ${PORT}

# Command to run the application
CMD streamlit run main.py --server.port=${PORT} --server.address=${HOST}
