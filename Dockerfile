FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install streamlit --upgrade

# Copy the rest of the application
COPY . .

# Set environment variable for port
ENV PORT=8501

# Expose port
EXPOSE 8501

# Run the application
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
