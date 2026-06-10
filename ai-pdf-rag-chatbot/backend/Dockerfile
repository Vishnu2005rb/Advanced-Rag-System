FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compiling C++ packages (like ChromaDB/hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set up a writable data directory for non-root container users (like Hugging Face UID 1000)
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Expose port 8000 as default
EXPOSE 8000

# Start Uvicorn, falling back to port 8000 if PORT environment variable is not defined (e.g. on Render/Railway/HuggingFace)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
