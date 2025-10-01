# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirement files first (for caching)
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend code
COPY . .
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
