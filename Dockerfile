FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app folder into /app
COPY app/ /app/app/


# Expose port
EXPOSE 80

# Run the app using python -m (so imports always work)
CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
