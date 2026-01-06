# 1. Use a lightweight Python base image (Slim version is faster/smaller)
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first to leverage Docker caching
# (This makes re-builds faster if you only change code, not libraries)
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image size small by removing temporary cache files
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app (main.py and model.pkl)
COPY . .

# 6. Inform Docker that the container listens on port 8080
EXPOSE 8080

# 7. Start the application
# We use 'sh -c' to properly expand the ${PORT} variable that DigitalOcean injects
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]