FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the analytics database directory (if needed)
RUN touch analytics.db

# Expose the port used by the health check
EXPOSE 7860

# Run the bot
CMD ["python", "bot.py"]
