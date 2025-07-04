FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install pip + upgrade
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Start the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
