# Use a slim Python 3.10 image to keep it lightweight
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Poppler for PDF imaging, Tesseract for OCR)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-tur \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the data directory (for PDFs and FAISS)
RUN mkdir -p data/pdfs

# Streamlit-specific environment variables for Render
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck for Render
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
