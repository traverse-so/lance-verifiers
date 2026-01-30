FROM python:3.11-slim

# System dependencies for Chromium + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    fonts-noto-cjk \
    libglib2.0-0 \
    libnss3 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    playwright \
    opencv-python-headless \
    Pillow \
    numpy \
    scipy

# Install Playwright browsers
RUN playwright install chromium --with-deps

WORKDIR /workspace

# Keep container alive
CMD ["tail", "-f", "/dev/null"]
