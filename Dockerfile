FROM python:3.10-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required build tools and system dependencies
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    openssl-dev \
    gcc \
    musl-dev \
    libstdc++ \
    g++ \
    linux-headers \
    jpeg-dev \
    zlib-dev \
    libjpeg \
    freetype-dev \
    lapack-dev \
    blas-dev \
    libxml2-dev \
    libxslt-dev \
    git

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "api:app"]
