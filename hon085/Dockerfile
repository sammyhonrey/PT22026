FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/fetch_assets.py

EXPOSE 7860
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
