FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/tmp/huggingface

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY app.py /app/app.py
COPY src/ /app/src/
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["python", "app.py"]
