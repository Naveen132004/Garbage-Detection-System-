# Hugging Face Spaces (Docker SDK) image for the Garbage Detection System
FROM python:3.11-slim

# System libraries OpenCV needs on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

COPY --chown=user . .

# Download the model weights at build time unless Weights/best.pt is already in the image.
# Override with a Space variable named MODEL_URL to use your own trained model.
ARG MODEL_URL=https://huggingface.co/kendrickfff/waste-classification-yolov8-ken/resolve/main/yolov8n-waste-12cls-best.pt
RUN mkdir -p Weights && if [ ! -f Weights/best.pt ]; then curl -fL -o Weights/best.pt "$MODEL_URL"; fi \
    && mkdir -p uploads results data maps && chown -R user:user /app

USER user
ENV HOST=0.0.0.0 \
    PORT=7860 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 7860
CMD ["sh", "-c", "gunicorn app1:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]
