FROM python:3.11.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev libwayland-dev wayland-protocols libxkbcommon-dev libdbus-1-dev linux-libc-dev imagemagick poppler-utils wget && rm -rf /var/lib/apt/lists/*

RUN wget -P /root/.keras-ocr/ https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.h5 https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/crnn_kurapan.h5

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
