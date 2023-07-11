# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/MatiasGallo/inmunohistoquimicaWeb.git . && git checkout streamlitcloud && git pull

RUN pip3 install -r requirements.txt

RUN pip install --upgrade "protobuf<=3.20.1"
RUN pip install --upgrade "click<=8.0.4"
RUN pip install "altair<5"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "drawPolygon.py", "--server.port=8501", "--server.address=0.0.0.0"]