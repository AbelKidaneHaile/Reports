FROM nvidia/cuda:11.0.3-base-ubuntu20.04
WORKDIR /app

#installing python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
COPY streamlit_webapp.py streamlit_webapp.py
COPY prediction.py prediction.py
COPY main.py main.py
COPY image image
COPY models models

RUN pip3 install -r requirements.txt

CMD ["streamlit run streamlit_webapp.py --server.port 8080"]