FROM apache/airflow:2.7.3-python3.10

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libreoffice \
        libreoffice-writer \
        libreoffice-java-common \
        default-jre \
        default-jdk \
        poppler-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN python -m pip install --user --no-cache-dir -r /requirements.txt
