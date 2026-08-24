FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA_VERSION=py39_24.5.0-0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git make && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
    -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:${PATH}
ENV PYTHONNOUSERSITE=1
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

WORKDIR /workspace/RTDFL
COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rtdfl-ae"]
CMD ["make", "smoke"]
