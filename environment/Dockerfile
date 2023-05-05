# hash:sha256:45bc22ac8d0c33607026d91e252c5b960f2b6ed6f638bff69cf5c90894917483
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        numpy==1.22.4 \
        pandas==1.5.3 \
        scipy==1.9.1 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    anndata==0.8.0 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    shapely \
    squidpy==1.2.3
