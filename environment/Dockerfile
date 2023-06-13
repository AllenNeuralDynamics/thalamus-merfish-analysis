# hash:sha256:f580b44246250678a4c5fba4637e06777bd4a669a148516bdef1dc0856ad0bd5
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        pandas==1.5.3 \
        scipy==1.9.1 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    anndata==0.8.0 \
    bg-atlasapi==1.0.2 \
    colorcet==3.0.1 \
    geopandas==0.13.1 \
    hdbscan==0.8.29 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    shapely==2.0.1 \
    squidpy==1.2.3
