# hash:sha256:6414fd07639184613af87d0b8aa4a10e662afa1e1f7787298a23be9cea959f76
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY
ARG CUSTOM_KEY_2

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        scipy==1.9.1 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    anndata==0.8.0 \
    colorcet==3.0.1 \
    geojson==3.0.1 \
    geopandas==0.13.1 \
    hdbscan==0.8.33 \
    nibabel==5.1.0 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    shapely==2.0.1 \
    streamlit==1.30.0 \
    zarr==2.17.0

COPY postInstall /
RUN /postInstall
