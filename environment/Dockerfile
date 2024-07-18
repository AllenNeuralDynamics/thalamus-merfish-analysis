# hash:sha256:28733b41885b80541c58e2586c408a4c608370874bf02213431573799da4c3c6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

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

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    anndata==0.8.0 \
    colorcet==3.0.1 \
    colour-science==0.4.4 \
    geojson==3.0.1 \
    geopandas==0.13.1 \
    hdbscan==0.8.33 \
    nibabel==5.1.0 \
    pydantic==2.7.1 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    streamlit==1.35.0 \
    zarr==2.17.0

COPY postInstall /
RUN /postInstall
