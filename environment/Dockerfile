# hash:sha256:75cb2b618ccb9b42f99c17f8b16f4c665b4c9804ebec8b7dec89de7182c52995
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY

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
    nibabel==5.1.0 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    shapely==2.0.1 \
    streamlit==1.30.0

COPY postInstall /
RUN /postInstall
