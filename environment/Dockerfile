# hash:sha256:a5fefa231c2e66bd655239ed3bc637248fe8d02d7a75e8f1742c9376b3091435
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.7.2 \
        pandas==2.0.3 \
        scipy==1.9.1 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --upgrade-strategy=only-if-needed \
    anndata==0.10.2 \
    colorcet==3.0.1 \
    geojson==3.0.1 \
    geopandas==0.13.1 \
    hdbscan==0.8.33 \
    nibabel==5.1.0 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    shapely==2.0.1

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall
