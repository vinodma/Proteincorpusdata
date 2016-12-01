FROM jupyter/datascience-notebook
MAINTAINER deepdive-dev@googlegroups.com

# install dependencies of Snorkel's dependencies
USER root
RUN apt-get update && apt-get install -qy \
        python-dev libxml2-dev libxslt1-dev zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
USER jovyan

# grab a shallow clone of Snorkel
ARG BRANCH=master
ENV BRANCH=$BRANCH
RUN git clone https://github.com/HazyResearch/snorkel.git \
        --branch $BRANCH \
        --depth 10 \
 && cd snorkel \
 && git submodule update --init --recursive

# set up Snorkel
WORKDIR snorkel
RUN pip2 install --requirement python-package-requirement.txt \
 && pip3 install --requirement python-package-requirement.txt
RUN ./install-parser.sh
WORKDIR ..

ENV SNORKELHOME=/home/jovyan/work/snorkel

RUN ln -sfn /work . \
 && mkdir -p \
    'snorkel IS NOT PERSISTENT!!! YOUR CHANGES WILL DISAPPEAR!!!'/'PLEASE GO BACK' \
    'work is the right place to keep your files with the --volume flag'/'PLEASE GO BACK' \
 && chmod -R a-w 'snorkel '* 'work '* \
 && chmod a-w .
 
