FROM ubuntu:focal

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TINI_VERSION v0.16.1
ENV PATH="/opt/phenotrex/py38-venv/bin:$PATH"
ENV TMPDIR="/mnt"

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ADD . /opt/phenotrex
WORKDIR /opt/phenotrex

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing \
    && apt-get install -y bzip2 ca-certificates cmake libssl-dev git pigz build-essential python3-dev python3-venv \
    && apt-get clean \
    && python3 -m venv --system-site-packages py38-venv \
    && . py38-venv/bin/activate \
    && pip install pytest pytest-cov \
    && make full-install \
    && make test \
    && make clean \
    && rm -rf docs tests

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
