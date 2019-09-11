# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-cpu-jupyter.Dockerfile

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=16.04

FROM ubuntu:${UBUNTU_VERSION} AS base

ARG UID=1000
ARG GID=1000

RUN groupdel users
RUN groupadd --gid ${GID} users
RUN useradd --uid ${UID} --gid ${GID} -ms /bin/bash test
RUN echo 'test:test' | chpasswd

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        joe less wget   && \
    apt-get clean 

# &&  rm -rf /var/lib/apt/lists/*

ENV CI_BUILD_PYTHON python

# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

ARG USE_PYTHON_3_NOT_2
#ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python3
#ARG PIP=pip${_PY_SUFFIX}
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

#RUN apt-get update && apt-get install -y \
#    ${PYTHON} \
#    ${PYTHON}-pip
#
#RUN ${PIP} --no-cache-dir install --upgrade \
#    pip \
#    setuptools
#
## Some TF tools expect a "python" binary
#RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

#RUN ${PIP} --no-cache-dir install \
#    Pillow \
#    h5py \
#    keras_applications \
#    keras_preprocessing \
#    matplotlib \
#    mock \
#    numpy \
#    scipy \
#    sklearn \
#    pandas \
#    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
#    enum34

## Install bazel
ARG BAZEL_VERSION=0.15.0
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

#ARG V=21
#COPY Files/bazel-0.${V}.0-installer-linux-x86_64.sh /opt/installer.sh
#RUN chmod +x /opt/installer.sh && \
#    /opt/installer.sh && \
#    rm -f /opt/installer.sh

# Copy in and install Anaconda3 from the shell archive
# Anaconda3-5.1.0-Linux-x86_64.sh
# Here use installation path /opt/Anaconda as an example, the installation command is
#    -b： Batch mode with no PATH modifications to ~/.bashrc. Assumes that you agree to the license agreement. Does not edit the .bashrc or .bash_profile files.
#    -p： Installation prefix/path.
#    -f： Force installation even if prefix -p already exists.
#
# installation_dir='/opt/Anaconda'
# bash ~/Downloads/Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p ${installation_dir}
ARG ANACONDA=Anaconda3-2018.12-Linux-x86_64.sh
COPY Files/${ANACONDA} /opt
RUN cd /opt; chmod 755 ${ANACONDA} && \
  ./${ANACONDA} -b -p /opt/anaconda3 && \
  echo 'export PATH=/opt/anaconda3/bin:$PATH' >> /home/test/.bashrc && \
  echo 'export PATH"/opt/anaconda3/bin:$PATH' >> /root/.bashrc && \
  rm -f ${ANACONDA}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#RUN ${PIP} install jupyter matplotlib
#RUN ${PIP} install jupyter_http_over_ws
#RUN jupyter serverextension enable --py jupyter_http_over_ws
#
#RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
#RUN mkdir /.local && chmod a+rwx /.local
#RUN apt-get install -y --no-install-recommends wget
#
#WORKDIR /tf/tensorflow-tutorials
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
##COPY readme-for-jupyter.md README.md
#RUN apt-get autoremove -y && apt-get remove -y wget
#WORKDIR /tf
#EXPOSE 8888
#
#RUN ${PYTHON} -m ipykernel.kernelspec
#
#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
CMD ["bash"]
