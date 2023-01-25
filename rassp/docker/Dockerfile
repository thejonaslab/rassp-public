# FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# FROM continuumio/anaconda3
# NOTE: --platform=linux/amd64 is needed if trying to build Docker image on M1
# https://stackoverflow.com/questions/71040681/qemu-x86-64-could-not-open-lib64-ld-linux-x86-64-so-2-no-such-file-or-direc
# FROM --platform=linux/amd64 ubuntu:18.04
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget git vim build-essential cmake emacs tmux
RUN useradd -ms /bin/bash build

# - /home/
#   - /home/build/
#     - environment.yml
#     - setup.py
#     - mambaforge.sh
#     - mamba
#     - /home/build/rassp
USER build
ENV HOME /home/build
ENV RASSP $HOME/rassp
WORKDIR /home/build/

# RUN ["/bin/bash", "-c", "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh"]
RUN ["/bin/bash", "-c", "wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O $HOME/mambaforge.sh"]

# RUN chmod 0755 $HOME/miniconda.sh

RUN ["/bin/bash", "-c", "bash $HOME/mambaforge.sh -b -p $HOME/mamba"]
ENV PATH="$HOME/mamba/bin:$PATH"
RUN rm $HOME/mambaforge.sh
# RUN ["/bin/bash", "-c", "bash"]

### Create conda environment
# COPY --chown=build "environment.yml" $HOME/environment.yml
ENV CONDA_ENV=mspred

# NOTE: this step takes a long time...
# RUN ["/bin/bash", "-c", "conda config --set channel_priority flexible"]
# RUN ["/bin/bash", "-c", "conda env create -n $CONDA_ENV -f /home/build/environment.yml -v"]
RUN ["/bin/bash", "-c", "mamba create -c conda-forge -c pytorch -c nvidia -n $CONDA_ENV python=3.7 numpy=1.21 numba pandas matplotlib pytorch torchvision torchaudio pytorch-cuda=11.6 cython numba scikit-learn rdkit cython -v"]

### SETUP
RUN mkdir $RASSP

# add models and source code
RUN mkdir -p $RASSP/models
COPY --chown=build setup.py $HOME
COPY --chown=build models/* $RASSP/models/
COPY --chown=build util.py inference_script.py run_rassp.py netutil.py $RASSP/
# COPY --chown=build util.py netutil.py $RASSP/
COPY --chown=build model/ $RASSP/model/
COPY --chown=build featurize/ $RASSP/featurize/
COPY --chown=build dataset/ $RASSP/dataset/
COPY --chown=build msutil/ $RASSP/msutil/
# COPY --chown=build nist17_hconfspcl_48a_512_uf4096-all.sparse.parquet $RASSP/

RUN conda init
RUN echo "source activate $CONDA_ENV" > ~/.bashrc

# these are necessary to get click working see
# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL C.UTF-8
ENV LANG=C.UTF-8
WORKDIR $RASSP

ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

# install rassp
WORKDIR $HOME

# pip install other packages
RUN ["/bin/bash", "-c", "source activate $CONDA_ENV; python -m pip install scipy python-igraph ruffus tqdm boto3 networkx graphviz tensorflow-gpu tensorboardX git+https://github.com/thejonaslab/tinygraph.git diskcache PyYAML natsort pyarrow seaborn SQLAlchemy click Pathlib"]

# pip install `rassp` as local module
RUN ["/bin/bash", "-c", "source activate $CONDA_ENV; python -m pip install -e ."]

# precompile all the .pyc and .pyx files in msutil
WORKDIR $RASSP
RUN ["/bin/bash", "-c", "source activate $CONDA_ENV; python -c 'import msutil'"]

# RUN echo "source activate $CONDA_ENV; python inference_script.py \$@" > run_command.sh
# RUN echo "source activate $CONDA_ENV; python \$@" > run_command.sh
# RUN chmod 0755 run_command.sh
# ENTRYPOINT ["/bin/bash", "run_command.sh"]
# ENTRYPOINT ["bash", "run_command.sh"]
