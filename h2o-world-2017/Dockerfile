FROM ubuntu:16.04
MAINTAINER H2O.ai <ops@h2o.ai>

RUN \
  apt-get -y update

RUN \
  apt-get -y install \
  curl \
  apt-utils \
  wget \
  cpio \
  net-tools \
  git \
  zip \
  vim \
  dirmngr

RUN \
  apt-get -y install r-base

RUN \
  apt-get -y install software-properties-common python-software-properties && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get -y update

RUN \
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
  apt-get -y install oracle-java8-installer

ENV H2O_BRANCH_NAME=rel-weierstrass
ENV H2O_BUILD_NUMBER=7
ENV H2O_PROJECT_VERSION=3.14.0.${H2O_BUILD_NUMBER}

RUN \
  wget http://h2o-release.s3.amazonaws.com/h2o/${H2O_BRANCH_NAME}/${H2O_BUILD_NUMBER}/h2o-${H2O_PROJECT_VERSION}.zip && \
  unzip h2o-${H2O_PROJECT_VERSION}.zip && \
  rm h2o-${H2O_PROJECT_VERSION}.zip

RUN \
  apt-get -y install r-cran-jsonlite && \
  apt-get -y install r-cran-rcurl && \
  mkdir -p /usr/local/lib/R/site-library && \
  chmod 777 /usr/local/lib/R/site-library

# ----- USER BEGIN -----

RUN \
  apt-get -y install sudo && \
  useradd -ms /bin/bash h2o && \
  usermod -a -G sudo h2o && \
  echo 'h2o ALL=NOPASSWD: ALL' >> /etc/sudoers
USER h2o
WORKDIR /home/h2o

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh
RUN bash Miniconda3-4.3.11-Linux-x86_64.sh -b -p /home/h2o/Miniconda3
ENV PATH=/home/h2o/Miniconda3/bin:$PATH

# Install Jupyter Notebook
RUN conda install -y jupyter notebook

RUN \
  pip install /h2o-${H2O_PROJECT_VERSION}/python/h2o*.whl

RUN \
  R CMD INSTALL /h2o-${H2O_PROJECT_VERSION}/R/h2o*.gz

USER root
WORKDIR /

# ----- USER END -----

RUN \
  locale-gen en_US.UTF-8 && \
  update-locale LANG=en_US.UTF-8

RUN \
  apt-get -y install gdebi-core && \
  wget https://download2.rstudio.org/rstudio-server-1.1.383-amd64.deb && \
  gdebi --non-interactive rstudio-server-1.1.383-amd64.deb

RUN \
  echo "server-app-armor-enabled=0" >> /etc/rstudio/rserver.conf

RUN echo "h2o:h2o" | chpasswd

# ----

USER h2o
WORKDIR /home/h2o

RUN \
  jupyter notebook --generate-config && \
  sed -i "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = 'h2o'/" .jupyter/jupyter_notebook_config.py

USER root
WORKDIR /

# ----

# Create a Log directory
RUN \
  mkdir /log && \
  chmod o+w /log

# Add shell wrapper
COPY run.sh /run.sh

ENTRYPOINT ["./run.sh"]
  
EXPOSE 54321
EXPOSE 8888
EXPOSE 8787
