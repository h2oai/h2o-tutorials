FROM ubuntu:16.04
MAINTAINER H2O.ai <ops@h2o.ai>

# Linux
RUN \
  apt-get -y update && \
  apt-get -y install \
    apt-transport-https \
    apt-utils \
    python-software-properties \
    software-properties-common

# RStudio
RUN \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
  echo "deb https://cran.rstudio.com/bin/linux/ubuntu xenial/" >> /etc/apt/sources.list

# Java8
RUN \
  add-apt-repository -y ppa:webupd8team/java

RUN \
  apt-get -y update

# Linux
RUN \
  apt-get -y install \
    cpio \
    curl \
    dirmngr \
    gdebi-core \
    git \
    net-tools \
    sudo \
    vim \
    wget \
    zip

# R
RUN \
  apt-get -y install \
    r-base \
    r-base-dev \
    r-cran-jsonlite \
    r-cran-rcurl && \
  mkdir -p /usr/local/lib/R/site-library && \
  chmod 777 /usr/local/lib/R/site-library

# Java 8
RUN \
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
  apt-get -y install oracle-java8-installer

# Log directory used by run.sh
RUN \
  mkdir /log && \
  chmod o+w /log

# RStudio
RUN \
  locale-gen en_US.UTF-8 && \
  update-locale LANG=en_US.UTF-8 && \
  wget https://download2.rstudio.org/rstudio-server-1.1.383-amd64.deb && \
  gdebi --non-interactive rstudio-server-1.1.383-amd64.deb && \
  echo "server-app-armor-enabled=0" >> /etc/rstudio/rserver.conf

# ----- USER H2O -----

# h2o user
RUN \
  useradd -ms /bin/bash h2o && \
  usermod -a -G sudo h2o && \
  echo "h2o:h2o" | chpasswd && \
  echo 'h2o ALL=NOPASSWD: ALL' >> /etc/sudoers

USER h2o
WORKDIR /home/h2o

# Miniconda
ENV MINICONDA_FILE=Miniconda3-4.3.11-Linux-x86_64.sh
RUN \
  wget https://repo.continuum.io/miniconda/${MINICONDA_FILE} && \
  bash ${MINICONDA_FILE} -b -p /home/h2o/Miniconda3 && \
  /home/h2o/Miniconda3/bin/conda create -y --name h2o python=3.5 anaconda && \
  /home/h2o/Miniconda3/envs/h2o/bin/jupyter notebook --generate-config && \
  sed -i "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = 'h2o'/" .jupyter/jupyter_notebook_config.py && \
  rm ${MINICONDA_FILE}

# H2O
ENV H2O_BRANCH_NAME=rel-weierstrass
ENV H2O_BUILD_NUMBER=7
ENV H2O_PROJECT_VERSION=3.14.0.${H2O_BUILD_NUMBER}
RUN \
  wget http://h2o-release.s3.amazonaws.com/h2o/${H2O_BRANCH_NAME}/${H2O_BUILD_NUMBER}/h2o-${H2O_PROJECT_VERSION}.zip && \
  unzip h2o-${H2O_PROJECT_VERSION}.zip && \
  rm h2o-${H2O_PROJECT_VERSION}.zip && \
  bash -c "source /home/h2o/Miniconda3/bin/activate h2o && pip install h2o-${H2O_PROJECT_VERSION}/python/h2o*.whl" && \
  R CMD INSTALL h2o-${H2O_PROJECT_VERSION}/R/h2o*.gz

USER root
WORKDIR /

# ----- RUN INFORMATION -----

# Entry point
COPY run.sh /run.sh

ENTRYPOINT ["./run.sh"]
  
EXPOSE 54321
EXPOSE 8888
EXPOSE 8787
