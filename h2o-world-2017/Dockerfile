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
  /home/h2o/Miniconda3/bin/conda create -y --name pysparkling python=2.7 anaconda && \
  /home/h2o/Miniconda3/bin/conda create -y --name h2o python=3.5 anaconda && \
  /home/h2o/Miniconda3/envs/h2o/bin/jupyter notebook --generate-config && \
  sed -i "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = 'h2o'/" .jupyter/jupyter_notebook_config.py && \
  rm ${MINICONDA_FILE}

# H2O
ENV H2O_BRANCH_NAME=rel-wheeler
ENV H2O_BUILD_NUMBER=2
ENV H2O_PROJECT_VERSION=3.16.0.${H2O_BUILD_NUMBER}
ENV H2O_DIRECTORY=h2o-${H2O_PROJECT_VERSION}
RUN \
  wget http://h2o-release.s3.amazonaws.com/h2o/${H2O_BRANCH_NAME}/${H2O_BUILD_NUMBER}/h2o-${H2O_PROJECT_VERSION}.zip && \
  unzip ${H2O_DIRECTORY}.zip && \
  rm ${H2O_DIRECTORY}.zip && \
  bash -c "source /home/h2o/Miniconda3/bin/activate h2o && pip install ${H2O_DIRECTORY}/python/h2o*.whl" && \
  R CMD INSTALL ${H2O_DIRECTORY}/R/h2o*.gz

# Spark
ENV SPARK_VERSION=2.2.0
ENV SPARK_HADOOP_VERSION=2.7
ENV SPARK_DIRECTORY=spark-${SPARK_VERSION}-bin-hadoop${SPARK_HADOOP_VERSION}
ENV SPARK_HOME=/home/h2o/bin/spark
RUN \
  mkdir bin && \
  cd bin && \
  mkdir -p ${SPARK_HOME} && \
  wget http://mirrors.sonic.net/apache/spark/spark-${SPARK_VERSION}/${SPARK_DIRECTORY}.tgz && \
  tar zxvf ${SPARK_DIRECTORY}.tgz -C ${SPARK_HOME} --strip-components 1 && \
  rm ${SPARK_DIRECTORY}.tgz && \
  bash -c "source /home/h2o/Miniconda3/bin/activate pysparkling && pip install tabulate future colorama"


# Sparkling Water
ENV SPARKLING_WATER_BRANCH_NUMBER=2.2
ENV SPARKLING_WATER_BRANCH_NAME=rel-${SPARKLING_WATER_BRANCH_NUMBER}
ENV SPARKLING_WATER_BUILD_NUMBER=4
ENV SPARKLING_WATER_PROJECT_VERSION=${SPARKLING_WATER_BRANCH_NUMBER}.${SPARKLING_WATER_BUILD_NUMBER}
ENV SPARKLING_WATER_DIRECTORY=sparkling-water-${SPARKLING_WATER_PROJECT_VERSION}
RUN \
  cd bin && \
  wget http://h2o-release.s3.amazonaws.com/sparkling-water/${SPARKLING_WATER_BRANCH_NAME}/${SPARKLING_WATER_BUILD_NUMBER}/${SPARKLING_WATER_DIRECTORY}.zip && \
  unzip ${SPARKLING_WATER_DIRECTORY}.zip && \
  mv ${SPARKLING_WATER_DIRECTORY} sparkling-water && \
  rm ${SPARKLING_WATER_DIRECTORY}.zip && \
  /home/h2o/Miniconda3/envs/pysparkling/bin/ipython profile create pyspark 

COPY conf/pyspark/00-pyspark-setup.py /home/h2o/.ipython/profile_pyspark/startup/
COPY conf/pyspark/kernel.json /home/h2o/Miniconda3/envs/h2o/share/jupyter/kernels/pyspark/
ENV SPARKLING_WATER_HOME=/home/h2o/bin/sparkling-water

######################################################################
# ADD CONTENT FOR INDIVIDUAL HANDS-ON SESSIONS HERE
######################################################################

# Prologue
RUN \
  mkdir data && \
  mkdir Python && \
  mkdir R

# Megan's NLP demo
RUN \
  mkdir data/nlp && \
  mkdir Python/nlp && \
  wget -O data/nlp/w2v.hex 'https://s3.amazonaws.com/tomk/h2o-world/megan/w2v.hex' && \
  wget -O data/nlp/AmazonReviews.csv 'https://s3.amazonaws.com/tomk/h2o-world/megan/AmazonReviews.csv'
COPY nlp/AmazonReviews.ipynb Python/nlp
COPY nlp/stopwords.csv data/nlp


# Kuba's pysparkling demo
COPY pysparkling/AmazonFineFoodPipeline.ipynb  Python
COPY pysparkling/img/scenario.png Python/img/scenario.png
RUN mkdir data/pysparkling
 
RUN  wget -O data/pysparkling/AmazonReviews_Predictions.csv 'https://s3.amazonaws.com/tomk/h2o-world/kuba/AmazonReviews_Predictions.csv'
RUN  wget -O data/pysparkling/AmazonReviews_Train.csv 'https://s3.amazonaws.com/tomk/h2o-world/kuba/AmazonReviews_Train.csv'

# Erin's AutoML R & Python demos
RUN \
  mkdir data/automl && \
  mkdir R/automl && \
  mkdir Python/automl
COPY automl/R/automl_binary_classification_product_backorders.Rmd R/automl
COPY automl/R/automl_regression_powerplant_output.Rmd R/automl
COPY automl/Python/automl_binary_classification_product_backorders.ipynb Python/automl
COPY automl/Python/automl_regression_powerplant_output.ipynb Python/automl
COPY automl/data/product_backorders.csv data/automl
COPY automl/data/powerplant_output.csv data/automl

# Epilogue
USER root
RUN \
  chown -R h2o:h2o /home/h2o/data && \
  chown -R h2o:h2o /home/h2o/Python && \
  chown -R h2o:h2o /home/h2o/R

######################################################################

# ----- RUN INFORMATION -----

USER h2o
WORKDIR /home/h2o

# Entry point
COPY run.sh /run.sh

ENTRYPOINT ["/run.sh"]
  
EXPOSE 54321
EXPOSE 8888
EXPOSE 8787
EXPOSE 4040
