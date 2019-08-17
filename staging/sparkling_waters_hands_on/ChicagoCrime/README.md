# Hands On Sparkling Water Session

## Introduction

This session will cover a hands on training of H2O's Sparkling Water in Scala and Python.  Attendees do not need to follow the workshop for both Scala and Python, they can choose which language they are most comfortable with and stick with that.


## Sparkling Water Installation Instructions

## Mac Users

### Scala Users


Prerequisites for using Sparkling Water is: 

* JDK version 7+
* Spark release: 2.3.0 - [Spark Download Page](https://spark.apache.org/downloads.html)

#### Installation Instructions:


1. Download [Sparkling Water 2.3.5](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-2.3/5/index.html)
2. In terminal, export the following environment variables:

  ```bash
  export SPARK_HOME="/path/to/spark/installation" 
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  export MASTER="local[*]" 
  ```

3. From your terminal, run:

  ```bash
  cd ~/Downloads
  unzip sparkling-water-2.3.5.zip
  cd sparkling-water-2.3.5
  bin/sparkling-shell --conf "spark.executor.memory=1g"
  ```

4. Check to see that we can successfully start H2O cloud inside the Spark cluster.

  ```scala
  import org.apache.spark.h2o._
  val h2oContext = H2OContext.getOrCreate(sc) 
  import h2oContext._ 
  ```

5. If installation was successful, you can stop H2O and Spark services with the following command in the Spark shell:
  
  ```scala
  h2oContext.stop(stopSparkContext = true)
  ```

### Python Users

Prerequisites for using PySparkling is: 

* JDK version 7+
* Python 2.7
* Spark release: 2.3.0 - [Spark Download Page](https://spark.apache.org/downloads.html)

#### Installation Instructions:


1. In your terminal, install python dependencies:

  ```bash
  pip install requests
  pip install tabulate
  pip install six
  pip install future
  pip install ipython
  ```
  
  **Troubleshooting Tip**: If you see that you are missing dependencies for ipython, try the following: 
  
  ```
  pip install --upgrade setuptools pip
  pip uninstall ipython
  pip install ipython
  ```

2. Close the terminal used to do pip install.

3. Download [Sparkling Water 2.3.5](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-2.3/5/index.html)

4. From your terminal, run the following commands. This will take you to the sparkling water folder.

  ```bash
  cd ~/Downloads
  unzip sparkling-water-2.3.5.zip
  cd sparkling-water-2.3.5
  ```

5. In terminal, export the following environment variables:

  ```bash
  export SPARK_HOME="/path/to/spark/installation" 
  export SPARK_LOCAL_IP="127.0.0.1"
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  export MASTER="local[*]" 
  ```

6. From the top of the sparkling water directory, run the PySparkling shell
  
  ```bash
  # Run the pysparkling shell
  bin/pysparkling
  ```

7. In the python shell, check to see that we can successfully start H2O cloud inside the Spark cluster.

  ```python
  from pysparkling import *
  import h2o
  hc = H2OContext.getOrCreate(sc)
  ```

8. If installation was successful, you can stop H2O and Spark services with the following command in python:
  
  ```python
  h2o.shutdown(prompt = False)
  sc.stop()
  ```

## Windows Users

The Windows environments require several additional steps to make Spark and later Sparkling Water working.

### Scala Users


Prerequisites for using Sparkling Water is: 

* JDK version 7+
* Spark release: 2.3.0 - [Spark Download Page](https://spark.apache.org/downloads.html)

#### Installation Instructions:


1. Download [Sparkling Water 2.3.5](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-2.3/5/index.html)

2. Unzip the sparkling water folder.

3. Set the following environment variables:

  ```bash
  SET SPARK_HOME="/path/to/spark/installation" 
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  SET MASTER="local[*]" 
  ```

4. From https://github.com/steveloughran/winutils, download `winutils.exe` for Hadoop version which is referenced by your Spark distribution (for example, for `spark-2.3.0-bin-hadoop2.7.tgz` you need `wintutils.exe` for [hadoop2.7](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe?raw=true)).

5. Put `winutils.exe` into a new directory `%SPARK_HOME%\hadoop\bin` and set:
  ```
  SET HADOOP_HOME=%SPARK_HOME%\hadoop
  ```
  
6. Create a new file `%SPARK_HOME%\hadoop\conf\hive-site.xml` which setup default Hive scratch dir. The best location is a writable temporary directory, for example `%TEMP%\hive`:
  ```
  <configuration>
    <property>
      <name>hive.exec.scratchdir</name>
      <value>PUT HERE LOCATION OF TEMP FOLDER</value>
      <description>Scratch space for Hive jobs</description>
    </property>
  </configuration>
  ```
  
  > Note: you can also use Hive default scratch directory which is `\tmp\hive`. In this case, you need to create directory manually and call `winutils.exe chmod 777 \tmp\hive` to setup right permissions.
  
7. Set `HADOOP_CONF_DIR` property
  ```
  SET HADOOP_CONF_DIR=%SPARK_HOME%\hadoop\conf
  ```

8. Go to the top of the sparkling water folder.  From the command line, run:

  ```bash
  bin\sparkling-shell --conf "spark.executor.memory=1g"
  ```
  
9. Check to see that we can successfully start H2O cloud inside the Spark cluster.

  ```scala
  import org.apache.spark.h2o._
  val h2oContext = H2OContext.getOrCreate(sc) 
  import h2oContext._ 
  ```
10. If installation was successful, you can stop H2O and Spark services with the following command in the Spark shell:
  
  ```scala
  h2oContext.stop(stopSparkContext = true)
  ```

### Python Users

Prerequisites for using PySparkling is: 

* JDK version 7+
* Python 2.7
* Spark release: 2.3.0 - [Spark Download Page](https://spark.apache.org/downloads.html)

#### Installation Instructions:

1. In your terminal, install python dependencies:

  ```bash
  pip install requests
  pip install tabulate
  pip install six
  pip install future
  pip install ipython
  ```
  
  **Troubleshooting Tip**: If you see that you are missing dependencies for ipython, try the following: 
  
  ```
  pip install --upgrade setuptools pip
  pip uninstall ipython
  pip install ipython
  ```

2. Close the terminal used to do pip install.

3. Download [Sparkling Water 2.3.5](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-2.3/5/index.html).

4. Unzip the sparkling water folder.

5. Set the following environment variables:

  ```bash
  SET SPARK_HOME="/path/to/spark/installation" 
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  SET MASTER="local[*]" 
  ```

6. From https://github.com/steveloughran/winutils, download `winutils.exe` for Hadoop version which is referenced by your Spark distribution (for example, for `spark-2.3.0-bin-hadoop2.7.tgz` you need `wintutils.exe` for [hadoop2.7](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe?raw=true)).

7. Put `winutils.exe` into a new directory `%SPARK_HOME%\hadoop\bin` and set:

  ```
  SET HADOOP_HOME=%SPARK_HOME%\hadoop
  ```
  
8. Create a new file `%SPARK_HOME%\hadoop\conf\hive-site.xml` which setup default Hive scratch dir. The best location is a writable temporary directory, for example `%TEMP%\hive`:

  ```
  <configuration>
    <property>
      <name>hive.exec.scratchdir</name>
      <value>PUT HERE LOCATION OF TEMP FOLDER</value>
      <description>Scratch space for Hive jobs</description>
    </property>
  </configuration>
  ```
  
  > Note: you can also use Hive default scratch directory which is `\tmp\hive`. In this case, you need to create directory manually and call `winutils.exe chmod 777 \tmp\hive` to setup right permissions.
  
9. Set `HADOOP_CONF_DIR` property
  ```
  SET HADOOP_CONF_DIR=%SPARK_HOME%\hadoop\conf
  ```

10. Go to the top of the sparkling water directory.  From the command line, run the PySparkling shell
  
  ```bash
  # Run the pysparkling shell
  bin\pysparkling
  ```

11. In the python shell, check to see that we can successfully start H2O cloud inside the Spark cluster.

  ```python
  from pysparkling import *
  import h2o
  hc = H2OContext.getOrCreate(sc)
  ```

12. If installation was successful, you can stop H2O and Spark services with the following command in python:
  
  ```python
  h2o.shutdown(prompt = False)
  sc.stop()
  ```