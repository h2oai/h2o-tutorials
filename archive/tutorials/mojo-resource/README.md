# Introduction

This tutorial describes how to package an H2O MOJO model as a JAR resource.

## Outline

1. Create and export an H2O Model from R
2. Create a new Java project
3. Setup main class & MOJO resource
4. Compile and package as a JAR

**Requirements**
1. H2O 3.10.4.2 or newer
2. Maven (https://maven.apache.org/install.html) required to run tutorial but in general any other build tool (gradle etc.) can be used

This repo represents the end state of the tutorial, and can be cloned, compiled, and run as follows:

    git glone https://github.com/h2oai/h2o-tutorials
    cd h2o-tutorials/mojo-resource
    mvn compile && mvn package
    java -cp target/mojo-resource-1.0-SNAPSHOT.jar:libs/* Main
    
The output should be as follows

    setosa: 0.009344361534466918
    versicolor: 0.9813250958541073
    virginica: 0.009330542611425827
    Prediction: versicolor

### 1. Create and export an H2O Model from R

Run the following R snippet (train_and_export_model.R) to create a basic GBM model on the Iris dataset and export it as an H2O MOJO.
Make sure to include get_genmodel_jar=TRUE so that the h2o-genmodel.jar dependency is exported as well. 
This small library may be required for building (unless using Maven respository) and will be required to run the final JAR. 

```r
library(h2o)
iris.hex <- as.h2o(iris)
iris.gbm <- h2o.gbm(y="Species", training_frame=iris.hex, model_id="irisgbm")
h2o.download_mojo(model=iris.gbm, path="/path/to/export/to", get_genmodel_jar = TRUE)
```
    
### 2. Create a new Java Project

Create a new Java project or clone this repo. This repo will use Maven but other options (gradle etc.) are available.

```
mkdir -p my-mojo-project/src/main/java
cd my-mojo-project
```

Create a pom.xml (for Maven) in the project root and make sure to include the h2o-genmodel dependency

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.nickkarpov</groupId>
    <artifactId>mojo-resource</artifactId>
    <version>1.0-SNAPSHOT</version>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.0.2</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addClasspath>true</addClasspath>
                            <classpathPrefix>lib/</classpathPrefix>
                            <mainClass>Main</mainClass>
                        </manifest>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>

    <dependencies>
        <dependency>
            <groupId>ai.h2o</groupId>
            <artifactId>h2o-genmodel</artifactId>
            <version>3.10.4.2</version>
        </dependency>
    </dependencies>
</project>
```

### 3. Add MOJO as resource and write main class

Add a resources directory and copy the exported MOJO from (1) to it

```
# from project root
mkdir src/main/resources
cp /path/to/irisgbm.zip src/main/resources
```

Setup the main class to do scoring. The following snippet (Main.java) loads the MOJO and scores a single row using the EasyPrediction API. 
*http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html to learn more about the POJO, MOJO, and EasyPrediction API*

```java
import java.net.URL;
import hex.genmodel.*;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.prediction.MultinomialModelPrediction;

public class Main {
    public static void main(String[] args) throws Exception {
        URL mojoURL = Main.class.getResource("irisgbm.zip");
        MojoReaderBackend reader = MojoReaderBackendFactory.createReaderBackend(mojoURL, MojoReaderBackendFactory.CachingStrategy.MEMORY);
        MojoModel model = ModelMojoReader.readFrom(reader);
        EasyPredictModelWrapper modelWrapper = new EasyPredictModelWrapper(model);
        RowData testRow = new RowData();
        for (int i = 0; i < args.length; i++)
            testRow.put("C"+i, Double.valueOf(args[i]));
        MultinomialModelPrediction prediction = (MultinomialModelPrediction) modelWrapper.predict(testRow);
        for (int i = 0; i < prediction.classProbabilities.length; i++)
            System.out.println(modelWrapper.getResponseDomainValues()[i] + ": "+ prediction.classProbabilities[i]);
        System.out.println("Prediction: " + prediction.label);
    }
}
```

### 4. Compile and package as a JAR

Use Maven to compile & package the project and run to see the output. Don't forget to include h2o-genmodel.jar (it is captured by libs/* in the example below)

```
# from project root
mvn compile && mvn package
java -cp target/mojo-resource-1.0-SNAPSHOT.jar:libs/* Main
# output will be 
setosa: 0.009344361534466918
versicolor: 0.9813250958541073
virginica: 0.009330542611425827
Prediction: versicolor
```