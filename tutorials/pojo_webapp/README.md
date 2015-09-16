# H2O generated model POJO WebApp Example

This example shows a generated Java POJO being called using a REST API from a JavaScript Web application.

## Pieces at work

### Processes

(Front-end)   

1.  Web browser

(Back-end)   

1.  Jetty servlet container

> Note:  Not to be confused with the H2O embedded web port (default 54321) which is also powered by Jetty.

## Files

(Front-end)   

1.  JavaScript program

(Back-end)   

1.  POJO java code
1.  genmodel.jar
1.  Servlet code
1.  web.xml deployment descriptor


## Steps to run

##### Step 1: Create the gradle wrapper to get a stable version of gradle.

```
$ gradle wrapper
```

##### Step 2: Install H2O's R package if you don't have it yet.

<http://h2o-release.s3.amazonaws.com/h2o/rel-slater/1/index.html#R>

##### Step 3: Build the project (Unix only for now).

```
$ make clean
$ make
```

##### Step 4: Deploy the .war file in a Jetty servlet container.

```
$ ./gradlew jettyRunWar
```

##### Step 5: Visit the webapp in a browser.

<http://localhost:8080/pojo_webapp>



## Underneath the hood

Make a prediction with curl and get a JSON response.

```
$ curl -v "http://localhost:8080/pojo_webapp/predict?feature1=blah&feature2=blah?feature3=blah"
{
    "response" : "blah"
}
```

## References

The starting point for this example was taken from the gradle distribution.  It shows how to do basic war and jetty plugin operations.

1. <https://services.gradle.org/distributions/gradle-2.7-all.zip>
2. unzip gradle-2.7-all
3. cd gradle-2.7/samples/webApplication/customized

