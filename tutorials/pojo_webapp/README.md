# H2O generated POJO model WebApp Example

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
$ curl "http://localhost:8080/pojo_webapp/predict?pclass=1&sex=male&age=25&fare=1"
{
  labelIndex : 0,
  label : "0",
  "classProbabilities" : [
    0.684132522471987,
    0.3158674775280131
  ]
}
```

```
$ curl "http://localhost:8080/pojo_webapp/predict?pclass=3&sex=male&age=40&fare=1000"
{
  labelIndex : 0,
  label : "0",
  "classProbabilities" : [
    0.8946904250568489,
    0.10530957494315105
  ]
}
```

```
$ curl "http://localhost:8080/pojo_webapp/predict?pclass=3&sex=junk&age=40&fare=1000"
[... HTTP error response simplified below ...]
Error 406 Unknown categorical level (sex,junk)
```

## References

The starting point for this example was taken from the gradle distribution.  It shows how to do basic war and jetty plugin operations.

1. <https://services.gradle.org/distributions/gradle-2.7-all.zip>
2. unzip gradle-2.7-all
3. cd gradle-2.7/samples/webApplication/customized

