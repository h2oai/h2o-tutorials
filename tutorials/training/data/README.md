## POJO Deployment for CSV file

```
mkdir tmpdir
cd tmpdir
curl http://127.0.0.1:54321/3/h2o-genmodel.jar > h2o-genmodel.jar
curl http://127.0.0.1:54321/3/Models.java/GBM_BadLoan > GBM_BadLoan.java
javac -cp h2o-genmodel.jar -J-Xmx2g -J-XX:MaxPermSize=128m GBM_BadLoan.java
java -ea -cp ./h2o-genmodel.jar:. -Xmx4g -XX:MaxPermSize=256m -XX:ReservedCodeCacheSize=256m hex.genmodel.tools.PredictCsv --header --model GBM_BadLoan --input ../test.csv --output ../predictions.csv
```


