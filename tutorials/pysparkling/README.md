## Requirements

- Oracle Java 7+ ([USB](../../))
- [Spark 1.5.1](http://spark.apache.org/downloads.html) ([USB](../../Spark))
- [Sparkling Water 1.5.6](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-1.5/6/index.html) ([USB](../../SparklingWater))
- [Chicago Crime dataset](https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCrimes10k.csv) ([USB](../data/chicagoCrimes10k.csv))
- [Chicago Census dataset](https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCensus.csv) ([USB](../data/chicagoCensus.csv))
- [Chicago Weather dataset](https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoAllWeather.csv) ([USB](../data/chicagoAllWeather.csv))
- H2O python - to be installed ([USB](../../))
- H2O package ([USB](../../))
- Python 2.7	(pre-installed)
- Numpy 1.9.2 (pre-installed)
- $ pip install requests
- $ pip install tabulate



## Running the notebook
- Go to the Sparkling Water directory
- Build Sparkling Water ( creates python EGG ): `./gradlew build -x test`
- Run this line - `IPYTHON_OPTS="notebook" bin/pysparkling ` and locate the desired notebook file

