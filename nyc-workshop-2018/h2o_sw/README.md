# Build instructions

### Creating the image

```
make
```

### Testing the image

```
make run
```

* http://localhost:8888
* password: h2o

go to h2o-3-hands-on directory.  open the ipython notebook there.  run all cells.

go to the sparkling-water-hands-on directory.  open the ipython notebook there.  run all cells.

### Exporting the image

```
make save
```

# Debugging

If there is any one thing likely to fail in the future it's version numbers of the packages in the conf files.  Specifically the file `conf/pyspark/00-pyspark-setup.py` contains code to import `python/lib/py4j-0.10.7-src.zip` which will break if that exact version number changes on the internet.


# Final lab info

AMI: ami-4ffcb737

instance: m4.4xlarge

required ports: 4040, 8888, 54321