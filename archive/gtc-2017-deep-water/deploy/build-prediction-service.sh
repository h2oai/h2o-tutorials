curl -m 500 -X POST \
--form mojo=@model_mnist_lenet_mx.zip \
--form jar=@h2o-genmodel.jar \
--form deepwater=@deepwater-all.jar \
localhost:55000/makewar > model_mnist_lenet_mx.war
