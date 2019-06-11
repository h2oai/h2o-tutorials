import h2o
from h2o.estimators import H2OXGBoostEstimator
import pandas

h2o.init()
train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv.zip", destination_frame='train')
train['CAPSULE'] = train['CAPSULE'].asfactor()

dart_model = H2OXGBoostEstimator(
    nfolds=5,
    ntrees=25,
    max_depth=5,
    learn_rate=0.1,
    min_rows=20,
    booster="dart"
)
dart_model.train(
    y="CAPSULE",     
    x=["AGE", "RACE", "PSA", "GLEASON"],
    training_frame=train
)

dart_file = dart_model.download_mojo(path="./model_dart.zip")
print("Dart model saved to " + dart_file)

lin_model = H2OXGBoostEstimator(
    nfolds=5,
    ntrees=25,
    max_depth=5,
    learn_rate=0.1,
    min_rows=20
)
lin_model.train(
    y="CAPSULE",     
    x=["AGE", "RACE", "PSA", "GLEASON"],
    training_frame=train
)

lin_file = lin_model.download_mojo(path="./model_linear.zip")
print("Model saved to " + lin_file)
