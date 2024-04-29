package my.automl;

import ai.h2o.automl.*;
import hex.grid.Grid;
import hex.tree.drf.DRFModel;
import hex.tree.drf.DRFModel.DRFParameters;
import water.Job;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static ai.h2o.automl.ModelingStep.ModelStep.DEFAULT_MODEL_TRAINING_WEIGHT;


public class MyDRFStepsProvider implements ModelingStepsProvider<MyDRFStepsProvider.DRFSteps> {

  public static class DRFSteps extends ModelingSteps {

    static final String NAME = Algo.DRF.name();
    static abstract class DRFGridStep extends ModelingStep.GridStep<DRFModel> {

      DRFGridStep(String id, AutoML autoML) {
        super(NAME, Algo.DRF, id, autoML);
      }

      public DRFParameters prepareModelParameters() {
        DRFParameters drfParameters = new DRFParameters();
        drfParameters._sample_rate = 0.8;
        drfParameters._col_sample_rate_per_tree = 0.8;
        drfParameters._col_sample_rate_change_per_level = 0.9;
        return drfParameters;
      }
    }

    private ModelingStep[] grids = new ModelingStep[]{
            new DRFGridStep("grid_1", aml()) {
              @Override
              public Map<String, Object[]> prepareSearchParameters() {
                Map<String, Object[]> searchParams = new HashMap<>();
                searchParams.put("_ntrees", IntStream.rangeClosed(5, 1000).filter(i -> i % 50 == 0).boxed().toArray());
                searchParams.put("_nbins", IntStream.of(5, 10, 15, 20, 30).boxed().toArray());
                searchParams.put("_max_depth", IntStream.rangeClosed(3, 20).boxed().toArray());
                searchParams.put("_min_rows", IntStream.of(3, 5, 10, 20, 50, 80, 100).boxed().toArray());
                return searchParams;
              }
            },
    };

    public DRFSteps(AutoML autoML) {
      super(autoML);
    }

    @Override
    protected ModelingStep[] getGrids() {
      return grids;
    }

    @Override
    public String getProvider() {
      return NAME;
    }
  }

  @Override
  public String getName() {
    return "MyDRF";
  }

  @Override
  public DRFSteps newInstance(AutoML aml) {
    return new DRFSteps(aml);
  }
}

