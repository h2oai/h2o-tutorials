package my.automl;

import ai.h2o.automl.*;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import hex.grid.Grid;
import water.Job;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static ai.h2o.automl.ModelingStep.ModelStep.DEFAULT_MODEL_TRAINING_WEIGHT;


public class MyGLMStepsProvider implements ModelingStepsProvider<MyGLMStepsProvider.GLMSteps> {

    public static class GLMSteps extends ModelingSteps {

        static abstract class GLMGridStep extends ModelingStep.GridStep<GLMModel> {

            GLMGridStep(String id, int weight, AutoML autoML) {
                super(Algo.GLM, id, weight, autoML);
            }

            GLMParameters prepareModelParameters() {
                GLMParameters glmParameters = new GLMParameters();
                glmParameters._lambda_search = true;
                glmParameters._family =
                        aml().getResponseColumn().isBinary() && !(aml().getResponseColumn().isNumeric()) ? GLMParameters.Family.binomial
                                : aml().getResponseColumn().isCategorical() ? GLMParameters.Family.multinomial
                                : GLMParameters.Family.gaussian;  // TODO: other continuous distributions!
                return glmParameters;
            }
        }

        private ModelingStep[] grids = new ModelingStep[] {
                new GLMGridStep("solvers", DEFAULT_MODEL_TRAINING_WEIGHT, aml()) {
                    @Override
                    protected Job<Grid> startJob() {
                        GLMParameters glmParameters = prepareModelParameters();
                        glmParameters._alpha = IntStream.rangeClosed(0, 10).asDoubleStream().map(i -> i / 10).toArray();
                        glmParameters._missing_values_handling = GLMParameters.MissingValuesHandling.MeanImputation;

                        Map<String, Object[]> searchParams = new HashMap<>();
                        searchParams.put("_standardize", new Boolean[] { true, false });
                        searchParams.put("_solver", GLMParameters.Solver.values());

                        return hyperparameterSearch(makeKey("MyGLM", false), glmParameters, searchParams);
                    }
                },
        };

        public GLMSteps(AutoML autoML) {
            super(autoML);
        }

        @Override
        protected ModelingStep[] getGrids() {
            return grids;
        }
    }

    @Override
    public String getName() {
        return "MyGLM";
    }

    @Override
    public GLMSteps newInstance(AutoML aml) {
        return new GLMSteps(aml);
    }
}

