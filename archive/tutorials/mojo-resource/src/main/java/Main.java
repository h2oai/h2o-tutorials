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