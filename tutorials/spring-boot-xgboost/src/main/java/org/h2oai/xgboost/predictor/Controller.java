package org.h2oai.xgboost.predictor;


import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.AbstractPrediction;
import hex.genmodel.easy.prediction.BinomialModelPrediction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class Controller {

    @Autowired
    EasyPredictModelWrapper model;

    @Autowired
    EasyPredictModelWrapper linearModel;

    @RequestMapping("/")
    public String index(@RequestParam(name = "age", defaultValue = "50") String age) throws PredictException {
        return predict(model, age);
    }

    private String predict(EasyPredictModelWrapper model, String age) throws PredictException {
        RowData row = new RowData();
        row.put("AGE", age);
        row.put("RACE", "2");
        row.put("DCAPS", "2");
        row.put("VOL", "0");
        row.put("GLEASON", "6");
        StringBuilder out = new StringBuilder();
        AbstractPrediction p = model.predict(row);
        if (p instanceof BinomialModelPrediction) {
            BinomialModelPrediction bp = (BinomialModelPrediction) p;
            out.append("Has penetrated the prostatic capsule (1=yes; 0=no): ").append(bp.label).append("\n");
            out.append("Class probabilities: ");
            for (int i = 0; i < bp.classProbabilities.length; i++) {
                if (i > 0) {
                    out.append(",");
                }
                out.append(bp.classProbabilities[i]);
            }
        } else {
            out.append("Unexpected type of prediction: ").append(p.toString());
        }
        return "<pre>" + out.toString() + "</pre>";
    }

}
