package org.h2oai.xgboost.predictor;


import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import java.io.IOException;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Bean
    public EasyPredictModelWrapper model() throws IOException {
        return new EasyPredictModelWrapper(
            MojoModel.load("model.zip")
        );
    }

}
