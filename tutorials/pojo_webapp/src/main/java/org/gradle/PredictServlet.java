package org.gradle;

import java.io.*;
import java.util.Map;

import javax.servlet.http.*;
import javax.servlet.*;

import hex.genmodel.easy.prediction.BinomialModelPrediction;
import hex.genmodel.easy.*;

public class PredictServlet extends HttpServlet {
  private BinomialModelPrediction predict (RowData row) throws Exception {
    // Potential improvement for performance by not instantiating a new model every time.
    MyModel rawModel = new MyModel();
    EasyPredictModelWrapper model = new EasyPredictModelWrapper(rawModel);

    // Make the prediction.
    return model.predictBinomial(row);
  }

  public void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    RowData row = new RowData();

    // Fill RowData with parameter information from the HTTP request.
    Map<String, String[]> parameterMap;
    parameterMap = request.getParameterMap();
    System.out.println();
    for (Map.Entry<String, String[]> entry : parameterMap.entrySet()) {
      String key = entry.getKey();
      String[] values = entry.getValue();
      for (String value : values) {
        System.out.println("Key: " + key + " Value: " + value);
        if (value.length() > 0) {
          row.put(key, value);
        }
      }
    }

    try {
      BinomialModelPrediction p = predict(row);
      System.out.println("p[1]: " + p.classProbabilities[1]);

      // Build a JSON response for the prediction.
      StringBuilder sb = new StringBuilder();
      sb.append("{\n");
      sb.append("  \"labelIndex\" : ").append(p.labelIndex).append(",\n");
      sb.append("  \"label\" : \"").append(p.label).append("\",\n");
      sb.append("  \"classProbabilities\" : ").append("[\n");
      for (int i = 0; i < p.classProbabilities.length; i++) {
        double d = p.classProbabilities[i];
        if (Double.isNaN(d)) {
          throw new RuntimeException("Probability is NaN");
        }
        else if (Double.isInfinite(d)) {
          throw new RuntimeException("Probability is infinite");
        }

        sb.append("    ").append(d);
        if (i != (p.classProbabilities.length - 1)) {
          sb.append(",");
        }
        sb.append("\n");
      }
      sb.append("  ]\n");
      sb.append("}\n");

      // Emit the prediction to the servlet response.
      response.getWriter().write(sb.toString());
      response.setStatus(HttpServletResponse.SC_OK);
    }
    catch (Exception e) {
      // Prediction failed.
      System.out.println(e.getMessage());
      response.sendError(HttpServletResponse.SC_NOT_ACCEPTABLE, e.getMessage());
    }
  }

  /**
   * A few simple test cases.
   */
  public static void main(String[] args) {
    {
      RowData row = new RowData();
      row.put("fare", "1");
      row.put("sex", "female");
      row.put("pclass", "1");
      row.put("age", "25");

      PredictServlet s = new PredictServlet();
      try {
        BinomialModelPrediction p = s.predict(row);
        System.out.println(p.classProbabilities[1]);
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    {
      RowData row = new RowData();
      row.put("fare", "10");
      row.put("sex", "male");
      row.put("pclass", "3");
      row.put("age", "75");

      PredictServlet s = new PredictServlet();
      try {
        BinomialModelPrediction p = s.predict(row);
        System.out.println(p.classProbabilities[1]);
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
