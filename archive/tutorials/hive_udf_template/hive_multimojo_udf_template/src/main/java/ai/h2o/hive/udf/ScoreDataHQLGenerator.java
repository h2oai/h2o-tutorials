package ai.h2o.hive.udf;

import hex.genmodel.GenModel;

import java.io.PrintWriter;

public class ScoreDataHQLGenerator {

    public static void main (String[] args) {
        ModelGroup _models = new ModelGroup();
        _models.addMOJOsFromJARResource();

        try {
            String[] mojo_names = _models.getMOJONames();

            System.out.println("-- model order (alphabetical)");
            for(int i = 0; i < _models.size(); i++) {
                GenModel m = _models.get(i);
                System.out.println("-- Name: " + mojo_names[i] + "\n"
                        + "--   Category: " + m.getModelCategory() + "\n"
                        + "--   Hive Select: scores["+i+"][0 - "+(m.getPredsSize()-1)+"]");
            }

            System.out.println();
            System.out.println("-- add jars");
            System.out.println("ADD JAR localjars/h2o-genmodel.jar;");
            System.out.println("ADD JAR target/ScoreData-1.0-SNAPSHOT.jar;");
            System.out.println();
            System.out.println("-- create fn definition");
            System.out.println("CREATE TEMPORARY FUNCTION fn AS \"ai.h2o.hive.udf.MojoUDF\";");
            System.out.println();
            System.out.println("-- column names reference");
            System.out.println("set hivevar:colnames=" + _models.getColNamesString() + ";");
            System.out.println();
            System.out.println("-- sample query, returns nested array");
            System.out.println("-- select fn(${colnames}) from adult_data_set");

            // Also write to file
            PrintWriter writer;
            try {
                String[] cols = _models.getColNamesString().split(",");
                writer = new PrintWriter("colnames.txt", "UTF-8");
                for (int i = 0; i < cols.length; i++) {
                    writer.println(cols[i]);
                }
                writer.close();
            } catch (Exception e) {}
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
