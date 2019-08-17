package ai.h2o.hive.udf;

import java.util.Arrays;
import java.util.ArrayList;

import hex.genmodel.GenModel;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;


@UDFType(deterministic = true, stateful = false)
@Description(name = "MojoUDF", value = "mojoudf(*) - Returns a score for the given row",
        extended = "Example:\n" + "> SELECT mojoudf(*) FROM target_data;")
class MojoUDF extends GenericUDF {

    private PrimitiveObjectInspector[] inFieldOI;

    GenModel[] _models; //Array of Genmodel's

    ModelGroup _mg; //ModelGroup Object

    public void log(String s) {
        System.out.println("MojoUDF: " + s);
    } //Logging function for output to hive console

    //Override getDisplayString method of GenericUDF
    @Override
    public String getDisplayString(String[] args) {
        return "MojoUDF(" + Arrays.asList(_models[0].getNames()) + ").";
    }

    //Override configure method of GenericUDF to make it Mapreduce
    @Override
    public void configure(MapredContext context) {
        super.configure(context);
        context.toString();
    }

    //Override Initialize method of GenericUDF and use ObjectInspector class to evaluate type of input
    //Input in this case is primitives
    @Override
    public ObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {

        //Log the time it takes for initialize method
        long start = System.currentTimeMillis();
        log("Begin: Initialize()");

        //New ModelGroup object and add mojos to this list
        _mg = new ModelGroup();
        _mg.addMOJOsFromJARResource();
        if (args.length != _mg._groupPredictors.size()) {
            throw new UDFArgumentLengthException("Incorrect number of arguments." + " mojoUDF() requires: " +
                    Arrays.asList(_mg._groupPredictors.keySet()) + ", in the listed order. Received " + args.length + " arguments.");
        }

        //Check input types
        inFieldOI = new PrimitiveObjectInspector[args.length];
        PrimitiveObjectInspector.PrimitiveCategory pCat;
        for (int i = 0; i < args.length; i++) {
            if (args[i].getCategory() != ObjectInspector.Category.PRIMITIVE)
                throw new UDFArgumentException("mojoudf(...): Only takes primitive field types as parameters");
            pCat = ((PrimitiveObjectInspector) args[i]).getPrimitiveCategory();
            if (pCat != PrimitiveObjectInspector.PrimitiveCategory.STRING
                    && pCat != PrimitiveObjectInspector.PrimitiveCategory.DOUBLE
                    && pCat != PrimitiveObjectInspector.PrimitiveCategory.FLOAT
                    && pCat != PrimitiveObjectInspector.PrimitiveCategory.LONG
                    && pCat != PrimitiveObjectInspector.PrimitiveCategory.INT
                    && pCat != PrimitiveObjectInspector.PrimitiveCategory.SHORT)
                throw new UDFArgumentException("mojoudf(...): Cannot accept type: " + pCat.toString());
            inFieldOI[i] = (PrimitiveObjectInspector) args[i];
        }

        long end = System.currentTimeMillis() - start;
        log("End: initialize(), took: " + Long.toString(end));

        return ObjectInspectorFactory.getStandardListObjectInspector(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector));
    }

    //Override the evaluate method of GenericUDF
    @Override
    public Object evaluate(DeferredObject[] record) throws HiveException {
        if (record != null) {
            if (record.length == _mg._groupPredictors.size()) {
                double[] data = new double[record.length]; //initialize a double array as big as number of records

                for (int i = 0; i < record.length; i++) {
                    try {
                        //Check all the different datatypes which a record might be and convert it to a doubleValue
                        Object o = inFieldOI[i].getPrimitiveJavaObject(record[i].get());
                        if (o instanceof java.lang.String) {
                            data[i] = _mg.mapEnum(i, ((String) o).replace("\"", ""));
                            if (data[i] == -1)
                                throw new UDFArgumentException("mojoudf(...): The value " + (String) o + " is not a known category" +
                                        "for column " + _mg._groupIdxToColNames.get(i));
                        } else if (o instanceof Double) {
                            data[i] = ((Double) o).doubleValue();
                        } else if (o instanceof Float) {
                            data[i] = ((Float) o).doubleValue();
                        } else if (o instanceof Long) {
                            data[i] = ((Long) o).doubleValue();
                        } else if (o instanceof Integer) {
                            data[i] = ((Integer) o).doubleValue();
                        } else if (o instanceof Short) {
                            data[i] = ((Short) o).doubleValue();
                        } else if (o == null) {
                            return null;
                        } else {
                            throw new UDFArgumentException("mojoudf(...): Cannot accept type: " + o.getClass().toString()
                                    + " for argument # " + i + ".");
                        }
                    } catch (Throwable e) {
                        throw new UDFArgumentException("Unexpected exception on argument # " + i + "." + e.toString());
                    }
                }

                try {
                    //Call to scoreAll which does the actual scoring, save result to result_set
                    ArrayList<ArrayList<Double>> result_set = _mg.scoreAll(data);

                    return result_set;
                } catch (Throwable e) {
                    throw new UDFArgumentException("H2O predict function threw exception: " + e.toString());
                }
            } else {
                throw new UDFArgumentException("Incorrect number of arguments." + " mojoudf() requires: " +
                        Arrays.asList(_mg._groupPredictors.size() + ", in order. Received " + record.length + " arguments."));
            }
        } else {
            return null;
        }

    }
}