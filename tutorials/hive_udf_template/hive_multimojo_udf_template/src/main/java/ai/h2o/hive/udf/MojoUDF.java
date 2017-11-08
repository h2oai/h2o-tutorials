package ai.h2o.hive.udf;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.ArrayList;

import hex.genmodel.GenModel;
import hex.genmodel.easy.RowData;
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
import org.apache.hadoop.mapred.JobConf;


@UDFType(deterministic = true, stateful = false)
@Description(name = "MojoUDF", value = "mojoudf(*) - Returns a score for the given row",
        extended = "Example:\n" + "> SELECT mojoudf(*) FROM target_data;")
class MojoUDF extends GenericUDF {

    ModelGroup _mg;

    public void log (String s) {
        System.out.println("MojoUDF: " + s);
    }

    @Override
    public String getDisplayString(String[] args) {
        return "MojoUDF(" + Arrays.asList(_mg.getColNamesString()) + ").";
        //        this._predixors.addAll(Arrays.asList(Arrays.copyOfRange(m.getNames(), 0, m.getNames().length -1)));

    }

    @Override
    public void configure(MapredContext context) {
        super.configure(context);
        context.toString();
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {

        long start = System.currentTimeMillis();
        log("Begin: Initialize()");

        _mg = new ModelGroup();
        _mg.addMOJOsFromJARResource();

        if (args.length != _mg._predixors.size()) {
            throw new UDFArgumentLengthException("Incorrect number of arguments." + " MojoUDF() requires: " +
            Arrays.asList(_mg._predixors) + ", in the listed order. Received " + args.length + " arguments." );
        }

        long end = System.currentTimeMillis() - start;
        log("End: initialize(), took: " + Long.toString(end) + "milliseconds.");

        ObjectInspector doubleOI = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.DOUBLE);
        return ObjectInspectorFactory.getStandardListObjectInspector(doubleOI);
    }

    @Override
    public Object evaluate(DeferredObject[] record) throws HiveException {
        if (record != null) {
            if (record.length == _mg._predixors.size()) {
                try {
                    RowData row = new RowData();

                    for (int i = 0; i < record.length; i++) {
                        row.put(_mg._predixors.toArray()[i].toString(), record[i].get().toString());
                    }
                    return _mg.scoreAll(row);
                } catch (Throwable e) {
                    throw new UDFArgumentException("H2O predict function threw an exception: " + e.toString());
                }
            } else {
                throw new UDFArgumentException("Incorrect number of arguments." + " MojoUDF() requires: " +
                Arrays.asList(_mg._groupPredictors.size()) + ", in order. Received " + record.length +" arguments.");
            }
        } else {
            return null;
        }
    }
}
