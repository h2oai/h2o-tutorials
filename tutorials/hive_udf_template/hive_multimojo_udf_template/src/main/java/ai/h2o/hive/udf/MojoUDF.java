package ai.h2o.hive.udf;

import java.util.Arrays;

import hex.genmodel.easy.RowData;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@UDFType(deterministic = true, stateful = false)
@Description(name = "MojoUDF", value = "mojoudf(*) - Returns a score for the given row",
        extended = "Example:\n" + "> SELECT mojoudf(*) FROM target_data;")
class MojoUDF extends GenericUDF {
    ModelGroup _mg;

    @Override
    public String getDisplayString(String[] args) {
        return "MojoUDF(" + Arrays.asList(_mg.getColNamesString()) + ").";
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {
        _mg = new ModelGroup();
        _mg.addMOJOsFromJARResource();
        return ObjectInspectorFactory.getStandardListObjectInspector(
                ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector));
    }

    @Override
    public Object evaluate(DeferredObject[] record) throws HiveException {
        RowData row = new RowData();
        for (int i = 0; i < record.length; i++) {
            row.put(_mg._predixors.toArray()[i].toString(), record[i].get().toString());
        }
        return _mg.scoreAll(row);
    }
}
