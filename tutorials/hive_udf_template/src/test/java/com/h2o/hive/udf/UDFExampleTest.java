package com.h2o.hive.udf;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaFloatObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Test;
import junit.framework.Assert;

public class UDFExampleTest {
  @Test public void testUDFReturnsCorrectValues() throws HiveException {
    // set up the models we need
    ScoreDataUDF example = new ScoreDataUDF();
    //From the test data set: "RELP", "SCHL", "COW", "MAR", "INDP", "RAC1P", "SEX", "POBP", "AGEP", "WKHP", "LOG_CAPGAIN", "LOG_CAPLOSS"
    ObjectInspector RELP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector SCHL_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector COW_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector MAR_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector INDP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector RAC1P_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector SEX_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector POBP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector AGEP_OI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
    ObjectInspector WKHP_OI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
    ObjectInspector LOG_CAPGAIN_OI = PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
    ObjectInspector LOG_CAPLOSS_OI = PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
    JavaFloatObjectInspector resultInspector = (JavaFloatObjectInspector) example.initialize(new ObjectInspector[]{RELP_OI,
            SCHL_OI, COW_OI, MAR_OI, INDP_OI, RAC1P_OI, SEX_OI, POBP_OI, AGEP_OI, WKHP_OI, LOG_CAPGAIN_OI, LOG_CAPLOSS_OI });
    // test our results
    // Data from first line of test file: "0" "21" "1" "1" "7590" "1" "2" "1" 48 40 0.0 0.0
    Object result1 = example.evaluate(new DeferredObject[]{new DeferredJavaObject("0"), new DeferredJavaObject("21"), // RELP, SCHL
            new DeferredJavaObject("1"), new DeferredJavaObject("1"), new DeferredJavaObject("7590"), // COW, MAR, INDP
            new DeferredJavaObject("1"), new DeferredJavaObject("2"),new DeferredJavaObject("1"), // RAC1P, SEX, POBP
            new DeferredJavaObject(48), new DeferredJavaObject(40), new DeferredJavaObject(0.0), // AGEP, WKHP, LOG_CAPGAIN
            new DeferredJavaObject(0.0)}); // LOG_CAPLOSS
    Assert.assertEquals(10.476669311523438, resultInspector.get(result1), Math.ulp(resultInspector.get(result1)));
    // Wrong number of arguments
    Object result2 = example.evaluate(new DeferredObject[]{new DeferredJavaObject("0"), new DeferredJavaObject("21")});
    Assert.assertNull(result2);
    // Arguments are null
    Object result3 = example.evaluate(new DeferredObject[]{new DeferredJavaObject(null), new DeferredJavaObject(null), // RELP, SCHL
            new DeferredJavaObject(null), new DeferredJavaObject(null),new DeferredJavaObject(null), // RAC1P, SEX, POBP
            new DeferredJavaObject(null), new DeferredJavaObject(null), new DeferredJavaObject(null), // AGEP, WKHP, LOG_CAPGAIN
            new DeferredJavaObject(null)}); // LOG_CAPLOSS
    Assert.assertNull(result3);
  }
}
