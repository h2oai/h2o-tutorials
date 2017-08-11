package ai.h2o.hive.udf;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaDoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Test;
import junit.framework.Assert;

public class UDFExampleTest {
  @Test public void testUDFReturnsCorrectValues() throws HiveException {
    // set up the models we need
    ScoreDataUDF example = new ScoreDataUDF();
    //From the test data set: "AGEP", "COW", "SCHL", "MAR", "INDP", "RELP", "RAC1P", "SEX", "WKHP", "POBP", "LOG_CAPGAIN", "LOG_CAPLOSS"
    ObjectInspector AGEP_OI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
    ObjectInspector COW_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector SCHL_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector MAR_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector INDP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector RELP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector RAC1P_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector SEX_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector WKHP_OI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
    ObjectInspector POBP_OI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    ObjectInspector LOG_CAPGAIN_OI = PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
    ObjectInspector LOG_CAPLOSS_OI = PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
    JavaDoubleObjectInspector resultInspector = (JavaDoubleObjectInspector) example.initialize(new ObjectInspector[]{AGEP_OI,
            COW_OI, SCHL_OI, MAR_OI, INDP_OI, RELP_OI, RAC1P_OI, SEX_OI, WKHP_OI, POBP_OI, LOG_CAPGAIN_OI, LOG_CAPLOSS_OI });
    // test our results
    // Data from first line of test file: 48 "1" "21" "1" "7590" "0" "1" "2" 40 "1" 0.0 0.0
    Object result1 = example.evaluate(new DeferredObject[]{new DeferredJavaObject(48), new DeferredJavaObject("1"), // AGEP, COW 
            new DeferredJavaObject("21"), new DeferredJavaObject("1"), new DeferredJavaObject("7590"), // SCHL, MAR, INDP
            new DeferredJavaObject("0"), new DeferredJavaObject("1"),new DeferredJavaObject("2"), // RELP, RAC1P, SEX
            new DeferredJavaObject(40), new DeferredJavaObject("1"), new DeferredJavaObject(0.0), // WKHP, POBP, LOG_CAPGAIN
            new DeferredJavaObject(0.0)}); // LOG_CAPLOSS
    double tolerance = 1e-8;
    Assert.assertEquals(10.43278662820711D, resultInspector.get(result1), tolerance);
    // Wrong number of arguments

    try {
      example.evaluate(new DeferredObject[]{new DeferredJavaObject("0"), new DeferredJavaObject("21")});
      Assert.fail();
    } catch (UDFArgumentException expected) { Assert.assertTrue(true);}
    // Arguments are null
    Object result3 = example.evaluate(new DeferredObject[]{new DeferredJavaObject(null), new DeferredJavaObject(null), // AGEP, COW 
            new DeferredJavaObject(null), new DeferredJavaObject(null),new DeferredJavaObject(null), // SCHL, MAR, INDP 
            new DeferredJavaObject(null), new DeferredJavaObject(null), new DeferredJavaObject(null), //RELP, RAC1P, SEX
            new DeferredJavaObject(null), new DeferredJavaObject(null), new DeferredJavaObject(null), // WHKP, POPB, LOG_CAPGAIN
            new DeferredJavaObject(null)}); // LOG_CAPLOSS
    Assert.assertNull(result3);
  }
}
