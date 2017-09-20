package ai.h2o.hive.udf;

import java.io.File;
import java.io.IOException;
import java.net.MulticastSocket;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import hex.genmodel.*;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.MultinomialModelPrediction;
import hex.genmodel.easy.prediction.RegressionModelPrediction;
import hex.genmodel.easy.prediction.BinomialModelPrediction;

public class ModelGroup extends ArrayList<GenModel> {

    LinkedHashSet<String> _predixors;

    class Predictor {
        public int index;
        public String[] domains;
        public Predictor(int index, String[] domains) {
            this.index = index;
            this.domains = domains;
        }
        public String toString() {
            if (this.domains != null)
                return Integer.toString(this.index) + " " + Arrays.asList(this.domains);
            else
                return Integer.toString(this.index) + "numerical";
        }
    }

    public LinkedHashMap<String, Predictor> _groupPredictors;
    public ArrayList<String> _groupIdxToColNames;

    public ModelGroup() {
        this._predixors = new LinkedHashSet<String>();
        this._groupPredictors = new LinkedHashMap<String, Predictor>();
        this._groupIdxToColNames = new ArrayList<String>();
    }

    public String[] getMOJONames() throws Exception {
        String[] mojo_names = getResourceListing(ModelGroup.class, "models/");
        ArrayList<String> tmp = new ArrayList<String>();
        for(String s : mojo_names) {
            if(s.endsWith(".zip")) tmp.add(s);
        }

        Collections.sort(tmp, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                return s1.compareToIgnoreCase(s2);
            }
        });

        return tmp.toArray(new String[tmp.size()]);
    }

    public void addMOJOsFromJARResource() {
        try {
            String[] mojo_names = this.getMOJONames();
            for (int i = 0; i < mojo_names.length; i++) {
                MojoReaderBackend reader = MojoReaderBackendFactory.createReaderBackend(getClass().getResourceAsStream("/models/"+mojo_names[i]), MojoReaderBackendFactory.CachingStrategy.MEMORY);
                MojoModel model = ModelMojoReader.readFrom(reader);
                this.addModel(model);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    public void addModel(GenModel m) {
        String[] predictors = m.getNames();
        for (int i = 0; i < predictors.length; i++) {
            this._groupPredictors.put(predictors[i], new Predictor(this._groupPredictors.size(), m.getDomainValues(i)));
            this._groupIdxToColNames.add(predictors[i]);
        }
        this._predixors.addAll(Arrays.asList(Arrays.copyOfRange(m.getNames(), 0, m.getNames().length -1)));
        this.add(m);
    }

    public int mapEnum(int colIdx, String enumValue) {
        String[] domain = this._groupPredictors.get(this._groupIdxToColNames.get(colIdx)).domains;
        if (domain==null || domain.length==0) return -1;
        for (int i=0; i<domain.length;i++) if (enumValue.equals(domain[i])) return i;
        return -1;
    }

    public Object[] scoreAll(RowData data) {
        Object[] result_set = new Object[this.size()];
        try {
            for (int i = 0; i < this.size(); i++) {
                EasyPredictModelWrapper.Config config = new EasyPredictModelWrapper.Config();
                config.setConvertUnknownCategoricalLevelsToNa(true);
                config.setModel(this.get(i));
                switch (config.getModel().getModelCategory()) {
                    case Regression: {
                        EasyPredictModelWrapper modelWrapper = new EasyPredictModelWrapper(config);
                        RegressionModelPrediction prediction = (RegressionModelPrediction) modelWrapper.predictRegression(data);
                        result_set[i] = prediction.value;
                    }
                    case Binomial: {
                        EasyPredictModelWrapper modelWrapper = new EasyPredictModelWrapper(config);
                        BinomialModelPrediction prediction = (BinomialModelPrediction) modelWrapper.predictBinomial(data);
                        result_set[i] = prediction.label;
                    }
                    case Multinomial: {
                        ArrayList<Double> p = new ArrayList<Double>();
                        EasyPredictModelWrapper modelWrapper = new EasyPredictModelWrapper(config);
                        MultinomialModelPrediction prediction = (MultinomialModelPrediction) modelWrapper.predictMultinomial(data);
                        result_set[i] = prediction.label;
                    }
                }
            }
        } catch (PredictException pe) {
            pe.printStackTrace();
            throw new RuntimeException();
        }
        return result_set;
    }

    public String getColNamesString () {
        StringBuffer sb = new StringBuffer();
        int i = 1;
        for(String p: this._predixors) {
            if (this._predixors.size() != i) {
                i++;
                sb.append(p + ",");
            }
            else {
                sb.append(p);
            }
        }
        String result = sb.toString();
        return result;
    }

    /**
     * List directory contents for a resource folder. Not recursive.
     * This is basically a brute-force implementation.
     * Works for regular files and also JARs.
     *
     * @author Greg Briggs
     * @param clazz Any java class that lives in the same place as the resources you want.
     * @param path Should end with "/", but not start with one.
     * @return Just the name of each member item, not the full paths.
     * @throws URISyntaxException
     * @throws IOException
     */
    static String[] getResourceListing(Class clazz, String path) throws URISyntaxException, IOException {
        URL dirURL = clazz.getClassLoader().getResource(path);
        if (dirURL != null && dirURL.getProtocol().equals("file")) {
        /* A file path: easy enough */
            return new File(dirURL.toURI()).list();
        }

        if (dirURL == null) {
        /*
         * In case of a jar file, we can't actually find a directory.
         * Have to assume the same jar as clazz.
         */
            String me = clazz.getName().replace(".", "/")+".class";
            dirURL = clazz.getClassLoader().getResource(me);
        }

        if (dirURL.getProtocol().equals("jar")) {
        /* A JAR path */
            String jarPath = dirURL.getPath().substring(5, dirURL.getPath().indexOf("!")); //strip out only the JAR file
            JarFile jar = new JarFile(URLDecoder.decode(jarPath, "UTF-8"));
            Enumeration<JarEntry> entries = jar.entries(); //gives ALL entries in jar
            Set<String> result = new HashSet<String>(); //avoid duplicates in case it is a subdirectory
            while(entries.hasMoreElements()) {
                String name = entries.nextElement().getName();
                if (name.startsWith(path)) { //filter according to the path
                    String entry = name.substring(path.length());
                    int checkSubdir = entry.indexOf("/");
                    if (checkSubdir >= 0) {
                        // if it is a subdirectory, we just return the directory name
                        entry = entry.substring(0, checkSubdir);
                    }
                    result.add(entry);
                }
            }
            return result.toArray(new String[result.size()]);
        }

        throw new UnsupportedOperationException("Cannot list files for URL "+dirURL);
    }
}
