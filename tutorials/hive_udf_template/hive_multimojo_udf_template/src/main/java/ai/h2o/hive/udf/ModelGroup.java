package ai.h2o.hive.udf;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import hex.genmodel.*;

public class ModelGroup extends ArrayList<GenModel> {

    //Create a predictor class which keeps track of indexes and domains
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

    //Create a HashMap of Strings & Predictors. Make it linked as order matters.
    public LinkedHashMap<String, Predictor> _groupPredictors;

    //Create an ArrayList of strings
    public ArrayList<String> _groupIdxToColNames;

    //Create a constructor and initialize LinkedHashMap & Arraylist
    public ModelGroup() {
        this._groupPredictors = new LinkedHashMap<String, Predictor>();
        this._groupIdxToColNames = new ArrayList<String>();
    }

    //Function to get MOJO names and store them in a string array
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

    //Function to add MOJOs from JARs
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

    //Add models and add predictors to _groupPredictors and _groupIdxToColNames
    public void addModel(GenModel m) {
        String[] predictors = m.getNames();
        for(int i = 0; i < predictors.length; i++) {
            if(this._groupPredictors.get(predictors[i]) == null) {
                this._groupPredictors.put(predictors[i], new Predictor(this._groupPredictors.size(), m.getDomainValues(i)));
                this._groupIdxToColNames.add(predictors[i]);
            }
        }
        this.add(m);
    }

    public int mapEnum(int colIdx, String enumValue) {
        String[] domain = this._groupPredictors.get(this._groupIdxToColNames.get(colIdx)).domains;
        if (domain == null || domain.length == 0) return -1;
        for (int i = 0; i < domain.length; i++) if (enumValue.equals(domain[i])) return i;
        return -1;
    }

    //scoreAll function (this is where the actual scoring is done)
    public ArrayList<ArrayList<Double>> scoreAll(double[] data) {
        ArrayList<ArrayList<Double>> result_set = new ArrayList<ArrayList<Double>>();

        for (int i = 0; i < this.size(); i++) {
            GenModel m = this.get(i);
            String[] features = m.getNames();
            double[] model_data = new double[features.length];
            double[] model_response = new double[m.getPredsSize()];
            for(int j = 0; j < features.length; j++) {
                model_data[j] = data[this._groupPredictors.get(features[j]).index];
            }

            // get and add prediction to result
            double[] prediction = m.score0(model_data, model_response);
            ArrayList<Double> p = new ArrayList<Double>();
            for(double d: prediction) p.add(d);

            result_set.add(p);
        }
        return result_set;
    }

    public String getColNamesString () {
        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < this._groupIdxToColNames.size(); i++) {
            sb.append(this._groupIdxToColNames.get(i));
            if (i + 1 != this._groupIdxToColNames.size()) sb.append(",");
        }
        return sb.toString();
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
