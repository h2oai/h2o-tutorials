GLRM Model
----------

What is a Low Rank Model?
-------------------------

-   **Given:** Data table *A* with m rows and n columns
-   **Find:** Compress representation as numeric tables *X* and *Y*,
    where \# cols in X = \# rows in Y = small user-specified k &lt;&lt;
    max(m, n)
-   \# cols in *Y* is d = (total dimension of embedded features in
    *A*) &gt;= n

In the example below, we are seeing the decomposition of A when A
consists of only numeric columns.

![](GLRM.png)

-   *Y* = archetypal features created from columns of *A*
-   *X* = row of *A* in reduced feature space
-   GLRM can approximately reconstruct *A* from product *XY*

GLRM Models Uses
----------------

GLRM models have multiple use cases:

-   filling in missing entries
-   reduce storage
-   remove noise
-   understand (visualize, cluster)

Understanding the Data
----------------------

We can use GLRM to understand the data. GLRM generates principal
components to interpret the data. Each row of Y is an archetypal feature
formed from the columns of A, and each row of X corresponds to a row of
A projected into this reduced dimension feature space.

For this tutorial, we will use data that contains attributes about a
house and its price in King County. The data is a public dataset from
Kaggle: <https://www.kaggle.com/harlfoxem/housesalesprediction>. We can
use GLRM to reduce the dimensions of the data 2 columns. We can use this
to visualize the houses in a 2 dimensional plot.

We will begin by initializing the h2o cluster and importing the csv
file.

``` r
# Initate H2O Cluster
library('h2o')
h2o.init()
h2o.no_progress()

# Import Data
houses_data <- h2o.importFile("/Users/megankurka/Downloads/kc_house_data.csv")
```

    ##           id            date   price bedrooms bathrooms sqft_living
    ## 1 7129300520 20141013T000000  221900        3      1.00        1180
    ## 2 6414100192 20141209T000000  538000        3      2.25        2570
    ## 3 5631500400 20150225T000000  180000        2      1.00         770
    ## 4 2487200875 20141209T000000  604000        4      3.00        1960
    ## 5 1954400510 20150218T000000  510000        3      2.00        1680
    ## 6 7237550310 20140512T000000 1225000        4      4.50        5420
    ##   sqft_lot floors waterfront view condition grade sqft_above sqft_basement
    ## 1     5650      1          0    0         3     7       1180             0
    ## 2     7242      2          0    0         3     7       2170           400
    ## 3    10000      1          0    0         3     6        770             0
    ## 4     5000      1          0    0         5     7       1050           910
    ## 5     8080      1          0    0         3     8       1680             0
    ## 6   101930      1          0    0         3    11       3890          1530
    ##   yr_built yr_renovated zipcode     lat     long sqft_living15 sqft_lot15
    ## 1     1955            0   98178 47.5112 -122.257          1340       5650
    ## 2     1951         1991   98125 47.7210 -122.319          1690       7639
    ## 3     1933            0   98028 47.7379 -122.233          2720       8062
    ## 4     1965            0   98136 47.5208 -122.393          1360       5000
    ## 5     1987            0   98074 47.6168 -122.045          1800       7503
    ## 6     2001            0   98053 47.6561 -122.005          4760     101930

There are 3 columns that are being treated as integer that should be
converted to categorical or enum. The columns `condition`, `grade`, and
`view` are really ordinal columns so we will convert it to categorical.
Later on we can use ordinal loss for these functions. The column
`zipcode` is also being treated as an integer when it is really a
categorical column.

``` r
cat_cols <- c('condition', 'grade', 'zipcode', 'view', 'waterfront')
for(i in cat_cols){
  houses_data[[i]] <- as.factor(houses_data[[i]])
}
```

When we define our first GLRM, we need to know which columns we want to
include in the model. In this case, we will not use `zipcode`, `id`, or
`date` since they have many unique values. We will also want to select
the loss function for GLRM. The GLRM model is trying to minimize this
loss function. It is tightly coupled to:

-   the use case
    -   some loss functions are sensitive to outliers and some are
        robust to outliers
-   the data type
    -   some loss functions are specific to the type of data in the
        column

For this example, we will try Absolute loss for the numeric features.
This loss will be robust to outliers. For the columns, `condition`,
`grade`, and `view`, we will use Ordinal loss.

``` r
glrm_cols <- colnames(houses_data)[!(colnames(houses_data) %in% c("id", "date", "zipcode"))]

losses <- data.frame('index' = which(colnames(houses_data) %in% c("condition", "grade", "view")) - 1,
                     'loss' = rep("Ordinal", 3), stringsAsFactors = FALSE)

rank2_model <- h2o.glrm(training_frame = houses_data, cols = glrm_cols, loss = "Absolute",
                        model_id = "rank2", seed = 1234, k = 2, transform = "STANDARDIZE",
                        loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

``` r
# Decompose training frame into XY

X <- h2o.getFrame(rank2_model@model$representation_name)
Y <- rank2_model@model$archetypes
```

In this case, the rows of Y represent some house attribute profiles. The
rows of X can be used to cluster houses.

Let’s visualize the first two archetypes of Y.

``` r
# Visualize first two archetypes of Y
archetypes_y <- as.data.frame(t(Y))
archetypes_y$attribute <- rownames(archetypes_y)
```

<!--html_preserve-->

<script type="application/json" data-for="845e77708a39">{"x":{"visdat":{"845e37ee3366":["function () ","plotlyVisDat"]},"cur_data":"845e37ee3366","attrs":{"845e37ee3366":{"x":{},"y":{},"mode":"markers","text":{},"alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"annotations":[{"x":1.98609084451502,"y":1.98205520023289,"text":"waterfront0","xref":"x","yref":"y"},{"x":-2.11745946112692,"y":-2.1144266444132,"text":"waterfront1","xref":"x","yref":"y"},{"x":-0.7419060360914,"y":0.254364530700241,"text":"price","xref":"x","yref":"y"},{"x":-0.814647495790879,"y":0.152423125474584,"text":"bedrooms","xref":"x","yref":"y"},{"x":-0.341514938985721,"y":-0.0111518681551548,"text":"sqft_lot","xref":"x","yref":"y"},{"x":-0.698958247897331,"y":0.439941538775305,"text":"floors","xref":"x","yref":"y"},{"x":-0.370840446987953,"y":-0.00801844981667596,"text":"sqft_lot15","xref":"x","yref":"y"}],"xaxis":{"domain":[0,1],"title":"Arch1"},"yaxis":{"domain":[0,1],"title":"Arch2"},"hovermode":"closest","showlegend":false},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[2.2249960243324,2.22468691085489,2.22169862016196,2.19676040557273,1.98674339248175,2.06174305881417,0.893676132150139,0.912457540275106,0.279906473899676,0.0574331165083931,0.00725758945765528,0,2.22200799631874,2.20428827459108,1.4446247166004,0.955063471418731,1.87740990475544,1.0572672045627,1.00277094533189,0.463509648440003,0.178178053357712,1.87740990475544,1.98609084451502,-2.11745946112692,-0.7419060360914,-0.814647495790879,-0.544389258155209,-0.941650549001347,-0.341514938985721,-0.698958247897331,-1.07174225378397,-0.627740833539926,-0.380950303795745,-0.25779989584804,0.0799182057052151,-0.511790178227278,-0.929595440253925,-0.370840446987953],"y":[2.22238321071291,2.22207521018837,2.21909880092268,2.19426077089212,1.98509102245483,1.0625412514417,1.89806010701756,0.922904903944292,0.283365712812646,0.0581521490074093,0.00734115597039724,0,2.219406028188,2.20174500819261,1.44311872653765,0.951988688676763,1.87526622289849,1.05710154731866,1.00653734513517,0.465920053907033,0.179195509512778,1.87526622289849,1.98205520023289,-2.1144266444132,0.254364530700241,0.152423125474584,0.875651399090045,0.575465546346727,-0.0111518681551548,0.439941538775305,0.52327759041199,-0.477801269530296,0.650216479982677,-0.228783575847215,0.0905956666862319,0.274996654761166,0.498027453253715,-0.00801844981667596],"mode":"markers","text":["grade1","grade3","grade4","grade5","grade6","grade7","grade8","grade9","grade10","grade11","grade12","grade13","condition1","condition2","condition3","condition4","condition5","view0","view1","view2","view3","view4","waterfront0","waterfront1","price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built","yr_renovated","lat","long","sqft_living15","sqft_lot15"],"type":"scatter","marker":{"fillcolor":"rgba(31,119,180,1)","color":"rgba(31,119,180,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
Attributes that are close together are considered similar. For example,
`sqft_lot` and `sqft_lot15` are very close together. This makes sense
since `sqft_lot` is the sqft of the lot where as `sqft_lot15` is the
sqft of the lot in 2015. Additionally `floors`, `bedrooms`, and `price`
are very close together. The values `waterfront0` and `waterfront1` are
far apart.

Now let’s visualize the first two archetypes of X.

``` r
# Visualize first two archetypes of X
archetypes_x <- as.data.frame(X)
archetypes_x$id <- as.character(as.matrix(houses_data$id))
set.seed(1234)
sample_indices <- sample(c(1:nrow(archetypes_x)), 100)
archetypes_x <- archetypes_x[sample_indices, ]

# Plot
plot_ly(data = archetypes_x, x = ~Arch1, y = ~Arch2, mode = "markers", 
        text = ~paste0("House: ", id))
```

<!--html_preserve-->

<script type="application/json" data-for="845e391d8a31">{"x":{"visdat":{"845e72801db1":["function () ","plotlyVisDat"]},"cur_data":"845e72801db1","attrs":{"845e72801db1":{"x":{},"y":{},"mode":"markers","text":{},"alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"Arch1"},"yaxis":{"domain":[0,1],"title":"Arch2"},"hovermode":"closest","showlegend":false},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[0.504475241222412,0.492853015607874,0.493109295683158,0.491893777940395,0.500070766859054,0.501340579680763,0.494886569052145,0.492300135859101,0.500030418026493,0.503877878799791,0.492091828161211,0.501438444178323,0.494782361747712,0.491488425665565,0.499605189225645,0.496455779081928,0.500067816634468,0.491056769698806,0.493865436437488,0.501727463940556,0.494843926930814,0.493354428767878,0.502975394271149,0.494566047926167,0.49459042915832,0.503705427560436,0.504490353984817,0.491844359257585,0.496111692654959,0.502339951182192,0.495621548307713,0.503999426630728,0.500246064083406,0.496326475693617,0.504886687180407,0.496073366886543,0.501817104358159,0.497545948931204,0.491692017755886,0.491258066901882,0.497187893346351,0.502849611454159,0.499826893140718,0.494812637815298,0.501705005155082,0.492586755199598,0.492294732945533,0.494975410798952,0.502481931978633,0.499998405054988,0.492364856726877,0.496358902946517,0.494591905763882,0.494779703002895,0.497383570195332,0.50183231964561,0.494937714828757,0.5040944317829,0.498039944234694,0.493352050813676,0.50054601960217,0.492677313594515,0.504031475692419,0.501938368162984,0.491351682225281,0.504050274376121,0.497144609462183,0.493780714335707,0.491992986419129,0.494451915517629,0.50425094502122,0.494466966922442,0.495473822260987,0.494211572525062,0.494111937209837,0.496198652942185,0.503990553893482,0.491891883704808,0.493581139643404,0.497986725815825,0.4915998992249,0.49179560304852,0.491971630324243,0.501778256685942,0.498967804720173,0.504697199454522,0.499404715215724,0.493624005106773,0.504639891328785,0.496088506261938,0.491158437960458,0.502298126893026,0.496893104981234,0.501512625677795,0.498664520566227,0.502718353316181,0.499027298901852,0.501751790598302,0.504410043186255,0.503583641184154],"y":[0.493160098702546,0.500553070277137,0.500151662497379,0.503372283286246,0.496703728448695,0.494547511011949,0.497662665767139,0.504529382434274,0.49381489958682,0.492328920367943,0.501910146865887,0.49298314988012,0.498322486914152,0.502918338902691,0.495173293629148,0.497176823075398,0.495432644370152,0.502716638515237,0.502196904400668,0.494317269683762,0.499474004129456,0.500473528583431,0.492687733866962,0.501794572838812,0.497339917854644,0.492601336505445,0.492584142860626,0.503765587494468,0.500130289347591,0.492937425195107,0.49801176563102,0.492510420618207,0.496142227341497,0.498184184007275,0.493065770856944,0.499873847158492,0.493132514464235,0.498149357660555,0.501556750791814,0.50192005945097,0.496772134711579,0.493378530989253,0.494864061047869,0.501271486143668,0.493340943561946,0.501959913561617,0.50209851407484,0.499352760094121,0.493335205915612,0.496125892029365,0.501599423039656,0.496441213182539,0.50000408394738,0.498470512238436,0.495127829399027,0.493324127996325,0.499325402986894,0.492902999046019,0.497108897430841,0.499958808370356,0.497062231649313,0.502180106508365,0.492504536696425,0.494833769518749,0.502397146775771,0.492685024206314,0.497619658720069,0.501458276510673,0.502239839487422,0.499698826816754,0.492595879333812,0.499212847340973,0.498381236862018,0.500400008156644,0.500803693032664,0.498511272554786,0.492513177182884,0.502238594129573,0.501644237245102,0.497574894669583,0.503319604295468,0.502253012103277,0.502562151644849,0.493581299100068,0.495442011178553,0.492594605568077,0.496669731156678,0.504208051783471,0.492589486658941,0.497942194493805,0.502928667962148,0.493097757576203,0.497561851180537,0.494027595315912,0.497204662530781,0.493088155321105,0.495592447483269,0.494278900225469,0.492598763302445,0.492462557096362],"mode":"markers","text":["House: 5220300140","House: 164000237","House: 8078420230","House: 98030530","House: 6814600150","House: 9477100620","House: 3456000310","House: 98000950","House: 1126049103","House: 4337000070","House: 6308000010","House: 7577700136","House: 3376600010","House: 1042700300","House: 1189000825","House: 4232903990","House: 2787310130","House: 3904920980","House: 5700003585","House: 3362401815","House: 1024039001","House: 2141300420","House: 8121100015","House: 1937300270","House: 7635801311","House: 7889601300","House: 3046200125","House: 6790830060","House: 9828702588","House: 316000160","House: 3260000340","House: 1788800630","House: 4318200440","House: 7202290240","House: 795000820","House: 3896100130","House: 9353300600","House: 8656300380","House: 5045700330","House: 7853300020","House: 5112800060","House: 3223059141","House: 7895500290","House: 624069003","House: 2489200250","House: 5469500640","House: 3326059254","House: 5249802085","House: 236500010","House: 7614100020","House: 4221270340","House: 2795000080","House: 5631500947","House: 7504400710","House: 7409700215","House: 8099200030","House: 724069070","House: 7972601680","House: 1105000011","House: 1241500147","House: 6204200590","House: 2296500136","House: 3361400190","House: 5468730030","House: 9526600250","House: 7304301300","House: 3491300052","House: 1560920200","House: 1509500160","House: 7212660560","House: 179003055","House: 8563001130","House: 1328320800","House: 4315700275","House: 2926069062","House: 3047700045","House: 3883800011","House: 7312000240","House: 1137800460","House: 323059103","House: 293070090","House: 6305900350","House: 1454600038","House: 7518502030","House: 7923300115","House: 2114700090","House: 1972201965","House: 2624089007","House: 7813200115","House: 3416600490","House: 2825059256","House: 1773101215","House: 6706600090","House: 7227801955","House: 5413200140","House: 4363700200","House: 9477730080","House: 2473360060","House: 5379802181","House: 3797001900"],"type":"scatter","marker":{"fillcolor":"rgba(31,119,180,1)","color":"rgba(31,119,180,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
Houses that are close together are considered similar. If we examine the
two houses next to each other at the bottom of the graph, we can see a
lot of similarities. The two houses are highlighted in red.

<!--html_preserve-->

<script type="application/json" data-for="845e6c693584">{"x":{"visdat":{"845e3498147d":["function () ","plotlyVisDat"]},"cur_data":"845e3498147d","attrs":{"845e3498147d":{"x":{},"y":{},"mode":"markers","text":{},"color":{},"colors":"Set1","alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"showlegend":false,"xaxis":{"domain":[0,1],"title":"Arch1"},"yaxis":{"domain":[0,1],"title":"Arch2"},"hovermode":"closest"},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[0.504490353984817,0.504410043186255],"y":[0.492584142860626,0.492598763302445],"mode":"markers","text":["House: 3046200125","House: 5379802181"],"type":"scatter","name":"A","marker":{"fillcolor":"rgba(228,26,28,0.5)","color":"rgba(228,26,28,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null},{"x":[0.504475241222412,0.492853015607874,0.493109295683158,0.491893777940395,0.500070766859054,0.501340579680763,0.494886569052145,0.492300135859101,0.500030418026493,0.503877878799791,0.492091828161211,0.501438444178323,0.494782361747712,0.491488425665565,0.499605189225645,0.496455779081928,0.500067816634468,0.491056769698806,0.493865436437488,0.501727463940556,0.494843926930814,0.493354428767878,0.502975394271149,0.494566047926167,0.49459042915832,0.503705427560436,0.491844359257585,0.496111692654959,0.502339951182192,0.495621548307713,0.503999426630728,0.500246064083406,0.496326475693617,0.504886687180407,0.496073366886543,0.501817104358159,0.497545948931204,0.491692017755886,0.491258066901882,0.497187893346351,0.502849611454159,0.499826893140718,0.494812637815298,0.501705005155082,0.492586755199598,0.492294732945533,0.494975410798952,0.502481931978633,0.499998405054988,0.492364856726877,0.496358902946517,0.494591905763882,0.494779703002895,0.497383570195332,0.50183231964561,0.494937714828757,0.5040944317829,0.498039944234694,0.493352050813676,0.50054601960217,0.492677313594515,0.504031475692419,0.501938368162984,0.491351682225281,0.504050274376121,0.497144609462183,0.493780714335707,0.491992986419129,0.494451915517629,0.50425094502122,0.494466966922442,0.495473822260987,0.494211572525062,0.494111937209837,0.496198652942185,0.503990553893482,0.491891883704808,0.493581139643404,0.497986725815825,0.4915998992249,0.49179560304852,0.491971630324243,0.501778256685942,0.498967804720173,0.504697199454522,0.499404715215724,0.493624005106773,0.504639891328785,0.496088506261938,0.491158437960458,0.502298126893026,0.496893104981234,0.501512625677795,0.498664520566227,0.502718353316181,0.499027298901852,0.501751790598302,0.503583641184154],"y":[0.493160098702546,0.500553070277137,0.500151662497379,0.503372283286246,0.496703728448695,0.494547511011949,0.497662665767139,0.504529382434274,0.49381489958682,0.492328920367943,0.501910146865887,0.49298314988012,0.498322486914152,0.502918338902691,0.495173293629148,0.497176823075398,0.495432644370152,0.502716638515237,0.502196904400668,0.494317269683762,0.499474004129456,0.500473528583431,0.492687733866962,0.501794572838812,0.497339917854644,0.492601336505445,0.503765587494468,0.500130289347591,0.492937425195107,0.49801176563102,0.492510420618207,0.496142227341497,0.498184184007275,0.493065770856944,0.499873847158492,0.493132514464235,0.498149357660555,0.501556750791814,0.50192005945097,0.496772134711579,0.493378530989253,0.494864061047869,0.501271486143668,0.493340943561946,0.501959913561617,0.50209851407484,0.499352760094121,0.493335205915612,0.496125892029365,0.501599423039656,0.496441213182539,0.50000408394738,0.498470512238436,0.495127829399027,0.493324127996325,0.499325402986894,0.492902999046019,0.497108897430841,0.499958808370356,0.497062231649313,0.502180106508365,0.492504536696425,0.494833769518749,0.502397146775771,0.492685024206314,0.497619658720069,0.501458276510673,0.502239839487422,0.499698826816754,0.492595879333812,0.499212847340973,0.498381236862018,0.500400008156644,0.500803693032664,0.498511272554786,0.492513177182884,0.502238594129573,0.501644237245102,0.497574894669583,0.503319604295468,0.502253012103277,0.502562151644849,0.493581299100068,0.495442011178553,0.492594605568077,0.496669731156678,0.504208051783471,0.492589486658941,0.497942194493805,0.502928667962148,0.493097757576203,0.497561851180537,0.494027595315912,0.497204662530781,0.493088155321105,0.495592447483269,0.494278900225469,0.492462557096362],"mode":"markers","text":["House: 5220300140","House: 164000237","House: 8078420230","House: 98030530","House: 6814600150","House: 9477100620","House: 3456000310","House: 98000950","House: 1126049103","House: 4337000070","House: 6308000010","House: 7577700136","House: 3376600010","House: 1042700300","House: 1189000825","House: 4232903990","House: 2787310130","House: 3904920980","House: 5700003585","House: 3362401815","House: 1024039001","House: 2141300420","House: 8121100015","House: 1937300270","House: 7635801311","House: 7889601300","House: 6790830060","House: 9828702588","House: 316000160","House: 3260000340","House: 1788800630","House: 4318200440","House: 7202290240","House: 795000820","House: 3896100130","House: 9353300600","House: 8656300380","House: 5045700330","House: 7853300020","House: 5112800060","House: 3223059141","House: 7895500290","House: 624069003","House: 2489200250","House: 5469500640","House: 3326059254","House: 5249802085","House: 236500010","House: 7614100020","House: 4221270340","House: 2795000080","House: 5631500947","House: 7504400710","House: 7409700215","House: 8099200030","House: 724069070","House: 7972601680","House: 1105000011","House: 1241500147","House: 6204200590","House: 2296500136","House: 3361400190","House: 5468730030","House: 9526600250","House: 7304301300","House: 3491300052","House: 1560920200","House: 1509500160","House: 7212660560","House: 179003055","House: 8563001130","House: 1328320800","House: 4315700275","House: 2926069062","House: 3047700045","House: 3883800011","House: 7312000240","House: 1137800460","House: 323059103","House: 293070090","House: 6305900350","House: 1454600038","House: 7518502030","House: 7923300115","House: 2114700090","House: 1972201965","House: 2624089007","House: 7813200115","House: 3416600490","House: 2825059256","House: 1773101215","House: 6706600090","House: 7227801955","House: 5413200140","House: 4363700200","House: 9477730080","House: 2473360060","House: 3797001900"],"type":"scatter","name":"B","marker":{"fillcolor":"rgba(153,153,153,0.5)","color":"rgba(153,153,153,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
The houses are shown below:

|          id| date            |   price|  bedrooms|  bathrooms|  sqft\_living|  sqft\_lot|  floors| waterfront | view | condition | grade |  sqft\_above|  sqft\_basement|  yr\_built|  yr\_renovated| zipcode |      lat|      long|  sqft\_living15|  sqft\_lot15|
|-----------:|:----------------|-------:|---------:|----------:|-------------:|----------:|-------:|:-----------|:-----|:----------|:------|------------:|---------------:|----------:|--------------:|:--------|--------:|---------:|---------------:|------------:|
|  5379802181| 20141119T000000 |  193000|         2|          1|           680|       8640|       1| 0          | 0    | 4         | 5     |          680|               0|       1951|              0| 98188   |  47.4559|  -122.289|            1320|        13140|
|  3046200125| 20150406T000000 |  202000|         2|          1|           740|       6550|       1| 0          | 0    | 4         | 5     |          740|               0|       1946|              0| 98168   |  47.4807|  -122.332|            1080|         8515|

We can see that they have very similar sizes and price and were also
built during the same 5 year period.

#### Clustering

Now that we’ve examined the archetypal space, we can cluster the houses
based on their archetypes. This is more effective than clustering on the
raw data because the raw data has noise and correlated attributes.

``` r
kmeans_data <- h2o.cbind(houses_data, X)

kmeans_model <- h2o.kmeans(kmeans_data, x = c("Arch1",  "Arch2"), estimate_k = TRUE, k = 100, seed = 1234)
```

We can let H2O choose the best number of clusters by speciying
`estimate_k = TRUE`.

We will add the clusters to our original visualization.

``` r
# Visualize employee clusters
clusters <- h2o.predict(kmeans_model, kmeans_data)
colnames(clusters) <- "cluster"
kmeans_data <- h2o.cbind(kmeans_data, clusters)

kmeans_df <- as.data.frame(kmeans_data[c("Arch1", "Arch2", "cluster", "id")])[sample_indices, ]
kmeans_df$cluster <- as.factor(kmeans_df$cluster)
plot_ly(data = kmeans_df, x = ~Arch1, y = ~Arch2, mode = "markers", text = ~paste0("House: ", id), color = ~cluster)
```

<!--html_preserve-->

<script type="application/json" data-for="845e43f1699d">{"x":{"visdat":{"845e7d2202f9":["function () ","plotlyVisDat"]},"cur_data":"845e7d2202f9","attrs":{"845e7d2202f9":{"x":{},"y":{},"mode":"markers","text":{},"color":{},"alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"Arch1"},"yaxis":{"domain":[0,1],"title":"Arch2"},"hovermode":"closest","showlegend":true},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[0.504475241222412,0.500070766859054,0.501340579680763,0.500030418026493,0.503877878799791,0.501438444178323,0.499605189225645,0.500067816634468,0.501727463940556,0.502975394271149,0.503705427560436,0.504490353984817,0.502339951182192,0.503999426630728,0.500246064083406,0.504886687180407,0.501817104358159,0.497187893346351,0.502849611454159,0.499826893140718,0.501705005155082,0.502481931978633,0.499998405054988,0.496358902946517,0.497383570195332,0.50183231964561,0.5040944317829,0.498039944234694,0.50054601960217,0.504031475692419,0.501938368162984,0.504050274376121,0.50425094502122,0.503990553893482,0.497986725815825,0.501778256685942,0.498967804720173,0.504697199454522,0.499404715215724,0.504639891328785,0.502298126893026,0.501512625677795,0.498664520566227,0.502718353316181,0.499027298901852,0.501751790598302,0.504410043186255,0.503583641184154],"y":[0.493160098702546,0.496703728448695,0.494547511011949,0.49381489958682,0.492328920367943,0.49298314988012,0.495173293629148,0.495432644370152,0.494317269683762,0.492687733866962,0.492601336505445,0.492584142860626,0.492937425195107,0.492510420618207,0.496142227341497,0.493065770856944,0.493132514464235,0.496772134711579,0.493378530989253,0.494864061047869,0.493340943561946,0.493335205915612,0.496125892029365,0.496441213182539,0.495127829399027,0.493324127996325,0.492902999046019,0.497108897430841,0.497062231649313,0.492504536696425,0.494833769518749,0.492685024206314,0.492595879333812,0.492513177182884,0.497574894669583,0.493581299100068,0.495442011178553,0.492594605568077,0.496669731156678,0.492589486658941,0.493097757576203,0.494027595315912,0.497204662530781,0.493088155321105,0.495592447483269,0.494278900225469,0.492598763302445,0.492462557096362],"mode":"markers","text":["House: 5220300140","House: 6814600150","House: 9477100620","House: 1126049103","House: 4337000070","House: 7577700136","House: 1189000825","House: 2787310130","House: 3362401815","House: 8121100015","House: 7889601300","House: 3046200125","House: 316000160","House: 1788800630","House: 4318200440","House: 795000820","House: 9353300600","House: 5112800060","House: 3223059141","House: 7895500290","House: 2489200250","House: 236500010","House: 7614100020","House: 2795000080","House: 7409700215","House: 8099200030","House: 7972601680","House: 1105000011","House: 6204200590","House: 3361400190","House: 5468730030","House: 7304301300","House: 179003055","House: 3883800011","House: 323059103","House: 7518502030","House: 7923300115","House: 2114700090","House: 1972201965","House: 7813200115","House: 1773101215","House: 7227801955","House: 5413200140","House: 4363700200","House: 9477730080","House: 2473360060","House: 5379802181","House: 3797001900"],"type":"scatter","name":"0","marker":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null},{"x":[0.492853015607874,0.493109295683158,0.491893777940395,0.494886569052145,0.492300135859101,0.492091828161211,0.494782361747712,0.491488425665565,0.496455779081928,0.491056769698806,0.493865436437488,0.494843926930814,0.493354428767878,0.494566047926167,0.49459042915832,0.491844359257585,0.496111692654959,0.495621548307713,0.496326475693617,0.496073366886543,0.497545948931204,0.491692017755886,0.491258066901882,0.494812637815298,0.492586755199598,0.492294732945533,0.494975410798952,0.492364856726877,0.494591905763882,0.494779703002895,0.494937714828757,0.493352050813676,0.492677313594515,0.491351682225281,0.497144609462183,0.493780714335707,0.491992986419129,0.494451915517629,0.494466966922442,0.495473822260987,0.494211572525062,0.494111937209837,0.496198652942185,0.491891883704808,0.493581139643404,0.4915998992249,0.49179560304852,0.491971630324243,0.493624005106773,0.496088506261938,0.491158437960458,0.496893104981234],"y":[0.500553070277137,0.500151662497379,0.503372283286246,0.497662665767139,0.504529382434274,0.501910146865887,0.498322486914152,0.502918338902691,0.497176823075398,0.502716638515237,0.502196904400668,0.499474004129456,0.500473528583431,0.501794572838812,0.497339917854644,0.503765587494468,0.500130289347591,0.49801176563102,0.498184184007275,0.499873847158492,0.498149357660555,0.501556750791814,0.50192005945097,0.501271486143668,0.501959913561617,0.50209851407484,0.499352760094121,0.501599423039656,0.50000408394738,0.498470512238436,0.499325402986894,0.499958808370356,0.502180106508365,0.502397146775771,0.497619658720069,0.501458276510673,0.502239839487422,0.499698826816754,0.499212847340973,0.498381236862018,0.500400008156644,0.500803693032664,0.498511272554786,0.502238594129573,0.501644237245102,0.503319604295468,0.502253012103277,0.502562151644849,0.504208051783471,0.497942194493805,0.502928667962148,0.497561851180537],"mode":"markers","text":["House: 164000237","House: 8078420230","House: 98030530","House: 3456000310","House: 98000950","House: 6308000010","House: 3376600010","House: 1042700300","House: 4232903990","House: 3904920980","House: 5700003585","House: 1024039001","House: 2141300420","House: 1937300270","House: 7635801311","House: 6790830060","House: 9828702588","House: 3260000340","House: 7202290240","House: 3896100130","House: 8656300380","House: 5045700330","House: 7853300020","House: 624069003","House: 5469500640","House: 3326059254","House: 5249802085","House: 4221270340","House: 5631500947","House: 7504400710","House: 724069070","House: 1241500147","House: 2296500136","House: 9526600250","House: 3491300052","House: 1560920200","House: 1509500160","House: 7212660560","House: 8563001130","House: 1328320800","House: 4315700275","House: 2926069062","House: 3047700045","House: 7312000240","House: 1137800460","House: 293070090","House: 6305900350","House: 1454600038","House: 2624089007","House: 3416600490","House: 2825059256","House: 6706600090"],"type":"scatter","name":"1","marker":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
There are two clusters found. If we examine these clusters, we can see
that the top cluster represents more expensive and larger houses with a
higher grade.

``` r
cluster_stats <- h2o.group_by(kmeans_data, "cluster", mean("price"), mean("sqft_lot"), mean("bedrooms"), mean("grade"))
```

|  cluster|  mean\_price|  mean\_sqft\_lot|  mean\_bedrooms|  mean\_grade|
|--------:|------------:|----------------:|---------------:|------------:|
|        0|     392913.4|         10460.50|        2.974798|     4.997285|
|        1|     716569.1|         20678.67|        3.845747|     6.447904|

Filling in Missing Entries
--------------------------

Another use of GLRM would be to fill in any missing entries. We can view
missing entries as NA’s or events that have not yet happened. For
example, if we had data of customers and products, we could place NA’s
where a user has not yet purchased a product. In this way, by using GLRM
to impute missing values, we are also creating a recommender.

For our houses dataset, we do not having any missing values. We will
manually add missing values to our dataset to show how to use GLRM for
imputation.

### Split the Data

Given a training frame A, GLRM will produce a matrix factorization X and
Y. During validation, it approximately reconstructs A as a function of
the matrix product XY. This gives A = f(XY) which has the same number of
dimensions and features as A. Any missing values in A will also be
automatically filled in. Validation occurs by comparing A with the
validation frame and computing the squared error (for numeric entries)
and the number of mismatches (for categorical entries).

The correct way to do validation in GLRM is to take a dataset with no
(or very few) missing entries as your validation frame B. Create a deep
copy of B, then deliberately replace known values with missing values.

``` r
houses_data <- h2o.assign(houses_data, "houses_data.hex")
miss_data <- h2o.assign(houses_data, "miss_data.hex")
h2o.insertMissingValues(data = miss_data, fraction = 0.15, seed = 1234)
```

We can now compare the number of NA’s between the two datasets:

``` r
sum(is.na(houses_data))
```

    ## [1] 0

``` r
sum(is.na(miss_data))
```

    ## [1] 68240

### Determining Best Parameters

We will build a model using the default parameters and evaluate it. We
will set the rank (`k`) of the model to 5 and the regularization
strength to 2. We will tune these parameters later on. We will also tell
the GLRM model to use Ordinal loss for our ordinal columns: `view`,
`condition`, and `grade`.

``` r
glrm_k <- 5
gamma <- 2

losses <- data.frame('index' = which(colnames(houses_data) %in% c("condition", "grade", "view")) - 1,
                     'loss' = rep("Ordinal", 3), stringsAsFactors = FALSE)

base_model <- h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
                       model_id = "base_glrm", seed = 1234, 
                       k = glrm_k, gamma_x = gamma, gamma_y = gamma, 
                       regularization_x = "Quadratic", regularization_y = "Quadratic",
                       loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

Let’s take a look at the validation error.

``` r
print(h2o.performance(base_model, valid = T))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  3.262888e+15
    ## Misclassification Error (Categorical):  57857
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

We have a very high numeric error. We are also misclassifying more than
50% of the categorical values.  
Let’s improve this by transforming the numerical values in the dataset.
This is advised because some columns may have very large means which may
cause our model to be biased.

``` r
standardized_model <- h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data,
                               model_id = "standardized_glrm", seed = 1234,
                               k = glrm_k, gamma_x = gamma, gamma_y = gamma, 
                               regularization_x = "Quadratic", regularization_y = "Quadratic",
                               transform = "STANDARDIZE", impute_original = TRUE,
                               loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

Since we are transforming the data, we make sure to include
`impute_original = TRUE`. This will reverse the transformation during
prediction.

``` r
print(h2o.performance(standardized_model, valid = T))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  8.935894e+14
    ## Misclassification Error (Categorical):  62968
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

Here we see a great reduction in the numeric error.

### Effects of Rank

Rank controls the size of our decomposed matrices. If our rank is too
low, we will have high error because we are not including some
archetypes. If our rank is too high, our X and Y matrices are
unnecessarily large and this may cause overfitting. We will examine how
different ranks effect the reconstruction error on the validation data
by training a GLRM model for the ranks 1, 3, 5, …, 21.

``` r
library('plyr')
k_range <- seq(1, 21, 4)

k_models <- llply(k_range, function(k) 
  h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
           model_id = paste0("glrm_k_", k), seed = 1234,
           k = k, gamma_x = gamma, gamma_y = gamma, 
           regularization_x = "Quadratic", regularization_y = "Quadratic", 
           transform = "STANDARDIZE", impute_original = TRUE,
           loss_by_col = losses$loss, loss_by_col_idx = losses$index))
```

Below we generate a scree plot of the numeric error for each model.

``` r
# Plot Numeric Error and Rank
library('plotly')

k_error <- ldply(k_models, function(ml) 
  data.frame('k' = ml@parameters$k, 
             'numeric_error' = c(h2o.performance(ml, valid = FALSE)@metrics$numerr,
                                 h2o.performance(ml, valid = TRUE)@metrics$numerr),
             'type' = c("training", "validation"))
)

plot_ly(data = k_error, x = ~k, y = ~numeric_error, mode = "markers+lines", color = ~type, text = ~paste0("k: ", k))
```

<!--html_preserve-->

<script type="application/json" data-for="845e51bfbe20">{"x":{"visdat":{"845e50ddf4f":["function () ","plotlyVisDat"]},"cur_data":"845e50ddf4f","attrs":{"845e50ddf4f":{"x":{},"y":{},"mode":"markers+lines","text":{},"color":{},"alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"k"},"yaxis":{"domain":[0,1],"title":"numeric_error"},"hovermode":"closest","showlegend":true},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[1,5,9,13,17,21],"y":[2.59501458630861e+15,698826512518854,316129431004024,13929679626589.8,2579439445001.83,712948764326.498],"mode":"markers+lines","text":["k: 1","k: 5","k: 9","k: 13","k: 17","k: 21"],"type":"scatter","name":"training","line":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)"},"xaxis":"x","yaxis":"y","frame":null},{"x":[1,5,9,13,17,21],"y":[2.96514073380301e+15,893589414178790,538454057122535,384126971454495,169689731508388,163939817940835],"mode":"markers+lines","text":["k: 1","k: 5","k: 9","k: 13","k: 17","k: 21"],"type":"scatter","name":"validation","line":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
Based on our scree plot, it looks like our optimal k is 17. After this
value, the improvement plateaus. If our aim was to lower the
dimensionality of the data, we could choose the elbow of the graph as
`k`. In this case, the elbow is round `k = 5`.

The scree plot of the validation error should helps us to determine the
true rank of the data. The true rank of the data should encompass the
signal of the data, not the noise. When we see that the validation error
is increasing as k increases, the low rank matrix is beginning to
capture not just the signal but also the noise.

### Effects of Regularization Strength

Regularization strength is especially important if there are a lot of
missing values in the data to prevent overfitting. Regularization is
added to the loss function so that our new goal is to mimimize:

$$
L\_j(x\_i y\_j, A\_{ij}) + \\gamma\_x \\sum\_{i=1}^m r\_i(x\_i) + \\gamma\_y \\sum\_{j=1}^n \\hat{r}\_j(y\_j)
$$

We will examine how different regularization strengths effect the
reconstruction error on the validation data when `k = 17`. In a future
version of this tutorial, we will run a grid search over both `k` and
`gamma` since the effects of each parameter are dependent of each other.

``` r
gamma_range <- seq(0, 5, 1)

gamma_models <- llply(gamma_range, function(g) 
  h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
           model_id = paste0("glrm_gamma_", g), seed = 1234,
           k = 17, gamma_x = g, gamma_y = g, 
           regularization_x = "Quadratic", regularization_y = "Quadratic", 
           transform = "STANDARDIZE", impute_original = TRUE,
           loss_by_col = losses$loss, loss_by_col_idx = losses$index))
```

Below we generate a scree plot of the numeric error for each model.

``` r
# Plot Numeric Error and Regularization Strength/Gamma
gamma_error <- ldply(gamma_models, function(ml) 
  data.frame('gamma' = ifelse(is.null(ml@parameters$gamma_x), 0,  ml@parameters$gamma_x),
             'numeric_error' = c(h2o.performance(ml, valid = FALSE)@metrics$numerr,
                                 h2o.performance(ml, valid = TRUE)@metrics$numerr),
             'type' = c("training", "validation"))
)

plot_ly(data = gamma_error, x = ~gamma, y = ~numeric_error, mode = "markers+lines", 
        color = ~type, text = ~paste0("gamma: ", gamma))
```

<!--html_preserve-->

<script type="application/json" data-for="845e4fb6b4a1">{"x":{"visdat":{"845e2c5a1b4e":["function () ","plotlyVisDat"]},"cur_data":"845e2c5a1b4e","attrs":{"845e2c5a1b4e":{"x":{},"y":{},"mode":"markers+lines","text":{},"color":{},"alpha":1,"sizes":[10,100]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"gamma"},"yaxis":{"domain":[0,1],"title":"numeric_error"},"hovermode":"closest","showlegend":true},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"x":[0,1,2,3,4,5],"y":[808187797923.702,181792486350.192,2579439445001.83,3109578101357.24,3097676652496.57,3863929173707.22],"mode":"markers+lines","text":["gamma: 0","gamma: 1","gamma: 2","gamma: 3","gamma: 4","gamma: 5"],"type":"scatter","name":"training","line":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)"},"xaxis":"x","yaxis":"y","frame":null},{"x":[0,1,2,3,4,5],"y":[207597506129295,157637225972158,169689731508388,173674524758543,151687467736007,150263559114054],"mode":"markers+lines","text":["gamma: 0","gamma: 1","gamma: 2","gamma: 3","gamma: 4","gamma: 5"],"type":"scatter","name":"validation","line":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script>
<!--/html_preserve-->
The relationship between regularization strength and validation error
shows a different story than what we saw with rank. Some regularization
helps improve the validation error but too much hurts the performance.
It looks like we have the best performance with a regularization
strength around 1.

### Our Final Model

Below we generate the final model and view the performance metrics.

``` r
final_glrm <- gamma_models[[2]]
```

``` r
print(h2o.performance(final_glrm, valid = TRUE))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  1.576372e+14
    ## Misclassification Error (Categorical):  62837
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

### Matrix Completion and Denoising

Matrix Completion and Denoising occur by multiplying the matrices X and
Y (if the original data was numeric) or using the function `h2o.predict`
(in the event of mixed data types).

In our `miss_data`, we manually created ~68k missing values. We will use
our final GLRM model to impute these missing values.

``` r
# Predict data based on low rank matrices
houses_pred <- h2o.predict(final_glrm, miss_data)

# Add House ID
houses_pred <- h2o.cbind(miss_data$id, houses_pred)
```

Let’s take a look at the number of missing values in the predicted data.

``` r
print(sum(is.na(miss_data[glrm_cols])))
```

    ## [1] 58488

``` r
print(sum(is.na(houses_pred[c(2:ncol(houses_pred))])))
```

    ## [1] 0

Let’s examine the first record which had missing values added to the
features `floors` and `sqft_above`.

|  floors|  sqft\_above|
|-------:|------------:|
|       1|         1180|

|  reconstr\_floors|  reconstr\_sqft\_above|
|-----------------:|----------------------:|
|          1.229246|               1181.704|

The GLRM model is able to correctly estimate the number of floors and
the sqft of the interior above the ground level.

Compressing Data
----------------

If the goal of GLRM is to compress data, then choosing the best rank of
the model may be done slightly differently. As seen before, we want to
minimize the error on a validation dataset, however, now we need to keep
in mind the compression rate.

Compression Rate = Number of Non Zeros is X and Y Matrices/ Number of
Non Zeros in original data

If the decomposed matrices are dense, then the compression rate is
simply =
`(num rows of X + num cols of Y)*k/(num cols of A * num rows of A)`
where A is the original data.

We have a dense decomposition because we have used the quadratic
regularization. Therefore, the compression rate is
`((1470 + 77)*9)/(1470*35) = 27%`. We may choose to compress further by
using an L1 regularization instead of L2.

If we need to achieve a compression rate of 10%, then we would choose
the rank that minimizes the validation error under the restriction that
the compression rate is &lt;= 10%.
