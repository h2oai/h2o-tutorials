# Driverless AI

### Version 1.0.1

## Transformations

Transformations are applied to columns in the data.  The transformers create the engineered features.

We will describe the transformations using the example of predicting house prices on the example dataset:

| Date Built | Square Footage | Num Beds | Num Baths | State | Price |
|---|---|---|---|---|---|
| 01/01/1920 | 1700| 3 | 2 | NY | $700K |

### Frequent Transformer

* the count of each categorical value in the dataset
* the count can be either the raw count or the normalized count

#### Example

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | Freq_State|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 4,500 |

There are 4,500 properties in this dataset with state = NY.

### Bulk Interactions Transfomer

* add, divide, multiply, and subtract two columns in the data

#### Example

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | Interaction_NumBeds#subtract#NumBaths|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 1 | 

There is one more bedroom than there are number of bathrooms for this property.

### Truncated SVD Numeric Transformer

* truncated SVD trained on selected numeric columns of the data 
* the components of the truncated SVD will be new features

#### Example

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | TruncSVD_Price\_NumBeds\_NumBaths\_1|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 0.632 | 

The first component of the truncated SVD of the columns Price, Number of Beds, Number of Baths.

### Dates Transformer

* get year, get quarter, get month, get day, get day of year, get week, get week day, get hour, get minute, get second

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | DateBuilt_Month|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 1 | 

The home was built in the month January.

### Date Polar Transformer

This transformer expands the date using polar coordinates.  The Date Transformer (described above) will only expand the date into different units, for example month.  This does not capture the similarity betwen the months December and January (12 and 1) or the hours 23 and 0.  The polar coordinates capture the similarities between these cases by representing the unit of the date as a point in a cycle.  For example, the polar coordinates of: `get minute in hour`, would be the minute hand position on a clock.

* get hour in day, get minute in hour, get day in month, get day in year, get quarter in year, get month in year, get week in year, get week day in week

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | DateBuilt\_MonthInYear\_x| DateBuilt\_MonthInYear\_y|
|---|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 0.5 | 1 |

The polar coordinates of the month January in year is (0.5, 1).  This allows the model to catch the similarities between January and December.  This information was not captured in the simple Date Transformer.


### Text Transformer

* transform text column using methods: TFIDF or count (count of the word)
* this may be followed by dimensionality reduction using truncated SVD

### Categorical Target Encoding Transformer

* cross validation target encoding done on a categorical column

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | CV\_TE\_State|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 |3 | 2 | NY | 700,000 | 550,000 | 

The average price of properties in NY state is $550,000*.  

*In order to prevent overfitting, Driverless AI calculates this average on out-of-fold data using cross validation.

### Numeric to Categorical Target Encoding Transformer

* numeric column converted to categorical by binning
* cross validation target encoding done on the binned numeric column

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | CV\_TE\_SquareFootage|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 345,000 | 

The column `Square Footage` has been bucketed into 10 equally populated bins. This property lies in the `Square Footage` bucket 1,572 to 1,749.  The average price of properties with this range of square footage is $345,000*.

*In order to prevent overfitting, Driverless AI calculates this average on out-of-fold data using cross validation.


### Cluster Target Encoding Transformer

* selected columns in the data are clustered
* target encoding is done on the cluster ID

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | ClusterTE\_4\_NumBeds\_NumBaths\_SquareFootage|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 450,000 | 

The columns: `Num Beds`, `Num Baths`, `Square Footage` have been segmented into 4 clusters.  The average price of properties in the same cluster as the selected property is $450,000*.

*In order to prevent overfitting, Driverless AI calculates this average on out-of-fold data using cross validation.


### Cluster Distance Transformer

* selected columns in the data are clustered
* the distance to a chosen cluster center is calculated

| Date Built | Square Footage | Num Beds | Num Baths | State | Price | ClusterDist\_4\_NumBeds\_NumBaths\_SquareFootage_1|
|---|---|---|---|---|---|---|
| 01/01/1920 | 1700 | 3 | 2 | NY | 700,000 | 0.83 | 

The columns: `Num Beds`, `Num Baths`, `Square Footage` have been segmented into 4 clusters.  The difference from this record to Cluster 1 is 0.83. 
