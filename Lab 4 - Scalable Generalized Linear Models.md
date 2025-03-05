# Lab 4: Scalable Generalized Linear Models

[COM6012 Scalable Machine Learning **2025**](https://github.com/COM6012/ScalableML) by [Shuo Zhou](https://shuo-zhou.github.io/) at The University of Sheffield

## Study schedule

- [Section 1](#1-data-types-in-rdd-based-api): To finish in the lab session on 6th March. **Essential**
- [Section 2](#2-glms-in-pyspark): To finish by 6th March. **Essential**
- [Section 3](#3-exercises): To finish before the next Tuesday 11th March. ***Exercise***
- [Section 4](#4-additional-exercise-optional): To explore further. *Optional*

### Suggested reading

- [Data Types - RDD-based API](https://spark.apache.org/docs/3.5.4/mllib-data-types.html)

## Introduction

Unlike linear regression, where the output is assumed to follow a Gaussian distribution, in [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) the response variable $y$ follows some distribution from the [exponential family of distributions](https://en.wikipedia.org/wiki/Exponential_family).

Before you continue, open a new terminal in [Stanage](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html), use the `rse-com6012-4` queue, and activate the **myspark** environment. First log into the Stanage cluster

```sh
ssh $USER@stanage.shef.ac.uk
```

You need to replace `$USER` with your username (using **lowercase** and without `$`). Once logged in, we can start an interactive session from reserved resources by

```sh
srun --account=rse-com6012 --reservation=rse-com6012-4 --time=01:00:00 --pty /bin/bash
```

if the reserved resources are not available, start an interactive session from the general queue by

```sh
srun --pty bash -i
```

Now set up our conda environment, using

```sh
source myspark.sh # assuming you copied HPC/myspark.sh to your root directory (see Lab 1 Task 2)
```

if you created a `myspark.sh` script in Lab 1.  If not, use

```sh
module load Java/17.0.4
```

```sh
module load Anaconda3/2024.02-1
```

```sh
source activate myspark
```

Remember to `cd` into your `ScalableML` directory before you start the PySpark shell by running the `pyspark` command.  

**NOTE:** If you are experiencing a `segmentation fault` when entering the `pyspark` interactive shell, run `export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` to fix it. It is recommended to add this line to your `myspark.sh` file.

## 1. Data Types in RDD-based API

To deal with data efficiently, Spark considers different [data types](https://spark.apache.org/docs/3.5.0/mllib-data-types.html). In particular, MLlib supports local vectors and matrices stored on a single machine, as well as distributed matrices backed by one or more RDDs. Local vectors and local matrices are simple data models that serve as public interfaces. The underlying linear algebra operations are provided by [Breeze](http://www.scalanlp.org/). A training example used in supervised learning is called a “labeled point” in MLlib.

### [Local vector](https://spark.apache.org/docs/3.5.0/mllib-data-types.html#local-vector):  Dense vs Sparse

> A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine. MLlib supports two types of local vectors: dense and sparse. A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0) can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector.

Check out the [Vector in RDD API](https://spark.apache.org/docs/3.5.0/api/python/reference/api/pyspark.mllib.linalg.Vectors.html?highlight=mllib%20linalg%20vectors#pyspark.mllib.linalg.Vectors) or [Vector in DataFrame API](https://spark.apache.org/docs/3.5.0/api/python/reference/api/pyspark.ml.linalg.Vector.html?highlight=ml%20linalg%20vector#pyspark.ml.linalg.Vector) (see method `.Sparse()`) and [SparseVector in RDD API](https://spark.apache.org/docs/3.5.0/api/python/reference/api/pyspark.mllib.linalg.SparseVector.html?highlight=sparsevector#pyspark.mllib.linalg.SparseVector) or [SparseVector in DataFrame API](https://spark.apache.org/docs/3.5.0/api/python/reference/api/pyspark.ml.linalg.SparseVector.html?highlight=sparsevector#pyspark.ml.linalg.SparseVector). The official example is below

```python
import numpy as np
from pyspark.mllib.linalg import Vectors

dv1 = np.array([1.0, 0.0, 3.0])  # Use a NumPy array as a dense vector.
dv2 = [1.0, 0.0, 3.0]  # Use a Python list as a dense vector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])  # Create a SparseVector.
```

Note the vector created by `Vectors.sparse()` is of type `SparseVector()`

```python
sv1
# SparseVector(3, {0: 1.0, 2: 3.0})
```

To view the sparse vector in a dense format

```python
sv1.toArray()
# array([1., 0., 3.])
```

### [Labeled point](https://spark.apache.org/docs/3.5.0/mllib-data-types.html#labeled-point)

> A labeled point is a local vector, either dense or sparse, associated with a label/response. In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label, so we can use labeled points in both regression and classification. For binary classification, a label should be either 0 (negative) or 1 (positive). For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....

See [LabeledPoint API in MLlib](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.regression.LabeledPoint.html?highlight=labeledpoint#pyspark.mllib.regression.LabeledPoint). Now, we create a labeled point with a positive label and a dense feature vector, as well as a labeled point with a negative label and a sparse feature vector.

```python
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

neg
# LabeledPoint(0.0, (3,[0,2],[1.0,3.0]))
neg.label
# 0.0
neg.features
# SparseVector(3, {0: 1.0, 2: 3.0})
```

Now view the features as dense vector (rather than sparse vector)

```python
neg.features.toArray()
# array([1., 0., 3.])
```

### [Local matrix](https://spark.apache.org/docs/3.5.0/mllib-data-types.html#local-matrix)

> A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine. MLlib supports dense matrices, whose entry values are stored in a single double array in column-major order, and sparse matrices, whose non-zero entry values are stored in the Compressed Sparse Column (CSC) format in column-major order. For example, we create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)) and a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0)) in the following:

```python
from pyspark.mllib.linalg import Matrix, Matrices

dm2 = Matrices.dense(3, 2, [1, 3, 5, 2, 4, 6]) 
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print(dm2)
# DenseMatrix([[1., 2.],
#              [3., 4.],
#              [5., 6.]])
print(sm)
# 3 X 2 CSCMatrix
# (0,0) 9.0
# (2,1) 6.0
# (1,1) 8.0
```

See [Scala API for Matrices.sparse](https://spark.apache.org/docs/3.5.0/api/scala/org/apache/spark/mllib/linalg/Matrices$.html) and from its [source code](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/linalg/Matrices.scala), we can see it creates a CSC [SparseMatrix](https://spark.apache.org/docs/3.5.0/api/scala/org/apache/spark/mllib/linalg/SparseMatrix.html).

Here the [compressed sparse column (CSC or CCS) format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) is used for sparse matrix representation. You can learn it from this [simple explanation](https://stackoverflow.com/questions/44825193/how-to-create-a-sparse-cscmatrix-using-spark?answertab=votes#tab-top). To learn more about CSC, you may refer to a [top video](https://www.youtube.com/watch?v=fy_dSZb-Xx8) and a [top post with animation](https://matteding.github.io/2019/04/25/sparse-matrices/#compressed-sparse-matrices).
> values are read first by column, a row index is stored for each value, and column pointers are stored. For example, CSC is (val, row_ind, col_ptr), where val is an array of the (top-to-bottom, then left-to-right) non-zero values of the matrix; row_ind is the row indices corresponding to the values; and, col_ptr is the list of val indexes where each column starts.

```python
dsm=sm.toDense()
print(dsm)
# DenseMatrix([[9., 0.],
#              [0., 8.],
#              [0., 6.]])
```

### [Distributed matrix](https://spark.apache.org/docs/3.5.0/mllib-data-types.html#distributed-matrix)

> A distributed matrix has long-typed row and column indices and double-typed values, stored distributively in one or more RDDs. It is very important to choose the right format to store large and distributed matrices. Converting a distributed matrix to a different format may require a global shuffle, which is quite expensive. Four types of distributed matrices have been implemented so far.

#### RowMatrix

> The basic type is called RowMatrix. A RowMatrix is a row-oriented distributed matrix without meaningful row indices, e.g., a collection of feature vectors. It is backed by an RDD of its rows, where each row is a local vector. We assume that the number of columns is not huge for a RowMatrix so that a single local vector can be reasonably communicated to the driver and can also be stored / operated on using a single node.
> Since each row is represented by a local vector, the number of columns is limited by the integer range but it should be much smaller in practice.

Now we create an RDD of vectors `rows`, from which we create a RowMatrix `mat`.

```python
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mat = RowMatrix(rows)

m = mat.numRows()  # Get its size: m=4, n=3
n = mat.numCols()  

rowsRDD = mat.rows  # Get the rows as an RDD of vectors again.
```

We can view the RowMatrix in a dense matrix format

```python
rowsRDD.collect()
# [DenseVector([1.0, 2.0, 3.0]), DenseVector([4.0, 5.0, 6.0]), DenseVector([7.0, 8.0, 9.0]), DenseVector([10.0, 11.0, 12.0])]
```

## 2. GLMs in PySpark

In this Lab, we will look at Poisson regression over the [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). We will compare the performance of several models and algorithms on this dataset, including: Poisson Regression, Linear Regression implemented with IRLS and Linear Regression with general regularization.

We start by loading the data. Here we use the hourly recordings only.

```python
rawdata = spark.read.csv('./Data/hour.csv', header=True)
rawdata.cache()
```

```bash
DataFrame[instant: string, dteday: string, season: string, yr: string, mnth: string, hr: string, holiday: string, weekday: string, workingday: string, weathersit: string, temp: string, atemp: string, hum: string, windspeed: string, casual: string, registered: string, cnt: string]
```

The following is a description of the features

- **instant**: record index
- **dteday**: date
- **season**: season (1:springer, 2:summer, 3:fall, 4:winter)
- **yr**: year (0: 2011, 1:2012)
- **mnth**: month ( 1 to 12)
- **hr**: hour (0 to 23)
- **holiday**: weather day is holiday or not
- **weekday**: day of the week
- **workingday**: if day is neither weekend nor holiday is 1, otherwise is 0.

- **weathersit**:
  - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

- **temp**: Normalized temperature in Celsius. The values are divided to 41 (max)
- **atemp**: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- **hum**: Normalized humidity. The values are divided to 100 (max)
- **windspeed**: Normalized wind speed. The values are divided to 67 (max)
- **casual**: count of casual users
- **registered**: count of registered users
- **cnt**: count of total rental bikes including both casual and registered

From the above, we want to use the features season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum and windspeed to predict cnt.

```python
schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])
```

Transform to DoubleType

```python
new_schemaNames = new_rawdata.schema.names
from pyspark.sql.types import DoubleType
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))
```

```python
new_rawdata.printSchema()
```

```bash
root
  |-- season: double (nullable = true)
  |-- yr: double (nullable = true)
  |-- mnth: double (nullable = true)
  |-- hr: double (nullable = true)
  |-- holiday: double (nullable = true)
  |-- weekday: double (nullable = true)
  |-- workingday: double (nullable = true)
  |-- weathersit: double (nullable = true)
  |-- temp: double (nullable = true)
  |-- atemp: double (nullable = true)
  |-- hum: double (nullable = true)
  |-- windspeed: double (nullable = true)
  |-- casual: double (nullable = true)
  |-- registered: double (nullable = true)
  |-- cnt: double (nullable = true)
```

We now create the training and test data

```python
(trainingData, testData) = new_rawdata.randomSplit([0.7, 0.3], 42)
```

```python
new_schemaNames[0:new_ncolumns-3]
```

```bash
['season',
 'yr',
 'mnth',
 'hr',
 'holiday',
 'weekday',
 'workingday',
 'weathersit',
 'temp',
 'atemp',
 'hum',
 'windspeed']
```

And assemble the features into a vector

```python
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = new_schemaNames[0:new_ncolumns-3], outputCol = 'features') 
```

We now want to proceed to apply Poisson Regression over our dataset. We will use the [GeneralizedLinearRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html) model for which we can set the following parameters

<!-- > **maxIter**: max number of iterations.<p>
    **regParameter**: regularization parameter (>= 0). By setting this parameter to be >0 we are applying an $\ell_2$ regularization.<p>
**familiy**: The name of family which is a description of the error distribution to be used in the model. Supported options: gaussian (default), binomial, poisson, gamma and tweedie.<p>
    **link**: The name of link function which provides the relationship between the linear predictor and the mean of the distribution function. Supported options: identity, log, inverse, logit, probit, cloglog and sqrt. <p>
    The Table below shows the combinations of **family** and **link** functions allowed in this version of PySpark.<p> -->

> **maxIter**: max number of iterations.  
> **regParameter**: regularization parameter (≥ 0). By setting this parameter to be > 0, we are applying an ℓ₂ regularization.  
> **family**: The name of the family, which is a description of the error distribution to be used in the model.  
> Supported options: `gaussian` (default), `binomial`, `poisson`, `gamma`, and `tweedie`.  
> **link**: The name of the link function, which provides the relationship between the linear predictor and the mean of the distribution function.  
> Supported options: `identity`, `log`, `inverse`, `logit`, `probit`, `cloglog`, and `sqrt`.  
> The table below shows the combinations of **family** and **link** functions allowed in this version of PySpark.

<!-- <table>
<tr><td><b>Family</b></td><td><b>Response Type</b></td><td><b>Supported Links</b></td></tr>
<tr><td>Gaussian</td><td>Continuous</td><td>Identity, Log, Inverse</td></tr>
<tr><td>Binomial</td><td>Binary</td><td>Logit, Probit, CLogLog</td></tr>
<tr><td>Poisson</td><td>Count</td><td>Log, Identity, Sqrt</td></tr>
<tr><td>Gamma</td><td>Continuous</td><td>Inverse, Identity, Log</td></tr>
<tr><td>Tweedie</td><td>Zero-inflated continuous</td><td>Power link function</td></tr>
</table> -->

| Family   | Response Type                | Supported Links                  |
|----------|-----------------------------|----------------------------------|
| Gaussian | Continuous                   | Identity, Log, Inverse          |
| Binomial | Binary                        | Logit, Probit, CLogLog          |
| Poisson  | Count                         | Log, Identity, Sqrt             |
| Gamma    | Continuous                    | Inverse, Identity, Log          |
| Tweedie  | Zero-inflated continuous      | Power link function             |

```python
from pyspark.ml.regression import GeneralizedLinearRegression
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
```

We now create a Pipeline

```python
from pyspark.ml import Pipeline
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)
```

We fit the pipeline to the dataset

```python
pipelineModel = pipeline.fit(trainingData)
```

We now evaluate the RMSE

```python
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)
```

```bash
RMSE = 142.214 
```

```python
pipelineModel.stages[-1].coefficients
```

```bash
DenseVector([0.133, 0.4267, 0.002, 0.0477, -0.1086, 0.005, 0.015, -0.0633, 0.7031, 0.6608, -0.746, 0.2307])
```

## 3. Exercises

**Note**: A *reference* solution will be provided in for this part by the following Thursday.

### Exercise 1

The variables season, yr, mnth, hr, holiday, weekday, workingday and weathersit are categorical variables that have been treated as continuous variables. In general this is not optimal since we are indirectly imposing a geometry or order over a variable that does not need to have such geometry. For example, the variable season takes values 1 (spring), 2 (summer), 3 (fall) and 4 (winter). Indirectly, we are saying that the distance between spring and winter (1 and 4) is larger than the distance between spring (1) and summer (3). There is not really a reason for this. To avoid this imposed geometries over variables that do not follow one, the usual approach is to transform categorical features to a representation of one-hot encoding. Use the [OneHotEncoder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html) estimator over the Bike Sharing Dataset to represent the categorical variables. Using the same training and test data compute the RMSE over the test data using the same Poisson model.

### Exercise 2

Compare the performance of Linear Regression over the same dataset using the following algorithms:

1. Linear Regression using $\ell_1$ regularization and optimization OWL-QN.
2. Linear Regression using elasticNet regularization and optimization OWL-QN.
3. Linear Regression using $\ell_2$ regularization and optimization L-BFGS.
4. Linear Regression using $\ell_2$ regularization and optimization IRLS.

## 4. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

The type of features used for regression can have a dramatic influence over the performance. When we use one-hot encoding for the categorical features, the prediction error of the Poisson regression reduces considerable (see Question 1). We could further preprocess the features to see how the preprocessing can influence the performance. Test the performance of Poisson regression and the Linear Regression models in Question 2 when the continuous features are standardized (the mean of each feature is made equal to zero and the standard deviation is equal to one). Standardization is performed over the training data only, and the means and standard deviations computed over the training data are later used to standardize the test data.
