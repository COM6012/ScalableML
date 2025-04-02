# Lab 8: Matrix Factorization for Collaborative Filtering in Recommender Systems and PCA for Dimensionality Reduction

[COM6012 Scalable Machine Learning **2025**](https://github.com/COM6012/ScalableML) by Shuo Zhou, Xianyuan Liu, and Haiping Lu, 3rd April 2025

## Study schedule

- [Task 1](#1-movie-recommendation-via-collaborative-filtering): To finish in the lab session on 3rd April . **Essential**
- [Task 2](#2-pca): To finish in the lab session on 3rd April . **Essential**
- [Task 3](#3-exercises): To finish by 29th April. ***Exercise***
- [Task 4](#4-additional-ideas-to-explore-optional): To explore further. *Optional*

### Suggested reading

- [Collaborative Filtering in Spark](https://spark.apache.org/docs/3.5.4/ml-collaborative-filtering.html)
- [DataBricks movie recommendations tutorial](https://github.com/databricks/spark-training/blob/master/website/movie-recommendation-with-mllib.md). [DataBricks](https://en.wikipedia.org/wiki/Databricks) is a company founded by the creators of Apache Spark. Check out their packages at [their GitHub page](https://github.com/databricks). They offer a free (up to 15GB memory) cloud computing platform [Databricks Community Edition](https://community.cloud.databricks.com/login.html) that you can try out.
- [Collaborative Filtering on Wikipedia](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering)
- [Python API on ALS for recommender system](https://spark.apache.org/docs/3.5.4/api/python/reference/api/pyspark.ml.recommendation.ALS.html)
- Chapter 16 *ALS: Stock Portfolio Recommendations* (particularly Section *Demo*) of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf)
- [Extracting, transforming and selecting features](https://spark.apache.org/docs/3.5.4/ml-features.html)
- [PCA in Spark DataFrame API `pyspark.ml`](https://spark.apache.org/docs/3.5.4/ml-features.html#pca)
- [SVD in Spark RDD API `pyspark.mllib`](https://spark.apache.org/docs/3.5.4/mllib-dimensionality-reduction.html#singular-value-decomposition-svd)
- [StandardScaler in Spark](https://spark.apache.org/docs/3.5.4/ml-features.html#standardscaler) to standardize/normalize data to unit standard deviation and/or zero mean.
- [PCA on Wiki](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Understanding Dimension Reduction with Principal Component Analysis (PCA)](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
- [Principal Component Analysis explained on Kaggle](https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained) with data available [here](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv), and background info [here](https://rstudio-pubs-static.s3.amazonaws.com/345463_37f54d1c948b4cdfa181541841e0db8a.html)

### PySpark APIs in pictures (highly recommended)

- [**Learn PySpark APIs via Pictures**](https://github.com/jkthompson/pyspark-pictures) (**from recommended repositories** in GitHub, i.e., found via **recommender systems**!)

## 1. Movie recommendation via collaborative filtering

### Getting started

First log into the Stanage cluster

```sh
ssh $USER@stanage.shef.ac.uk
```

You need to replace `$USER` with your username (using **lowercase** and without `$`).

Once logged in, we can request 2 cpu cores from reserved resources by

```sh
srun --account=rse-com6012 --reservation=rse-com6012-8 --cpus-per-task=2 --time=01:00:00 --pty /bin/bash
```

if the reserved resources are not available, request core from the general queue by

```sh
srun --pty --cpus-per-task=2 bash -i
```

Now set up our conda environment, using

```sh
source myspark.sh # assuming you copied HPC/myspark.sh to your root directory (see Lab 1 Task 2)
```

if you created a `myspark.sh` script in Lab 1.  If not, use

```sh
module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark
```

Now we can start the PySpark shell by

```sh
cd com6012/ScalableML # our main working directory
pyspark --master local[2] # start pyspark with the 2 cpu cores requested above.
```

If you are experiencing a `segmentation fault` when entering the `pyspark` interactive shell, run `export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` to fix it. It is recommended to add this line to your `myspark.sh` file.

### Collaborative filtering

[Collaborative filtering](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering) is a classic approach for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix primarily based on the matrix *itself*.  `spark.ml` currently supports **model-based** collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries, using the **alternating least squares (ALS)** algorithm.

[API](https://spark.apache.org/docs/3.5.4/api/python/reference/api/pyspark.ml.recommendation.ALS.html): `class pyspark.ml.recommendation.ALS(*, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False, alpha=1.0, userCol='user', itemCol='item', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=10, intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='nan', blockSize=4096)`

The following parameters are available:

- *rank*: the number of latent factors in the model (defaults to 10).
- *maxIter* is the maximum number of iterations to run (defaults to 10).
- *regParam*: the regularization parameter in ALS (defaults to 0.1).
- *numUserBlocks*/*numItemBlocks*: the number of blocks the users and items will be partitioned into in order to parallelize computation (defaults to 10).
- *implicitPrefs*: whether to use the explicit feedback ALS variant or one adapted for implicit feedback data (defaults to false which means using explicit feedback).
- *alpha*: a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations (defaults to 1.0).
- *nonnegative*: whether or not to use nonnegative constraints for least squares (defaults to false).
- *coldStartStrategy*: can be set to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values (defaults to "nan", assigning NaN to a user and/or item factor is not present in the model.
- *blockSize*: the size of the user/product blocks in the blocked implementation of ALS (to reduce communication).

### Movie recommendation

In the cells below, we present a small example of collaborative filtering with the data taken from the [MovieLens](http://grouplens.org/datasets/movielens/) project. Here, we use the old 100k dataset, which has been downloaded in the `Data` folder but you are encouraged to view the source.

The dataset looks like this:

```markdown
    196     242     3       881250949
    186     302     3       891717742
    22      377     1       878887116
    244     51      2       880606923
```

This is a **tab separated** list of

```markdown
    user id | item id | rating | timestamp 
```

#### Explicit vs. implicit feedback

The data above is typically viewed as a user-item matrix with the ratings as the entries and users and items determine the row and column indices. The ratings are **explicit feedback**. The *Mean Squared Error* of rating prediction can be used to evaluate the recommendation model.

The ratings can also be used differently. We can treat them as numbers representing the strength in observations of user actions, i.e., as **implicit feedback** similar to the number of clicks, or the cumulative duration someone spent viewing a movie. Such numbers are then related to the level of confidence in observed user preferences, rather than explicit ratings given to items. The model then tries to find latent factors that can be used to predict the expected preference of a user for an item.

#### Cold-start problem

The cold-start problem refers to the cases when some users and/or items in the test dataset were not present during training the model. In Spark, these users and items are either assigned `NaN` (not a number, default) or dropped (option `drop`).

#### MovieLens100k

Let's study ALS for movie recommendation on the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/).

```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
```

Read the data in and split words (tab separated):

```python
lines = spark.read.text("Data/MovieLens100k.data").rdd
parts = lines.map(lambda row: row.value.split("\t"))
```

We need to convert the text (`String`) into numbers (`int` or `float`) and then convert RDD to DataFrame:

```python
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD).cache()
```

Check data:

```python
ratings.show(5)
# +------+-------+------+---------+
# |userId|movieId|rating|timestamp|
# +------+-------+------+---------+
# |   196|    242|   3.0|881250949|
# |   186|    302|   3.0|891717742|
# |    22|    377|   1.0|878887116|
# |   244|     51|   2.0|880606923|
# |   166|    346|   1.0|886397596|
# +------+-------+------+---------+
# only showing top 5 rows
```

Check data type:

```python
ratings.printSchema()
# root
#  |-- userId: long (nullable = true)
#  |-- movieId: long (nullable = true)
#  |-- rating: double (nullable = true)
#  |-- timestamp: long (nullable = true)
```

Prepare the training/test data with seed `6012`:

```python
myseed=6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()
```

Build the recommendation model using ALS on the training data. Note we set cold start strategy to `drop` to ensure we don't get NaN evaluation metrics:

```python
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model = als.fit(training)
# DATE TIME WARN InstanceBuilder$NativeBLAS: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
# DATE TIME WARN InstanceBuilder$NativeBLAS: Failed to load implementation from:dev.ludovic.netlib.blas.ForeignLinkerBLAS
# DATE TIME WARN InstanceBuilder$NativeLAPACK: Failed to load implementation from:dev.ludovic.netlib.lapack.JNILAPACK
```

The warnings on BLAS and LAPACK are about [optimized numerical processing](https://spark.apache.org/docs/3.5.4/ml-guide.html#dependencies). The warning messages mean that a pure JVM implementation will be used instead of the optimized ones, which need to be [installed separately](https://spark.apache.org/docs/3.5.4/ml-linalg-guide.html). We are not installing them in this module but if you may try on your own machine (not HPC due to access right) if interested.

Evaluate the model by computing the RMSE on the test data:

```python
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
# Root-mean-square error = 0.9209573069829078
```

Generate top 10 movie recommendations for each user:

```python
userRecs = model.recommendForAllUsers(10)
userRecs.show(5,  False)
# +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |userId|recommendations                                                                                                                                                                         |
# +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |1     |[{169, 4.9709697}, {1142, 4.854675}, {1449, 4.797047}, {408, 4.7874727}, {119, 4.7250085}, {50, 4.707899}, {12, 4.657482}, {1122, 4.648316}, {178, 4.6348457}, {884, 4.6306896}]        |
# |3     |[{1643, 4.7289743}, {1463, 4.2027235}, {320, 4.1247706}, {1245, 4.038789}, {701, 3.8892932}, {347, 3.88081}, {1169, 3.819116}, {1605, 3.7705793}, {179, 3.7530453}, {1099, 3.6847708}]  |
# |5     |[{1589, 4.7718945}, {1643, 4.6552463}, {1463, 4.6354938}, {1240, 4.6032286}, {114, 4.5991983}, {169, 4.5080156}, {1462, 4.504652}, {408, 4.466927}, {613, 4.4491153}, {1367, 4.4088354}]|
# |6     |[{1463, 5.1079006}, {1643, 4.6280446}, {1367, 4.5027647}, {474, 4.442402}, {1203, 4.3943367}, {1449, 4.3834624}, {641, 4.356251}, {1142, 4.3523912}, {483, 4.3490787}, {647, 4.338783}] |
# |9     |[{1449, 5.2568903}, {1664, 5.2359605}, {197, 4.991274}, {1064, 4.9872465}, {963, 4.944022}, {663, 4.891681}, {1122, 4.8614693}, {884, 4.8461266}, {506, 4.8275256}, {936, 4.818603}]    |
# +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 5 rows
```

Generate top 10 user recommendations for each movie

```python
movieRecs = model.recommendForAllItems(10)
movieRecs.show(5, False)
# +-------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |movieId|recommendations                                                                                                                                                                 |
# +-------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |1      |[{810, 5.3465643}, {688, 5.0386543}, {137, 4.928752}, {507, 4.902011}, {477, 4.860208}, {366, 4.8342547}, {887, 4.8263907}, {849, 4.8156605}, {939, 4.779002}, {357, 4.7542334}]|
# |3      |[{324, 4.516063}, {372, 4.2431326}, {270, 4.219804}, {355, 4.1576357}, {261, 4.142714}, {770, 4.13035}, {174, 4.109981}, {907, 4.067356}, {347, 4.06455}, {859, 4.0524206}]     |
# |5      |[{811, 4.820167}, {341, 4.63373}, {849, 4.619356}, {628, 4.6059885}, {507, 4.5798445}, {907, 4.5308404}, {810, 4.508147}, {164, 4.502011}, {688, 4.497636}, {939, 4.464517}]    |
# |6      |[{341, 5.527562}, {34, 5.3643284}, {531, 5.175154}, {558, 5.095579}, {180, 5.0328345}, {770, 4.989878}, {777, 4.960204}, {628, 4.937691}, {717, 4.9131093}, {928, 4.8199334}]   |
# |9      |[{688, 5.235702}, {34, 5.0346103}, {219, 5.024946}, {173, 4.9955764}, {928, 4.9674325}, {628, 4.9368606}, {556, 4.839726}, {147, 4.83116}, {204, 4.824109}, {252, 4.8147955}]   |
# +-------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 5 rows
```

Generate top 10 movie recommendations for a specified set of users:

```python
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
users.show()
# +------+
# |userId|
# +------+
# |    26|
# |    29|
# |   474|
# +------+
userSubsetRecs.show(3,False)
# --------------------------------------------------------------------------+
# |userId|recommendations                                                                                                                                                                    |
# +------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |474   |[{884, 4.9401073}, {318, 4.885294}, {1449, 4.8396335}, {1122, 4.811585}, {408, 4.7511024}, {64, 4.73949}, {357, 4.7208233}, {1643, 4.713397}, {127, 4.7032666}, {1064, 4.6970286}] |
# |26    |[{1449, 4.0505023}, {884, 3.9505656}, {1122, 3.937915}, {1643, 3.93372}, {408, 3.9252203}, {483, 3.8798523}, {318, 3.8776188}, {114, 3.8497207}, {119, 3.8116817}, {127, 3.810518}]|
# |29    |[{1449, 4.656543}, {884, 4.630061}, {963, 4.581265}, {272, 4.550117}, {408, 4.5260196}, {114, 4.4871216}, {318, 4.4804883}, {483, 4.4729433}, {64, 4.456217}, {1122, 4.4341135}]   |
# +------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Generate top 10 user recommendations for a specified set of movies

```python
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movies.show()
# +-------+
# |movieId|
# +-------+
# |    474|
# |     29|
# |     26|
# +-------+
movieSubSetRecs.show(3,False)
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |movieId|recommendations                                                                                                                                                                  |
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |474    |[{928, 5.138258}, {810, 5.0860453}, {173, 5.057652}, {239, 5.0229735}, {794, 4.9939513}, {747, 4.9605308}, {310, 4.947432}, {686, 4.904605}, {339, 4.8961563}, {118, 4.8948402}] |
# |26     |[{270, 4.4144955}, {341, 4.3995667}, {366, 4.3244834}, {770, 4.249518}, {118, 4.2322574}, {414, 4.2000494}, {274, 4.184001}, {923, 4.1715975}, {173, 4.168408}, {180, 4.1619034}]|
# |29     |[{127, 4.425912}, {507, 4.315676}, {427, 4.2933187}, {811, 4.264675}, {472, 4.193143}, {628, 4.180931}, {534, 4.071936}, {907, 4.02065}, {939, 3.993926}, {677, 3.9678879}]      |
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Let us take a look at the learned factors:

```python
dfItemFactors=model.itemFactors
dfItemFactors.show()
# +---+--------------------+
# | id|            features|
# +---+--------------------+
# | 10|[0.19747244, 0.65...|
# | 20|[0.7627245, 0.401...|
# | 30|[0.67989284, 0.54...|
# | 40|[0.30699247, 0.47...|
# | 50|[-0.1456875, 1.03...|
# | 60|[0.94835776, 1.10...|
# | 70|[0.38704985, 0.29...|
# | 80|[-0.34140116, 1.0...|
# | 90|[0.2525186, 1.206...|
# |100|[0.5523014, 0.688...|
# |110|[0.018842308, 0.2...|
# |120|[0.044651575, 0.1...|
# |130|[-0.22696877, -0....|
# |140|[-0.10749633, 0.1...|
# |150|[0.5518842, 1.098...|
# |160|[0.44457805, 0.80...|
# |170|[0.54556036, 0.77...|
# |180|[0.42527583, 0.70...|
# |190|[0.29437917, 0.78...|
# |200|[0.3496798, 0.443...|
# +---+--------------------+
# only showing top 20 rows
```

## 2. PCA

[Principal component analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called **principal components (PCs)**. A PCA class trains a model to project vectors to a low-dimensional space using PCA and this is probably the most commonly used **dimensionality reduction** method.

### PCA in DataFrame-based API `pyspark.ml`  

Check out the [API](https://spark.apache.org/docs/3.5.4/api/python/reference/api/pyspark.ml.feature.PCA.html?highlight=pyspark%20ml%20feature%20pca#pyspark.ml.feature.PCA). Check [`pyspark.ml.feature.PCAModel`](https://spark.apache.org/docs/3.5.4/api/python/reference/api/pyspark.ml.feature.PCAModel.html?highlight=pyspark%20ml%20feature%20pcamodel#pyspark.ml.feature.PCAModel) too to see what is available for the fitted model. Let us project three 5-dimensional feature vectors into 2-dimensional principal components.

```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
df.show()
# +--------------------+
# |            features|
# +--------------------+
# | (5,[1,3],[1.0,7.0])|
# |[2.0,0.0,3.0,4.0,...|
# |[4.0,0.0,0.0,6.0,...|
# +--------------------+

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
# +----------------------------------------+
# |pcaFeatures                             |
# +----------------------------------------+
# |[1.6485728230883807,-4.013282700516296] |
# |[-4.645104331781534,-1.1167972663619026]|
# |[-6.428880535676489,-5.337951427775355] |
# +----------------------------------------+
```

Check the explained variance in percentage

```python
model.explainedVariance
# DenseVector([0.7944, 0.2056])
```

Take a look at the principal components Matrix. Each column is one principal component.

```python
 print(model.pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])
```

### PCA in RDD-based API `pyspark.mllib`

#### Eigendecomposition for PCA

`pyspark.mllib` supports PCA for **tall-and-skinny** (big $n$, small $d$) matrices stored in row-oriented format and any Vectors. We demonstrate how to compute principal components on a [RowMatrix](http://spark.apache.org/docs/3.5.4/mllib-data-types.html#rowmatrix) and use them to project the vectors into a low-dimensional space in the cell below.

```python
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])
rows.collect()
# [SparseVector(5, {1: 1.0, 3: 7.0}), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]

mat = RowMatrix(rows)
```

Compute the top 2 principal components, which are stored in a local dense matrix (the same as above).

```python
pc = mat.computePrincipalComponents(2)
print(pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])
```

Project the rows to the linear space spanned by the top 2 principal components (the same as above)

```python
projected = mat.multiply(pc)
projected.rows.collect()
# [DenseVector([1.6486, -4.0133]), DenseVector([-4.6451, -1.1168]), DenseVector([-6.4289, -5.338])]
```

Now we convert to dense rows to see the matrix

```python
from pyspark.mllib.linalg import DenseVector

denseRows = rows.map(lambda vector: DenseVector(vector.toArray()))
denseRows.collect()
# [DenseVector([0.0, 1.0, 0.0, 7.0, 0.0]), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]
```

#### SVD for PCA  - more *scalable* way to do PCA

Read [SVD in RDD-based API `pyspark.mllib`](https://spark.apache.org/docs/3.5.4/mllib-dimensionality-reduction.html#singular-value-decomposition-svd). As covered in the lecture, we will need SVD for PCA on large-scale data. Here, we use it on the same small toy example to examine the relationship with eigenvalue decomposition based PCA methods above.

We compute the top 2 singular values and corresponding singular vectors.

```python
svd = mat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.
```

If we are doing it right, the **right** singular vectors should be the same as the eigenvectors.

```python
print(V)
# DenseMatrix([[-0.31278534,  0.31167136],
#              [-0.02980145, -0.17133211],
#              [-0.12207248,  0.15256471],
#              [-0.71847899, -0.68096285],
#              [-0.60841059,  0.62170723]])
```

But it is **not the same**! Why? Remeber that we need to do **centering**! We can do so use the [StandardScaler (check out the API](https://spark.apache.org/docs/3.5.4/mllib-feature-extraction.html#standardscaler)) to center the data, i.e., remove the mean.

```python
from pyspark.mllib.feature import StandardScaler

standardizer = StandardScaler(True, False)
model = standardizer.fit(rows)
centeredRows = model.transform(rows)
centeredRows.collect()
# [DenseVector([-2.0, 0.6667, -1.0, 1.3333, -4.0]), DenseVector([0.0, -0.3333, 2.0, -1.6667, 1.0]), DenseVector([2.0, -0.3333, -1.0, 0.3333, 3.0])]
centeredmat = RowMatrix(centeredRows)
```

Compute the top 2 singular values and corresponding singular vectors.

```python
svd = centeredmat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.
```

Check the **PC** obtained this time (it is the same as the above PCA methods now)

```python
print(V)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])
```

Let us examine the relationships between the singular values and the eigenvalues.

```python
print(s)
# [6.001041088520536,3.0530049438580336]
```

We get the eigenvalues by taking squares of the singular values

```python
evs=s*s
print(evs)
# [36.012494146111734,9.320839187221594]
```

Now we compute the percentage of variance captures and compare with the above to verify (see/search `model.explainedVariance`).

```python
evs/sum(evs)
# DenseVector([0.7944, 0.2056])
```

## 3. Exercises

### More movie recommendations

Do the following on HPC. Run your code in batch mode to produce the results.

1. Download the MovieLens [ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) dataset using `wget` as in Lab 2 exercises to the `ScalableML/Data` directory on HPC. Use the `unzip` command to unzip the files to a directory of your choice (search "unzip linux" to see examples of usage). Read the [readme](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) for this dataset to understand the data.
2. Use `ALS` to learn five recommendation models on this dataset, using the same split ratio (`0.8, 0.2`) and seed (`6012`) as above but five different values of the `rank` parameter: 5, 10, 15, 20, 25. Plot the (five) resulting RMSE values (on the test set) against the five rank values.
3. Find the top five movies to recommend to any one user of your choice and display the titles and genres for these five movies (via programming).

### PCA on iris

Study the [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) `iris.csv` under `Data` with PCA.

1. Follow [Understanding Dimension Reduction with Principal Component Analysis (PCA)](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/) to do the same analysis using the DataFrame-based PCA `pca.fit()` from `pyspark.ml`.
2. Follow this lab to verify that using the other two RDD-based PCA APIs `computePrincipalComponents` and `computeSVD` will give the same PCA features.

## 4. Additional ideas to explore (*optional*)

### Databricks tutorial

- Complete the tasks in the [quiz provided by DataBricks](https://github.com/databricks/spark-training/blob/master/machine-learning/python/MovieLensALS.py) on their data or the data from MovieLens directly. [A solution](https://github.com/databricks/spark-training/blob/master/machine-learning/python/solution/MovieLensALS.py) is available but you should try before consulting the solution.

### Santander Kaggle competition on produce recommendation

- A Kaggle competition on [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) with a prize of **USD 60,000**, and **1,779 teams** participating.
- Follow this [PySpark notebook on an ALS-based solution](https://www.elenacuoco.com/2016/12/22/alternating-least-squares-als-spark-ml/).
- Learn the way to consider **implicit preferences** and do the same for other recommendation problems.

### Stock Portfolio Recommendations

- Follow Chapter *ALS: Stock Portfolio Recommendations* of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf)  to perform [Stock Portfolio Recommendations](https://en.wikipedia.org/wiki/Portfolio_investment))
- The data can be downloaded from [Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/online+retail) at UCI.
- Pay attention to the **data cleaning** step that removes rows containing null value. You may need to do the same when you are dealing with real data.
- The data manipulation steps are useful to learn.

### Context-aware recommendation and time-split evaluation

- See the method in [Joint interaction with context operation for collaborative filtering](https://www.sciencedirect.com/science/article/pii/S0031320318304242?dgcid=rss_sd_all) and implement it in PySpark.
- Perform the **time split recommendation** as discussed in the paper for the above recommender systems.

### [HR analytics](https://rstudio-pubs-static.s3.amazonaws.com/345463_37f54d1c948b4cdfa181541841e0db8a.html)

A company is trying to figure out why their best and experienced employees are leaving prematurely from a [dataset](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv). Follow the example [Principal Component Analysis explained on Kaggle](https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained) to perform such analysis in PySpark, using as many PySpark APIs as possible.

### Word meaning extraction

Use PySpark to perform the steps in IBM's notebook on [Spark-based machine learning for word meanings](https://github.com/IBMDataScience/word2vec/blob/master/Spark-based%20machine%20learning%20for%20word%20meanings.ipynb) that makes use of PCA, kmeans, and Word2Vec to learn word meanings.

### Bag of words analysis

Choose a [Bag of Words Data Set](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words). Let us take  the **NIPS full papers** data as an example.

The format of this data is

```markdown
    Number of documents
    Number of words in the vocabulary
    Total number of words in the collection
    docID wordID count
    docID wordID count
    ...
    docID wordID count
```

Our data matrix will be `doc` $\times$ `wordcount`. To begin, we need to read this data in. Possible steps would include:

1. extract the number of documents and the size of the vocabulary, and strip off the first 3 lines
2. combine the words per document
3. create sparse vectors (for better space efficiency)

Start from a small dataset to test your work, and then checking **whether** your work scales up to the big **NYTIMES** bagofwords data. Keep everything as parallel as possible.

### Large image datasets

Find some large-scale image datasets to examine the principal components and explore low-dimensional representations.
