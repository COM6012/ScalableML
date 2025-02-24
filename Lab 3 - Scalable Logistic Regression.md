# Lab 3: Scalable logistic regression

[COM6012 Scalable Machine Learning **2025**](https://github.com/COM6012/ScalableML) by [Shuo Zhou](https://shuo-zhou.github.io/) at The University of Sheffield

## Study schedule

- [Section 1](#1-spark-configuration): To finish by 27th February. **Essential**
- [Section 2](#2-logistic-regression-in-pyspark): To finish by 27th February. **Essential**
- [Section 3](#3-exercises): To finish by the following Tuesday 4th March. ***Exercise***
- [Section 4](#4-additional-exercise-optional): To explore further. *Optional*

## Introduction

**Dependencies.** For this lab, we need to install the ``matplotlib`` and `pandas` packages. Make sure you install the packages in the environment **myspark**

Before you continue, open a new terminal in [Stanage](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html), use the `rse-com6012-3` queue with two nodes, and activate the **myspark** environment. First log into the Stanage cluster

```sh
ssh $USER@stanage.shef.ac.uk
```

You need to replace `$USER` with your username (using **lowercase** and without `$`). Once logged in, we can start an interactive session from reserved resources by

```sh
srun --account=rse-com6012 --reservation=rse-com6012-3 --time=01:00:00 --pty /bin/bash
```

if the reserved resources are not available, start an interactive session from the general queue by

```sh
srun --pty bash -i
```

Now set up our conda environment, using

```sh
source myspark.sh # assuming you copied HPC/myspark.sh to your root directory (see Lab 1 Task 2)
```

If you have not created a `myspark.sh` script in Lab 1, use

```sh
module load Java/17.0.4
```

```sh
module load Anaconda3/2024.02-1
```

```sh
source activate myspark
```

If you are experiencing a `segmentation fault` when entering the `pyspark` interactive shell, run `export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` to fix it. It is recommended to add this line to your `myspark.sh` file.

You can now use pip to install the packages using

```sh
pip install matplotlib pandas
```

**You only need to install matplotlib and pandas in your environment once.**

The dataset that we will use is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php), where UCI stands for University of California Irvine. The UCI repository is and has been a valuable resource in Machine Learning. It contains datasets for classification, regression, clustering and several other machine learning problems. These datasets are open source and they have been uploaded by contributors of many research articles.

The particular dataset that we will use wil be referred to is the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase). A detailed description is in the previous link. The dataset contains 57 features related to word frequency, character frequency, and others related to capital letters. The description of the features and labels in the dataset is available [here](http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names). The output label indicated whether an email was considered 'ham' or 'spam', so it is a binary label.

<!-- Before we work on Logistic Regression, though, let us briefly look at the different file storage systems available in Stanage and different Spark configurations that are necessary to develop a well performing Spark job. -->

## 1. Spark configuration

Take a look at the configuration of the Spark application properties [here (the table)](https://spark.apache.org/docs/latest/configuration.html#application-properties). There are also several good references: [set spark context](https://stackoverflow.com/questions/30560241/is-it-possible-to-get-the-current-spark-context-settings-in-pyspark); [set driver memory](https://stackoverflow.com/questions/53606756/how-to-set-spark-driver-memory-in-client-mode-pyspark-version-2-3-1); [set local dir](https://stackoverflow.com/questions/40372655/how-to-set-spark-local-dir-property-from-spark-shell).

Recall that in the provided [`Code/LogMiningBig.py`](Code/LogMiningBig.py), you were asked to set the `spark.local.dir` to `/mnt/parscratch/users/YOUR_USERNAME` as in the following set of instructions

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

In the instructions above, we have configured Spark's `spark.local.dir` property to `/mnt/parscratch/users/YOUR_USERNAME` to use it as a "scratch" space (e.g. storing temporal files).

Detailed information about the different storage systems of HPC can be found in [this link](https://docs.hpc.shef.ac.uk/en/latest/hpc/filestore.html).

In shell, we can check *customized* (defaults are not shown) config via `sc`:

```python
sc._conf.getAll()
# [('spark.driver.port', '40888'), ('spark.app.startTime', '1708196748511'), ('spark.app.id', 'local-1708196749782'), ('spark.executor.id', 'driver'), ('spark.app.submitTime', '1708196747475'), ('spark.app.name', 'PySparkShell'), ('spark.driver.extraJavaOptions', '-Djava.net.preferIPv6Addresses=false -XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED -Djdk.reflect.useDirectMethodHandle=false'), ('spark.sql.catalogImplementation', 'hive'), ('spark.driver.host', 'node002.pri.stanage.alces.network'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.master', 'local[*]'), ('spark.submit.pyFiles', ''), ('spark.submit.deployMode', 'client'), ('spark.ui.showConsoleProgress', 'true'), ('spark.executor.extraJavaOptions', '-Djava.net.preferIPv6Addresses=false -XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED -Djdk.reflect.useDirectMethodHandle=false')]
```

### Driver memory and potential `out of memory` problem

**Note:** *Pay attention to the memory requirements that you set in the .sh file, and in the spark-submit instructions*

Memory requirements that you request from Stanage are configured in the following two lines appearing in your .sh file

```sh
#!/bin/bash
#SBATCH --cpus-per-task=2 # The smp parallel environment provides multiple cores on one node. <nn> specifies the max number of cores.
#SBATCH --mem-per-cpu=4G # --mem-per-cpu=xxG is used to specify the maximum amount (xx) of real memory to be requested per CPU core.
```

With the configuration above in the .sh file, we are requesting Stanage for 8GB (2 cores times 4GB per cores) of real memory. If we are working in the `rse-com6012-$Lab_ID` queue, we are requesting access to one of the five reserved [general CPU nodes](https://docs.hpc.shef.ac.uk/en/latest/stanage/cluster_specs.html#general-cpu-node-specifications) that we have for this course. We can check we have been allocated to one of these nodes because they are named as `node009` to `node013` in the Linux terminal. Each of these nodes has a total of 256 GB memory and 64 cores, i.e. 4 GB per core. When configuring your .sh file, you need to be careful about how you set these two parameters. In the past, we have seen .sh files intended to be run in one of our nodes with the following configuration

```sh
#!/bin/bash
#SBATCH --cpus-per-task=10 
#SBATCH --mem-per-cpu=30G 
```

**Do you see a problem with this configuration?** In this .sh file, they were requesting 300 GB of memory (10 nodes times 30 GB per core) which exceeds the available memory in each of these nodes, 256 GB.

As well as paying attention to your .sh file for memory requirements, we also need to configure memory requirements in the instructions when we use `spark-submit`, particularly, for the memory that will be allocated to the driver and to each of the executors. The default driver memory, i.e., `spark.driver.memory`, is ONLY **1G** (see [this Table](https://spark.apache.org/docs/3.2.1/configuration.html#application-properties)) so even if you have requested more memory, there can be out of memory problems due to this setting (read the setting description for `spark.driver.memory`). This is true for other memory-related settings as well like the `spark.executor.memory`.

The `spark.driver.memory` option can be changed by setting the configuration option, e.g.,

```sh
spark-submit --driver-memory 8g AS1Q2.py
```

**The amount of memory specified in `driver-memory` above should not exceed the amount you requested to Stanage via your .sh file or qrshx if you are working on interactive mode.**

In the past, we have seen .sh files intended to be run in one of our `rse-com6012-$Lab_ID` (this week is `rse-com6012-3`) nodes with the following configuration

```sh
#!/bin/bash
#SBATCH --account=rse-com6012   
#SBATCH --reservation=rse-com6012-3  
#SBATCH --time=00:30:00  # Change this to a longer time if you need more time
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=2
#SBATCH --output=./Output/output.txt  # This is where your output and errors are logged

module load Java/17.0.4

module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 10g ./Code/LogMiningBig.py  # .. is a relative path, meaning one level up
```

**Do you see a problem with the memory configuration in this .sh file?** Whoever submitted this .sh file was asking Stanage to assign them 2 cores and 4GB per core. At the same time, their Spark job was asking for 10GB for the driver node. The obvious problem here is that there will not be any node with 10GB available to be set as a driver node since all nodes requested from Stanage will have a maximum of 4G available.

### Other configuration changes

Other configuration properties that we might find useful to change dynamically are `executor-memory` and `master local`. By default, `executor-memory` is 1GB, which might not be enough in some large data applications. You can change the `executor-memory` when using spark-submit, for example

```sh
spark-submit --driver-memory 10g --executor-memory 10g ./Code/LogMiningBig.py  
```

Just as before, one needs to be careful that the amount of memory dynamically requested through spark-submit does not go beyond what was requested from Stanage. In the past, we have seen .sh files intended to be run in one of our `rse-com6012-3` nodes with the following configuration

```sh
#!/bin/bash
#SBATCH --account=rse-com6012   
#SBATCH --reservation=rse-com6012-3  
#SBATCH --time=00:30:00  # Change this to a longer time if you need more time
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G 
#SBATCH --output=./Output/output.txt  # This is where your output and errors are logged

module load Java/17.0.4

module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 20g --executor-memory 30g ./Code/LogMiningBig.py  
```

**Do you see a problem with the memory configuration in this .sh file?** This script is asking Stanage for each core to have 20GB. This is fine because the script is requesting 200GB in total (10 times 20GB) which is lower than the maximum of 1024GB. However, although spark-submit is requesting the same amount of 20GB per node for the `driver-memory`, the `executor-memory` is asking for 30G. There will not be any core with a real memory of 30G so the `executor-memory` request needs to be a maximum of 20G.

Another property that may be useful to change dynamically is `--master local`. So far, we have set the number of worker threads in the `SparkSession.builder` inside the Python script, for example,

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

But we can also specify the number of worker threads in spark-submit using

```sh
spark-submit --driver-memory 5g --executor-memory 5g --master local[10] ./Code/LogMiningBig.py  
```

It is important to notice, however, that if `master local` is specified in spark-submit, you would need to remove that configuration from the SparkSession.builder,

```python
spark = SparkSession.builder \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

Or make sure the number of worker threads you specify with SparkSession.builder matches the number of cores you specify when using spark-submit,

```python
spark = SparkSession.builder \
    .master("local[10]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

What happens if this is not the case? For example, if the number of worker threads specified in spark-submit is different from the number of worker threads specified in SparkSession. In the past, we have seen the following instructions in the .sh file

```sh
spark-submit --driver-memory 5g --executor-memory 5g --master local[5] ./Code/LogMiningBig.py  
```

and when inspecting the Python file, the following instruction for SparkSession

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

**Do you see a problem with the number of worker threads in the configuration for these two files?** While spark-submit is requesting 5 worker threads, the SparkSession is requesting 2 cores. According to Spark documentation "Properties set directly on the SparkConf take highest precedence, then flags passed to spark-submit or spark-shell, then options in the spark-defaults.conf file." (see [this link](https://spark.apache.org/docs/3.5.0/configuration.html#dynamically-loading-spark-properties)) meaning that the job will run with 2 cores and no 5 cores as intended in spark-submit.

Finally, the number of worker threads requested through spark-submit needs to match the number of cores requested from Stanage with `#SBATCH --cpus-per-task=nn` in the .sh file. In the past, we have seen the following instructions in the .sh file

```sh
#!/bin/bash
#SBATCH --account=rse-com6012
#SBATCH --reservation=rse-com6012-3  
#SBATCH --time=00:30:00  # Change this to a longer time if you need more time
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G 
#SBATCH --output=./Output/output.txt  # This is where your output and errors are logged

module load Java/17.0.4

module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 20g --executor-memory 20g --master local[15] ./Code/LogMiningBig.py 
```

with the corresponding python file including

```python
spark = SparkSession.builder \
    .master("local[15]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
    .getOrCreate()
```

**Do you see a problem with the number of worker threads in the configuration for these two files?** Although the number of worker threads requested through spark-submit and the SparkSession.builder are the same, that number does not match the number of cores requested to Stanage in the .sh file. Actually, spark-submit is requesting a higher number of worker threads than the ones that could potentially be assigned by Stanage.

#### To change more configurations

You may search for example usage, an example that we used in the past **for very big data** is here for your reference only:

```sh
spark-submit --driver-memory 20g --executor-memory 20g --master local[10] --local.dir /mnt/parscratch/users/USERNAME --conf spark.driver.maxResultSize=4g test.py
```

#### Observations

1. If the real memory usage of your job exceeds `--mem-per-cpu=xxG` multiplied by the number of cores / nodes you requested then your job will be killed (see the [HPC documentation](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#interactive-jobs)).
2. A reminder that the more resources you request to Stanage, the longer you need to wait for them to become available to you.

## 2. Logistic regression in PySpark

We start with the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase). We load the dataset and the names of the features and label. We cache the dataframe for efficiently performing several operations to rawdata inside a loop.

```python
import numpy as np
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# For being able to save files in a Parquet file format, later on, we need to rename
# two columns with invalid characters ; and (
spam_names[spam_names.index('char_freq_;')] = 'char_freq_semicolon'
spam_names[spam_names.index('char_freq_(')] = 'char_freq_leftp'
```

We now rename the columns using the more familiar names for the features.

```python
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])
```

We import the Double type from pyspark.sql.types, use the withColumn method for the dataframe and cast() the column to DoubleType.

```python
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(spam_names[i], rawdata[spam_names[i]].cast(DoubleType()))
```

We use the same seed that we used in the previous Notebook to split the data into training and test.

```python
(trainingDatag, testDatag) = rawdata.randomSplit([0.7, 0.3], 42)
```

**Save the training and tets sets** Once we have split the data into training and test, we can save to disk both sets so that we can use them later, for example, to compare the performance of different transformations to the data or ML models on the same training and test data. We will use the [Apache Parquet](https://en.wikipedia.org/wiki/Apache_Parquet) format to efficiently store both files.

```python
trainingDatag.write.mode("overwrite").parquet('./Data/spamdata_training.parquet')
testDatag.write.mode("overwrite").parquet('./Data/spamdata_test.parquet')
```

Let us read from disk both files

```python
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')
```

We create the VectorAssembler to concatenate all the features in a vector.

```python
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features') 
```

**Logistic regression** We are now in a position to train the logistic regression model. But before, let us look at a list of relevant parameters. A comprehensive list of parameters for [LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html) can be found in the Python API for PySpark.

> `maxIter`: max number of iterations
>
> `regParam`: regularization parameter ($\ge 0$)
>
> `elasticNetParam`: mixing parameter for ElasticNet. It takes values in the range [0,1]. For $\alpha=0$, the penalty is an $\ell_2$. For $\alpha=1$, the penalty is an $\ell_1$.
>
> `family`: binomial (binary classification) or multinomial (multi-class classification). It can also be 'auto'.
>
> `standardization`: whether to standardize the training features before fitting the model. It can be true or false (True by default).

The function to optimise has the form

$$f(\mathbf{w}) = LL(\mathbf{w}) + \lambda\Big[\alpha\|\mathbf{w}\|_1 + (1-\alpha)\frac{1}{2}\|\mathbf{w}\|_2\Big]$$

where $LL(\mathbf{w})$ is the logistic loss given as

$$
LL(\mathbf{w}) = \sum_{n=1}^N \log[1+\exp(-y_n\mathbf{w}^{\top}\mathbf{x}_n)].
$$

Let us train different classifiers on the same training data. We start with logistic regression, without regularization, so $\lambda=0$.

```python
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0, family="binomial")
```

We now create a pipeline for this model and fit it to the training data

```python
from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [vecAssembler, lr]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)
```

Let us compute the accuracy.

```python
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

```bash
Accuracy = 0.925362 
```

We now save the vector $\mathbf{w}$ obtained without regularization

```python
w_no_reg = pipelineModel.stages[-1].coefficients.values
```

We now train a second logistic regression classifier using only $\ell_1$ regularization ($\lambda=0.01$ and $\alpha=1$)

```python
lrL1 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=1, family="binomial")

# Pipeline for the second model with L1 regularization
stageslrL1 = [vecAssembler, lrL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(trainingData)

predictions = pipelineModellrL1.transform(testData)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

```bash
Accuracy = 0.913176 
```

We now save the vector $\mathbf{w}$ obtained for the L1 regularization

```python
w_L1 = pipelineModellrL1.stages[-1].coefficients.values
```

Let us plot the values of the coefficients $\mathbf{w}$ for the no regularization case and the L1 regularization case.

```python
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(w_no_reg)
ax1.set_title('No regularization')
ax2.plot(w_L1)
ax2.set_title('L1 regularization')
plt.savefig("Output/w_with_and_without_reg.png")
```

Let us find out which features are preferred by each method. Without regularization, the most relevant feature is

```python
spam_names[np.argmax(np.abs(w_no_reg))]
```

```bash
'word_freq_cs'
```

With L1 regularization, the most relevant feature is

```python
spam_names[np.argmax(np.abs(w_L1))]
```

```bash
'char_freq_$'
```

A useful method for the logistic regression model is the [summary](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegressionSummary.html) method.

```python
lrModel1 = pipelineModellrL1.stages[-1]
lrModel1.summary.accuracy
```

```bash
0.9111922141119222
```

The accuracy here is different to the one we got before. Why?

Other quantities that can be obtained from the summary include falsePositiveRateByLabel, precisionByLabel, recallByLabel, among others. For an exhaustive list, please read [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegressionSummary.html).

```python
print("Precision by label:")
for i, prec in enumerate(lrModel1.summary.precisionByLabel):
    print("label %d: %s" % (i, prec))
```

```bash
Precision by label:
label 0: 0.8979686057248384
label 1: 0.9367201426024956
```

## 3. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Thursday (the latest).

### Exercise 1

Try a pure L2 regularization and an elastic net regularization on the same data partitions from above. Compare accuracies and find the most relevant features for both cases. Are these features the same than the one obtained for L1 regularization?

### Exercise 2

Instead of creating a logistic regression model trying one type of regularization at a time, create a [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html) to be used inside a [CrossValidator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html) to fine tune the best type of regularization and the best parameters for that type of regularization. Use five folds for the CrossValidator.

## 4. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

Create a logistic regression classifier that runs on the [default of credit cards](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. Several of the features in this dataset are categorical. Use the tools provided by PySpark (pyspark.ml.feature) for treating categorical variables.

Note also that this dataset has a different format to the Spambase dataset above - you will need to convert from XLS format to, say, CSV, before using the data. You can use any available tool for this: for example, Excell has an export option, or there is a command line tool `xls2csv` available on Linux.
