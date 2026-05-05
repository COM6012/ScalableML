# Lab 9: Scalable neural networks


## Study schedule

- [Section 1](#1-Shallow-neural-networks-in-PySpark): To finish by 6th May. **Essential**
- [Section 2](#2-PyTorch-on-PySpark): To finish by 6th May. **Essential**
- [Section 3](#3-Exercises): To finish before next Thursday 14th May. ***Exercise***
- [Section 4](#4-additional-exercises-optional): To explore further. *Optional*

## Introduction

In this notebook we will explore the use of the neural network model that comes implemented in Spark ML. We will also look at Pandas UDFs that allow to efficiently use models trained on a single machine using frameworks different to Spark ML, e.g. PyTorch, Scikit-learn, but keeping the benefits of Spark to apply such models to large datasets.

**Dependencies.** For this lab, we need the packages `numpy`, `pyarrow`, `torch`, `scikit-learn`, `matplotlib`, and `pandas`. `pyspark` should already be installed in **myspark** from Lab 1.

Before continuing, activate your environment. If you have a `myspark.sh` script in your home directory (set up in Lab 1), run:

```
source ~/myspark.sh
```

Otherwise, run the three commands manually:

```
module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark
```

You must see `(myspark)` in your prompt before proceeding.

#### Step 1 — Check what is already installed

Run the following to see which packages are already available in your environment:

```python
python -c "
import numpy;      print('numpy:     ', numpy.__version__)
import pyarrow;    print('pyarrow:   ', pyarrow.__version__)
import torch;      print('torch:     ', torch.__version__)
import sklearn;    print('sklearn:   ', sklearn.__version__)
import matplotlib; print('matplotlib:', matplotlib.__version__)
import pandas;     print('pandas:    ', pandas.__version__)
"
```

If all six lines print version numbers without errors, all packages are installed — skip to Section 1. If any package raises an `ImportError`, follow the installation steps below.

#### Step 2 — Install missing packages

> **Important note on installation method.** On Stanage, some packages cannot be installed via `pip` because they require a newer GCC compiler than is available on the nodes. The packages `numpy`, `pyarrow`, `scikit-learn`, `matplotlib`, and `pandas` must be installed via **conda**, which provides pre-built binaries requiring no compilation. Only `torch` should be installed via **pip**, which provides a pre-built wheel for this platform.

Install via conda first:

```
conda install -y -c conda-forge numpy pyarrow scikit-learn matplotlib pandas
```

Then install PyTorch via pip:

```
pip install torch
```

> **Note on PyTorch.** This lab runs entirely on CPU — no GPU is required. The standard `pip install torch` is sufficient. If you are interested in GPU-accelerated PyTorch on Stanage for other work, refer to the [official Stanage PyTorch documentation](https://docs.hpc.shef.ac.uk/en/latest/stanage/software/apps/pytorch.html).

#### Step 3 — Verify installation

Run the check from Step 1 again to confirm all packages are now installed.

Make sure all six lines should print version numbers without errors before proceeding.

## 1. Shallow neural networks in PySpark

To illustrate the use of the neural network model that comes in Spark ML, we use the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase) that we already used in Lab 3.

We need to enable Arrow in Spark. More on this later in the lab.

```python
# Enable Arrow-based columnar data transfers. This line of code will be explained later
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
```

The following code is used to load the dataset and properly set the corresponding dataframe.

```python
# We load the dataset and the names of the features
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

# We rename the columns in the dataframe with names of the features in spambase.data.names
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# We cast the type string to double
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))
```

We now create the training and test sets.

```python
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 42)
```

We create instances for the vector assembler and the neural network.

```python
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')
```

The class that implements the neural network model is called the [MultilayerPerceptronClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html). The multilayer perceptron implemented in Spark ML only allows for sigmoidal activation functions in the intermediate layers and the softmax function in the output layer. We can then use the model for binary and multi-class classification.

The architecture of the network is specified through the argument ``layers`` which is a list. The length of the list is equivalent to the number of hidden layers plus two additional numbers that indicate the number of inputs and the number of outputs. The number of inputs is the first element of the list and the number of outputs is the last element of the list.

For example, if ``layers=[10, 5, 4, 3]``, then this neural network assumes a first layer of 10 nodes (the features), followed by two hidden layers of 5 and 4 nodes and a last layer of 3 outputs (classes).

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier
# The first element HAS to be equal to the number of input features
layers = [len(trainingData.columns)-1, 20, 5, 2]
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", maxIter=100, layers=layers, seed=1500)
```

We now create the pipeline, fit it to data and compute the performance over the test set.

```python
# Create the pipeline
from pyspark.ml import Pipeline
stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)

# We now make predictions
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

    Accuracy = 0.861386


## 2. PyTorch on PySpark

The alternatives for neural networks in Spark ML are rather limited. We can only do classification through the class `MultilayerPerceptronClassifier` and even for this model there are important restrictions, e.g. it can only use logistic sigmoid activation functions.

It is possible to use more advanced neural network models, including deep learning models, in PySpark. A way to do it is to make use of a powerful API in Spark SQL known as **pandas user defined functions** or [**pandas UDFs**](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html#pandas-udfs-a-k-a-vectorized-udfs) for short (they are also known as vectorized UDFs) or equivalently through different [**pandas functions APIs**](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html#pandas-function-apis), like **mapInPandas()**. PySpark uses [Apache Arrow](https://en.wikipedia.org/wiki/Apache_Arrow) to efficiently transfer data between the JVM (Java Virtual Machine) and Python processes, allowing the efficient use of pandas for data analytic jobs in the cluster. A comprehensive user guide of Pandas with Arrow can be found [here](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html).

A typical use case for pandas UDFs or pandas functions APIs in scalable machine learning consists of training a machine learning model on a single machine using a subset of the data and then using that model to provide predictions at scale by distributing the trained model to the executors, which later compute the predictions.

In this section of the Lab, we will train a **PyTorch** model using a subset of the spambase dataset and then we will use [mapInPandas()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.mapInPandas.html) to ask the executors to compute the predictions over the test set.

#### Using PyTorch to train a neural network model over the Spambase dataset

We want to use the same training data that we used in Section 1 of the lab. However, we first need to transform the spark dataframe into a pandas dataframe.

If you go back to the beginning of the lab, we used the instruction

`spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")`

This instruction enabled Arrow so that the transformation between spark dataframes and pandas dataframes can be done efficiently.

```python
# Convert the Spark DataFrame to a Pandas DataFrame using Arrow
trainingDataPandas = trainingData.select("*").toPandas()
```

We prepare the data for PyTorch. Unlike Keras, PyTorch works directly with NumPy arrays converted to tensors, so we extract the feature matrix and target vector as typed NumPy arrays.

```python
nfeatures = ncolumns - 1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures].values.astype(np.float32)
ytrain = trainingDataPandas.iloc[:, -1].values.astype(np.float32)
```

We now define the neural network model. In PyTorch, models are defined as Python classes that inherit from `torch.nn.Module`. The `__init__` method defines the layers and the `forward` method defines how data flows through them.

We use the same architecture as in Section 1: an input layer matching the number of features, two hidden layers of 20 and 5 nodes with ReLU activations, and a single output node with a Sigmoid activation for binary classification.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SpamNet(nn.Module):
    def __init__(self, input_size):
        super(SpamNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = SpamNet(nfeatures)
```

Unlike Keras, PyTorch does not have a `compile()` step. Instead, we define the loss function and optimiser as separate objects. We use binary cross-entropy loss (equivalent to Keras's `binary_crossentropy`) and RMSprop (equivalent to Keras's `rmsprop` optimiser).

```python
criterion = nn.BCELoss()                       # Binary cross-entropy loss
optimizer = optim.RMSprop(model.parameters())  # RMSprop optimiser
```

We now prepare the data for training. We convert the NumPy arrays into PyTorch tensors and apply a manual 80/20 training/validation split. We then wrap the training data in a `DataLoader`, which handles mini-batch creation automatically.

```python
X_tensor = torch.FloatTensor(Xtrain)
y_tensor = torch.FloatTensor(ytrain).unsqueeze(1)  # Add output dimension: shape (N, 1)

# Manual 80/20 train/validation split
val_size  = int(0.2 * len(X_tensor))
train_size = len(X_tensor) - val_size
X_train_t, X_val_t = X_tensor[:train_size], X_tensor[train_size:]
y_train_t, y_val_t = y_tensor[:train_size], y_tensor[train_size:]

# DataLoader feeds mini-batches to the model during training
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=100, shuffle=True)
```

Unlike Keras, which provides a built-in `fit()` method that handles the training loop internally, in PyTorch we write the training loop **explicitly**. This gives full control over every step: forward pass, loss computation, backpropagation, and parameter update.

The loop also alternates between two modes:
- `model.train()` — enables dropout and batch normalisation (if present) during training
- `model.eval()` — disables them during validation/inference

```python
epochs = 100
train_accs, val_accs = [], []

for epoch in range(epochs):
    # --- Training phase ---
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()              # Clear accumulated gradients
        output = model(X_batch)            # Forward pass
        loss = criterion(output, y_batch)  # Compute loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update model weights

    # --- Evaluation phase (no gradient tracking needed) ---
    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train_t) >= 0.5).float()
        train_acc   = (train_preds == y_train_t).float().mean().item()
        val_preds   = (model(X_val_t) >= 0.5).float()
        val_acc     = (val_preds == y_val_t).float().mean().item()

    train_accs.append(train_acc)
    val_accs.append(val_acc)
```

Let us plot the progress of the training.

```python
import matplotlib.pyplot as plt

epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, train_accs, 'bo', label='Training acc')
plt.plot(epochs_range, val_accs,   'b',  label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./Output/pytorch_nn_train_validation_history.png")
plt.close()
```

#### Distributing predictions with mapInPandas

We will now use `mapInPandas` to distribute the prediction computation to the Spark executors. For this to work, the trained model must be **picklable**.

**What is pickling?** Pickling is the process of converting a Python object into a byte stream so that it can be saved to disk or transmitted over a network. In Spark's distributed setting, when we call `mapInPandas`, Spark needs to send the trained model from the driver to each worker node. Pickling is exactly how Python achieves this serialisation. A model that is *picklable* can be broadcast to workers transparently, without any extra configuration.

**PyTorch and pickling.** PyTorch models are natively picklable — Python's built-in `pickle` module can serialise and deserialise them without any workaround. This makes the integration with `mapInPandas` straightforward. Note that some other frameworks (like Keras) required custom serialisation workarounds to achieve the same result; PyTorch avoids this complexity entirely, so we can pass the model directly to the worker nodes.

We first extract from `testData` a dataframe containing only the feature columns, and define the schema for the output dataframe (features plus a prediction column).

```python
import pandas as pd
from pyspark.sql.types import StructField, StructType, DoubleType

Xtest = testData.select(spam_names[0:ncolumns-1])

pred_field = [StructField("prediction", DoubleType(), True)]
new_schema = StructType(Xtest.schema.fields + pred_field)
```

We create a `predict` function that will be applied by the worker nodes. It iterates over batches of pandas DataFrames, converts each batch to a PyTorch tensor, runs the model, and returns the batch with the predictions appended. Note the use of `model.eval()` and `torch.no_grad()` to ensure inference mode is set and no unnecessary gradient computation is performed.

```python
def predict(iterator):
    model.eval()
    with torch.no_grad():
        for features in iterator:
            X = torch.FloatTensor(features.values.astype(np.float32))
            preds = model(X).numpy().flatten()
            yield pd.concat([features, pd.Series(preds, name="prediction")], axis=1)
```

We now apply `predict` to batches of the `Xtest` dataframe using `mapInPandas`.

```python
prediction_pytorch_df = Xtest.mapInPandas(predict, new_schema)
```

The resulting dataframe is a Spark dataframe. We select the column of predictions and transform it to pandas to compute the accuracy on the test data.

```python
ypred_pytorch = prediction_pytorch_df.select('prediction').toPandas().values.copy()
```

We use a threshold of 0.5 to assign predictions to class 0 and class 1.

```python
ypred_pytorch[ypred_pytorch <  0.5] = 0
ypred_pytorch[ypred_pytorch >= 0.5] = 1
```

> **Note.** The `.copy()` call is required because `.toPandas().values` returns a read-only NumPy array in recent versions of pandas and NumPy. Without it, the in-place assignment above raises a `ValueError: assignment destination is read-only`.

We now extract the target test labels from the `testData` dataframe.

```python
testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values
```

We finally use the `accuracy_score` method from scikit-learn to compute the accuracy.

```python
from sklearn.metrics import accuracy_score
print("Accuracy = %g " % accuracy_score(ypred_pytorch, ytest))
```

    Accuracy = 0.911653

The accuracy obtained by the PyTorch model is different from the one obtained using the neural network model in Spark ML even though they are using the same training data. Why is that?


## 3. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Wednesday (the latest).

### Exercise 1

Include a cross-validation step for the pipeline of the neural network applied to the spambase dataset in [section 1](#1-Shallow-neural-networks-in-PySpark). An example of a cross-validator can be found [here](http://spark.apache.org/docs/3.0.1/ml-tuning.html#cross-validation). Make <tt>paramGrid</tt> contains different values for the parameter ``layers`` and find the best parameters and associated accuracy on the test data.

### Exercise 2

Repeat [section 2](#2 PyTorch on PySpark) of this Lab but now experiment with a Scikit-learn model. Choose [a classifier from the ones available](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning), train the classifier over the same training data and use pandas and arrow to send the model to the worker nodes, which will provide the predictions. Do you need to worry about pickling for scikit-learn models?


## 4. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

Extend [section 2](#2 PyTorch on PySpark) by experimenting with different neural network architectures in PyTorch. Try adding more hidden layers, changing the activation functions (e.g. `nn.Tanh()` instead of `nn.ReLU()`), varying the number of neurons per layer, or adjusting the number of epochs and the batch size. How does the test accuracy change? Can you find an architecture that outperforms the one used in Section 2?
