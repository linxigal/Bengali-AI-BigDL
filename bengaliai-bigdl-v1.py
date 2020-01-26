from platform import python_version
print(python_version())

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bigdl.util import common
from bigdl.util.common import *
from pyspark import SparkContext
from bigdl.nn.layer import *
import bigdl.version
from pyspark.sql import SparkSession

from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.dataset.transformer import *

#from matplotlib.pyplot import imshow
#import matplotlib.pyplot as plt

# create sparkcontext with bigdl configuration
sc = SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[*]"))
init_engine() # prepare the bigdl environment
bigdl.version.__version__ # Get the current BigDL version

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

pdf = pd.read_parquet("/home/matt/Bengalia/train_image_data_0.parquet").iloc[0:50,:]
pdf_sml = pdf.iloc[0:25,:]
sdf = sqlContext.createDataFrame(pdf_sml)
rdd_train_images = sdf.drop('image_id').rdd
rdd_train_labels = sc.parallelize(pd.read_csv("/home/matt/Bengalia/train.csv")["grapheme_root"].iloc[0:25])
rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda features_label: common.Sample.from_ndarray(np.asarray([x / 255 for x in features_label[0]]),
                                                                                                                features_label[1] + 1))

pdf_sml = pdf.iloc[26:50,:]
sdf = sqlContext.createDataFrame(pdf_sml)
rdd_test_images = sdf.drop('image_id').rdd
rdd_test_labels = sc.parallelize(pd.read_csv("/home/matt/Bengalia/train.csv")["grapheme_root"].iloc[26:50])
rdd_test_sample = rdd_test_images.zip(rdd_test_labels).map(lambda features_label: common.Sample.from_ndarray(np.asarray([x / 255 for x in features_label[0]]),
                                                                                                             features_label[1] + 1))

from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import *

lenet_model = Sequential()
lenet_model.add(Reshape((1, 137, 236), input_shape=([32332])))
lenet_model.add(Convolution2D(3, 5, 5, activation="tanh", name="conv1_5x5"))
lenet_model.add(MaxPooling2D())
lenet_model.add(Flatten())
lenet_model.add(Dense(200, activation="tanh", name="fc1"))
lenet_model.add(Dense(168, activation="softmax", name="fc2"))
lenet_model.get_input_shape()
lenet_model.get_output_shape()

# Create an Optimizer

optimizer = Optimizer(
    model=lenet_model,
    training_rdd=rdd_train_sample,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=0.4, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(1),
    batch_size=5)

# Set the validation logic
optimizer.set_validation(
    batch_size=5,
    val_rdd=rdd_test_sample,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

# Boot training process
trained_model = optimizer.optimize()
print("Optimization Done.")
