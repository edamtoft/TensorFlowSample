import tensorflow as tf
import pandas as pd

measurements = [
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth"
]

species = [
  "Iris-setosa",
  "Iris-versicolor",
  "Iris-virginica"
]

# Prepare Data

data = pd.read_csv("./iris.data.csv", names=measurements+["Species"])

test_features = data.sample(frac=0.2)
train_features = data.drop(test_features.index)

train_labels = train_features.pop("Species")
test_labels = test_features.pop("Species")

# Set Up Classifier

columns = [tf.feature_column.numeric_column(key=column) for column in measurements]

classifier = tf.estimator.DNNClassifier(
  feature_columns=columns,
  hidden_units=[10,10],
  label_vocabulary=species,
  n_classes=len(species))

# Set Up Inputs

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_features,y=train_labels,shuffle=True,batch_size=50,num_epochs=100)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,y=test_labels,shuffle=True,batch_size=50,num_epochs=100)

# Train and Evaluate

classifier.train(input_fn=train_input_fn)

evaluation = classifier.evaluate(input_fn=test_input_fn)

print("model accuracy is {accuracy:.3f}%".format(accuracy=evaluation["accuracy"]*100))

# Predict A Species

predict_features = pd.DataFrame.from_dict({
    "SepalLength": [6.8],
    "SepalWidth": [2.8],
    "PetalLength": [4.8],
    "PetalWidth": [1.4]
})

predictions = classifier.predict(input_fn=tf.estimator.inputs.pandas_input_fn(x=predict_features,shuffle=False))

prediction = next(predictions)
class_id = prediction["class_ids"][0]
probability = prediction["probabilities"][class_id]
species = species[class_id]
print("prediction is {species} with {probability:.3f}% certainty".format(species=species, probability=probability*100))













    
