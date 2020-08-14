import pandas as pd
import numpy as np
import tensorflow as tf
import os

train_dataset = pd.read_csv('dataset_train.csv')
y_train = train_dataset.values[:,0]
x_train = train_dataset.values[:,1:]

test_dataset = pd.read_csv('dataset_test.csv')
y_test = test_dataset.values[:,0]
x_test= test_dataset.values[:,1:]

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

model = create_model()
#model.summary()
checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

test_state = x_test[1790]
print(test_state)
print(y_test[1790])
state_batch = (np.expand_dims(test_state,0))

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions_single = probability_model.predict(state_batch)
print(predictions_single)
print(np.argmax(predictions_single[0]))
