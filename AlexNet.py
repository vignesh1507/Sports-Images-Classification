import os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Building 'AlexNet'
network = input_data(shape=[None, 224, 224, 3])
network = conv_2d(network, 96, 7, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='momentum', loss='categorical_crossentropy')

model = tflearn.DNN(network, tensorboard_verbose=3, best_checkpoint_path='model_alex_high', best_val_accuracy=90.0)

if (os.path.exists('alexnetNewEdit5.tfl.meta')):
        model.load('alexnetNewEdit5.tfl')
else:
        # X_train, y_train = createTraindata()  # undefined function
        # model.fit(X_train, y_train, n_epoch=45, validation_set=0.2, shuffle=True,
        #         snapshot_step=200, snapshot_epoch=True, show_metric=True, batch_size=36) # 
        # model.save('/content/drive/MyDrive/AlexNet/alexnetEdit.tfl')

x_test, ImageName_ext = # createTestdata()  # undefined function
x = model.predict(x_test)
x = list(x)

output2 = []
for i in range(len(x)):
  output2.append([ImageName_ext[i],np.argmax(x[i])])
  
# createcsv(output2)  # undefined function
