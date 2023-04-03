from PIL import Image
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
# loss
def rmse(y_true, y_predict):
    return K.sqrt(K.mean(K.square((y_true/4.4)-(y_predict/4.4))))

def hNet(height, width, channels, drop =0.2, alpha = 0.3):
	
	def third_branch(x, up_scale):
		if (up_scale>1):
			x = LeakyReLU(0.3)(Conv2D(1, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(x))
			x = Conv2DTranspose(1, 2*up_scale, strides = up_scale, padding='same', activation=None, use_bias = False)(x)
		else:
			x = LeakyReLU(0.3)(Conv2D(1, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(x))
		return x
	
	inputs = Input((height, width, channels))
	# First branch
	conv1 = LeakyReLU(alpha)(Conv2D(32, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(inputs))
	conv1 = LeakyReLU(alpha)(Conv2D(32, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = LeakyReLU(alpha)(Conv2D(64, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(pool1))
	conv2 = LeakyReLU(alpha)(Conv2D(64, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = LeakyReLU(alpha)(Conv2D(128, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(pool2))
	conv3 = LeakyReLU(alpha)(Conv2D(128, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = LeakyReLU(alpha)(Conv2D(256, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(pool3))
	conv4 = LeakyReLU(alpha)(Conv2D(256, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = LeakyReLU(alpha)(Conv2D(512, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(pool4))
	conv5 = LeakyReLU(alpha)(Conv2D(512, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv5))
	drop5 = Dropout(drop)(conv5)

	# Second branch
	up6 = Concatenate()([Conv2DTranspose(256, 3, strides = 2, padding='same', kernel_initializer = 'he_normal')(drop5), conv4])
	conv6 = LeakyReLU(alpha)(Conv2D(256, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(up6))
	conv6 = LeakyReLU(alpha)(Conv2D(256, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv6))

	up7 = Concatenate()([Conv2DTranspose(128, 3, strides = 2, padding='same', kernel_initializer = 'he_normal')(conv6), conv3])
	conv7 = LeakyReLU(alpha)(Conv2D(128, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(up7))
	conv7 = LeakyReLU(alpha)(Conv2D(128, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv7))

	up8 = Concatenate()([Conv2DTranspose(64, 3, strides = 2, padding='same', kernel_initializer = 'he_normal')(conv7), conv2])
	conv8 = LeakyReLU(alpha)(Conv2D(64, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(up8))
	conv8 = LeakyReLU(alpha)(Conv2D(64, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv8))

	up9 = Concatenate()([Conv2DTranspose(32, 3, strides = 2, padding='same', kernel_initializer = 'he_normal')(conv8), conv1])
	conv9 = LeakyReLU(alpha)(Conv2D(32, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(up9))
	conv9 = LeakyReLU(alpha)(Conv2D(32, 3, activation = None, padding='same', kernel_initializer = 'he_normal')(conv9))

	# Third branch
	c1 = third_branch(drop5,16)
	c2 = third_branch(conv6,8)
	c3 = third_branch(conv7,4)
	c4 = third_branch(conv8,2)
	c5 = third_branch(conv9,1)

	o1 = Conv2D(1, 1, activation = 'linear')(c1)
	o2 = Conv2D(1, 1, activation = 'linear')(c2)
	o3 = Conv2D(1, 1, activation = 'linear')(c3)
	o4 = Conv2D(1, 1, activation = 'linear')(c4)
	o5 = Conv2D(1, 1, activation = 'linear')(c5)

	fuse = Concatenate()([c1, c2, c3, c4, c5])
	fuse = Conv2D(1, 1, activation = 'linear')(fuse)

	model = Model(inputs=[inputs], outputs=[o1,o2,o3,o4,o5,fuse])
	return model

def UNet(height, width, channels, alpha = 0.4, drop = 0.2):
    inputs = Input((height, width, channels))
    
    conv1 = LeakyReLU(alpha)(Conv2D(32, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(inputs))
    conv1 = LeakyReLU(alpha)(Conv2D(32, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = LeakyReLU(alpha)(Conv2D(64, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(pool1))
    conv2 = LeakyReLU(alpha)(Conv2D(64, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = LeakyReLU(alpha)(Conv2D(128, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(pool2))
    conv3 = LeakyReLU(alpha)(Conv2D(128, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = LeakyReLU(alpha)(Conv2D(256, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(pool3))
    conv4 = LeakyReLU(alpha)(Conv2D(256, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = LeakyReLU(alpha)(Conv2D(512, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(pool4))
    conv5 = LeakyReLU(alpha)(Conv2D(512, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv5))
    drop5 = Dropout(drop)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(drop5), conv4], axis=3)
    conv6 = LeakyReLU(alpha)(Conv2D(256, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(up6))
    conv6 = LeakyReLU(alpha)(Conv2D(256, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv6))

    up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv6), conv3], axis=3)
    conv7 = LeakyReLU(alpha)(Conv2D(128, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(up7))
    conv7 = LeakyReLU(alpha)(Conv2D(128, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv7))

    up8 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv7), conv2], axis=3)
    conv8 = LeakyReLU(alpha)(Conv2D(64, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(up8))
    conv8 = LeakyReLU(alpha)(Conv2D(64, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv8))

    up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv8), conv1], axis=3)
    conv9 = LeakyReLU(alpha)(Conv2D(32, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(up9))
    conv9 = LeakyReLU(alpha)(Conv2D(32, (3, 3), activation = None, padding='same', kernel_initializer = 'he_normal')(conv9))

    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

def step_decay(epoch, lr):
    drop = 0.997
    epochs_drop =100.0
    lrate = lr*math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train_model_hNet(model, dir, x_train, y_train, info):
    lrate = LearningRateScheduler(step_decay, verbose=1)
    
    #For NNet only, multiple outputs
    y_train = [y_train,y_train,y_train,y_train,y_train,y_train]
    model.compile(optimizer = Adam(lr = 1e-4), loss = ['mse', 'mse', 'mse', 'mse', 'mse', 'mse'], metrics=[rmse])
    # 模型训练，返回loss和测量参数
    model_checkpoint = ModelCheckpoint( dir + '/model_' + info + '.h5', monitor='val_loss',verbose=1, save_best_only=True)
    callbacks_list = [lrate, model_checkpoint]
    history=model.fit(x_train, y_train, batch_size = 1, epochs = 200, verbose=1 , shuffle  = True, callbacks = callbacks_list, validation_split=0.1)

    plt.figure()
    plt.plot(history.history['conv2d_23_rmse'])
    plt.plot(history.history['val_conv2d_23_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_23_rmse'][199]), xy=(200, history.history['val_conv2d_23_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '1_rmse.png')
    
    plt.figure()
    plt.plot(history.history['conv2d_24_rmse'])
    plt.plot(history.history['val_conv2d_24_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_24_rmse'][199]), xy=(200, history.history['val_conv2d_24_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '2_rmse.png')
    
    plt.figure()
    plt.plot(history.history['conv2d_25_rmse'])
    plt.plot(history.history['val_conv2d_25_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_25_rmse'][199]), xy=(200, history.history['val_conv2d_25_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '3_rmse.png')
    
    plt.figure()
    plt.plot(history.history['conv2d_26_rmse'])
    plt.plot(history.history['val_conv2d_26_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_26_rmse'][199]), xy=(200, history.history['val_conv2d_26_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '4_rmse.png')
    
    plt.figure()
    plt.plot(history.history['conv2d_27_rmse'])
    plt.plot(history.history['val_conv2d_27_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_27_rmse'][199]), xy=(200, history.history['val_conv2d_27_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '5_rmse.png')
    
    plt.figure()
    plt.plot(history.history['conv2d_28_rmse'])
    plt.plot(history.history['val_conv2d_28_rmse'])
    plt.axis([-20, 220, 0, 10])
    plt.annotate("{:.2f}".format(history.history['val_conv2d_28_rmse'][199]), xy=(200, history.history['val_conv2d_28_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    plt.savefig(dir+'/' + info + '6_rmse.png')

def train_model_UNet(model, dir, x_train, y_train, x_test, y_test, info):
    lrate = LearningRateScheduler(step_decay, verbose=1)
    model.compile(optimizer = Adam(lr = 1e-4), loss = ['mse'], metrics=[rmse])

    model_checkpoint = ModelCheckpoint( dir + '/model_' + info + '.h5', monitor='val_loss',verbose=1, save_best_only=True)
    callbacks_list = [lrate, model_checkpoint]
    history=model.fit(x_train, y_train, batch_size = 1, epochs = 200, verbose=1 , shuffle  = True, callbacks = callbacks_list, validation_split=0.1)

    #### plot the training rmse
    #plt.figure()
    #plt.plot(history.history['rmse'])
    #plt.plot(history.history['val_rmse'])
    #plt.axis([-20, 220, 0, 10])
    #plt.annotate("{:.2f}".format(history.history['val_rmse'][199]), xy=(200, history.history['val_rmse'][199]), xytext = (-20, 30),  xycoords = "data", textcoords = 'offset points', arrowprops = dict(arrowstyle="->", connectionstyle="arc3"))
    #plt.ylabel('RMSE')
    #plt.xlabel('Epoch')
    #plt.savefig(dir+'/' + info + '_rmse.png')

def evaluation(dir, X_test, Z_test):
    ### This is an example of write the point clouud .txt files from 6 different outputs of hNet, last two outputs are reliable.
    model = load_model(dir + '/model_hNet_fringe.h5', custom_objects={'rmse' : rmse})
    Z_1= model.predict(X_test[1:2], batch_size =1, verbose =1)
    for i in range(6):
        Z = np.reshape(Z_1[i], (480, 640))
        filepath = dir+'/first_' + str(i) +'.txt' 
        output = open(filepath,'w')

        for i in range(640):
            for j in range(480):
                if (Z[j,i]>50):
                    output.write(str(i))
                    output.write(" ")
                    output.write(str(480-1-j))
                    output.write(" ")
                    output.write('%.6f'%(Z[j,i]))
                    output.write("\n")
        output.close()

    ### This is an example of write the 3D .txt file from UNet.
    #model = load_model(dir + '/model_UNet_fringe.h5', custom_objects={'rmse' : rmse})
    #Z_1= model.predict(X_test[1:2], batch_size =1, verbose =1)
    #Z = np.reshape(Z_1, (480, 640))
    #filepath = dir+'/Unet_example.txt' 
    #output = open(filepath,'w')

    #for i in range(640):
    #    for j in range(480):
    #        if (Z[j,i]>50):
    #            output.write(str(i))
    #            output.write(" ")
    #            output.write(str(480-1-j))
    #            output.write(" ")
    #            output.write('%.6f'%(Z[j,i]))
    #            output.write("\n")
    #output.close()


if __name__ == '__main__':

    dir = os.path.abspath(os.curdir)
     
    X_train_fringe = np.load(dir + '/X_train_fringe.npy')
    X_train_speckle = np.load(dir + '/X_train_speckle.npy')
    Z_train = np.load(dir + '/Z_train.npy')

    X_test_fringe = np.load(dir + '/X_test_fringe.npy')
    X_test_speckle = np.load(dir + '/X_test_speckle.npy')
    Z_test = np.load(dir + '/Z_test.npy')
    
    model = hNet(480, 640, 1)
    train_model_hNet(model, dir, X_train_fringe, Z_train, 'hNet_fringe') 
    evaluation(dir, X_test_fringe, Z_test)

    #model = hNet(480, 640, 1)
    #train_model_hNet(model, dir, X_train_speckle, Z_train, 'hNet_speckle')
    #evaluation(dir, X_test_speckle, Z_test)

