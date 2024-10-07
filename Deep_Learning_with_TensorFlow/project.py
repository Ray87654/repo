import Multi_Dataload as data
import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, UpSampling2D, Dropout
from keras.models import Model
from keras.layers import BatchNormalization, Dense
from keras.utils  import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection   import train_test_split

if __name__ == "__main__":
    path = "C:\\Users\\User\\tensorflow(VS code)\\Project\\Traditional_Chinese" #Traditional_Chinese 的路徑
    (x_train,y_train),(x_test,y_test),list = data.load(path)
    
    path = "C:\\Users\\User\\tensorflow(VS code)\\Project\\Test_Data"
    (X_test,Y_test) = data.Test_Dataload(path, list)
    x_train , x_valid , y_train, y_valid = train_test_split(x_train ,y_train, test_size = 0.1)
    
    train_image    = x_train.astype("float32") / 255
    test_image     = x_test. astype("float32") / 255
    valid_image    = x_valid.astype("float32") / 255
    selftest_image = X_test. astype("float32") / 255

    train_label = to_categorical(y_train, num_classes = len(list))
    test_label  = to_categorical(y_test , num_classes = len(list))
    valid_label = to_categorical(y_valid, num_classes = len(list))
    
    datagen = ImageDataGenerator (
    # rescale = 1/.255
    rotation_range = 0.01 ,      #隨機旋轉的度數範圍
    # width_shift_range = 0.01 , #水平位置評移，距離上限為寬度乘以參數
    # height_shift_range = 0.1 , #垂直位置評移，距離上限為高度乘以參數
    # shear_range = 0.2 ,        #剪切強度
    # zoom_range = 0.1 ,         #圖片縮放: <1為放大、>1為縮小
    # channel_shift_range = 0.0  #通道數值偏移，用來改變圖片顏色
    # horizontal_flip = True ,   #隨機水平翻轉
    # fill_mode = "nearest" ,
    )
    
    Net = tf.keras.applications.ResNet50(include_top = False, weights = "imagenet", input_tensor = None, input_shape = (50, 50, 3))
    model = Sequential()
    # model.add(UpSampling2D(size = (2 , 2), interpolation='bilinear'))
    model.add(Net)
    model.add(BatchNormalization())
    model.add(Dropout(0.04))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(len(list), activation = "softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    # model.summary()
    
    checkpoint_filepath = "./check.h5"
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath, 
        save_weights_only = True,
        monitor = "val_accuracy", 
        mode = "max", 
        save_best_only = True
    )

    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = "val_accuracy",
        mode = "max", 
        factor = 0.1, 
        patience = 5,
        cooldown = 0,
        min_lr = 0.000001,
        verbose = 1 
    )

    callback_Earlystop = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", mode = "max", patience = 5)
    
    history = model.fit (
    datagen.flow(train_image, train_label, batch_size = 32),
    # train_image, train_label, 
    validation_data = (valid_image, valid_label),
    epochs = 100,
    verbose = 1,
    # callbacks = [callback_checkpoint, reduce_learning_rate, callback_Earlystop]
    )
    
    data.plot_acc_loss(history)
    
    predict_test = model.predict(test_image)
    data.visualizing_data(predict_test, list, test_image, y_test)
    
    predict_test = model.predict(selftest_image)
    data.visualizing_data_forTest(predict_test, list, selftest_image, Y_test)
    