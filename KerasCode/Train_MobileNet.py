import shutil
import cv2
import keras
from keras import backend as K, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet, VGG16
from keras.applications.mobilenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from google_images_download import google_images_download
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.models import load_model
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Flatten
from Utilities.Utilities import Utilities
import os
from PIL import Image
import split_folders


class Train_MobileNet:

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.mobile = keras.applications.mobilenet.MobileNet()
        self.util = Utilities()

        self.w = 224
        self.h = 224


    def prepare_image(self, file):
        img_path = ''
        img = image.load_img(img_path + file, target_size=(self.w, self.h, 3))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


    def load_image(self, img_path, show=False):
        img = image.load_img(img_path, target_size=(self.w, self.h))
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor,
                                    axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.  # imshow expects values in the range [0, 1]

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor


    def convert_h5_to_pb(self, modelo, tf=None):
        frozen_graph = self.util.freeze_session(K.get_session(), output_names=[out.op.name for out in modelo.outputs])
        tf.train.write_graph(frozen_graph, "Models/", "my_model.pb", as_text=False)


    def download_data_set(self, class_names, training_paht):
        response = google_images_download.googleimagesdownload()

        for i_class_names in class_names:
            arguments = {"keywords": i_class_names, "output_directory": training_paht, "limit": 100, "print_urls": False,
                         "format": "png", "size": "medium",
                         "o": training_paht}

            paths = response.download(arguments)


    def resize_image(self, image_path):
        for dirname in os.listdir(image_path):
            path = image_path + "/" + dirname
            for i, filename in enumerate(os.listdir(path)):
                oriimg = image.load_img(path + "\\" + filename)
                img = oriimg.resize((self.w, self.h), Image.ANTIALIAS)
                # print("Old Size: " + str(np.array(oriimg).shape) + " New Size: " + str(np.array(img).shape))
                img.save(image_path + "\\" + dirname + "\\" + dirname + "_" + str(i) + ".jpg")


    def raname_images(self, image_path):
        for dirname in os.listdir(image_path):
            path = image_path + "/" + dirname
            for i, filename in enumerate(os.listdir(path)):
                try:
                    os.rename(image_path + "/" + dirname + "/" + filename,
                              image_path + "/" + dirname + "/" + dirname + "_" + str(i) + ".jpg")
                except:
                    break


    def remove_corrupted_image(self, image_path):
        img_dir = image_path
        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            print('file folder >>>>>' + filepath)
            if filename == ".DS_Store":
                print('DS folder -----' + filepath)
                os.remove(filepath)
            else:
                for imagename in os.listdir(filepath):
                    imagepath = os.path.join(filepath, imagename)
                    try:
                        with Image.open(imagepath) as im:
                            print('Ok')
                    except:
                        print(imagepath + " -----> Removed")
                        os.remove(imagepath)


    def remove_old_dataset(self, dataset_path):
        for filename in os.listdir(dataset_path):
            try:
                foldet_to_remove = dataset_path + "/" + filename
                print(foldet_to_remove)
                shutil.rmtree(foldet_to_remove)
            except:
                print("Not Folder in the path")
                break


    def remove_files(self, files_paht):
        for filename in os.listdir(files_paht):
            try:
                print(files_paht + "/" + filename)
                os.remove(files_paht + "/" + filename)
            except:
                print("Not Files in the folder")
                break


    def remove_old_models(self, model_path):
        for folder in os.listdir(model_path):
            for i in os.listdir(model_path + "/" + folder):
                try:
                    print(model_path + "/" + folder + "/" + i)
                    os.remove(model_path + "/" + folder + "/" + i)
                except:
                    print("Not Files in the folder")
                    break


    def copy_image_to_all_class(self, src_path, dst_path):
        for folder in os.listdir(src_path):
            for i in os.listdir(src_path + "\\" + folder):
                src_dir = src_path + "\\" + folder + "\\" + i

                dst_dir = dst_path + "\\" + i

                print(src_dir)
                print(dst_dir)

                shutil.copy(src_dir, dst_dir)


    def create_mobile_net(self, type_net=0, n_class=0, alpha=0.75, depth_multiplier=1, pooling='avg', weights="imagenet",
                          include_top=False, dropout=1e-3, dropout_rate=0.001):
        if type_net == 0:
            base_model = keras.applications.mobilenet.MobileNet(input_shape=(self.w, self.h, 3), alpha=alpha,
                                                                depth_multiplier=depth_multiplier,
                                                                dropout=dropout,
                                                                pooling=pooling,
                                                                include_top=include_top,
                                                                weights=weights,
                                                                classes=n_class)

            x = base_model.output
            print(x)
            x = Dropout(dropout_rate, name='dropout')(x)  # drop=0.001
            preds = Dense(n_class, activation='softmax')(x)  # final layer with softmax activation
            model = Model(inputs=base_model.input, outputs=preds)
            return model

        elif type_net == 1:
            base_model = MobileNet(input_shape=(self.w, self.h, 3), alpha=alpha, depth_multiplier=depth_multiplier,
                                   dropout=dropout, pooling=pooling, include_top=include_top,
                                   weights=weights, classes=n_class)
            x = base_model.output
            print(x)
            x = Dropout(dropout_rate, name='dropout')(x)  # drop=0.001
            preds = Dense(n_class, activation='softmax')(x)  # final layer with softmax activation
            model = Model(inputs=base_model.input, outputs=preds)
            return model

        elif type_net == 2:
            base_model = MobileNet(weights=weights)
            x = base_model.output
            print(x)
            x = Dropout(dropout_rate, name='dropout')(x)  # drop=0.001
            preds = Dense(n_class, activation='softmax')(x)  # final layer with softmax activation
            model = Model(inputs=base_model.input, outputs=preds)
            return model
        elif type_net == 3:

            base_model = MobileNet(input_shape=(self.w, self.h, 3), alpha=alpha, depth_multiplier=depth_multiplier,
                                   dropout=dropout, pooling=pooling, include_top=include_top,
                                   weights=weights, classes=n_class)

            head_model = base_model.output
            head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
            head_model = Flatten(name="flatten")(head_model)
            head_model = Dense(128, activation="relu")(head_model)
            head_model = Dropout(0.5)(head_model)
            head_model = Dense(n_class, activation="softmax")(head_model)

            # place the head FC model on top of the base model (this will become
            # the actual model we will train)
            model = Model(inputs=base_model.input, outputs=head_model)

            return model

        elif type_net == 4:
            baseModel = VGG16(weights="imagenet", include_top=False,
                              input_tensor=Input(shape=(self.w, self.h, 3)))
            # show a summary of the base model
            print("[INFO] summary for base model...")
            print(baseModel.summary())

            head_model = baseModel.output
            head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
            head_model = Flatten(name="flatten")(head_model)
            head_model = Dense(128, activation="relu")(head_model)
            head_model = Dropout(0.5)(head_model)
            head_model = Dense(n_class, activation="softmax")(head_model)

            # place the head FC model on top of the base model (this will become
            # the actual model we will train)
            model = Model(inputs=baseModel.input, outputs=head_model)

            return model

        elif type_net == 5:
            base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=0.75, depth_multiplier=1,
                                                                dropout=0.001, pooling='avg', include_top=False,
                                                                weights="imagenet", classes=1000)
            x = base_model.output
            x = Dropout(0.001, name='dropout')(x)  # drop=0.001
            preds = Dense(1000, activation='softmax')(x)  # final layer with softmax activation
            model = Model(inputs=base_model.input, outputs=preds)
            return model

        elif type_net == 6:
            # Build the model.
            model = Sequential([
                Dense(64, activation='relu', input_shape=(224, 224, 3)),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax'),
            ])
            return model


    def print_model_estructure(self, model):
        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        for layer in model.layers:
            layer.trainable = False
        # or if we want to set the first 20 layers of the network to be non-trainable
        for layer in model.layers[:20]:
            layer.trainable = False
        for layer in model.layers[20:]:
            layer.trainable = True


    def train_model(self, model, train_path="", epochs=50, batch_size=64, class_mode='categorical', shuffle=True):
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

        train_generator = train_datagen.flow_from_directory(train_path,
                                                            target_size=(self.w, self.h),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode=class_mode,
                                                            shuffle=shuffle)

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Adam optimizer
        # loss function will be categorical cross entropy
        # evaluation metric will be accuracy

        step_size_train = 50  # train_generator.n / train_generator.batch_size
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=step_size_train,
                                      epochs=epochs)

        return history


    def save_model(self, path_stored_model, model):
        json_string = model.to_json()
        open(path_stored_model + 'model.json', 'w').write(json_string)

        # self.model.save_weights('Deep_Learning/Deep_Model/Face_model_weights.h5')
        model.save(path_stored_model + 'model_1.h5')


    def predic_class(self, model, test_image):
        new_image = self.load_image(test_image)
        pred = model.predict(new_image)
        return pred


    def convert_h5_to_tflite(self):
        keras_model_path = "Keras_Models\model_1.h5"
        tflite_path = "Tflite_Model\model.tflite"
        cmd = "tflite_convert --keras_model_file " + keras_model_path + " --output_file " + tflite_path
        os.system(cmd)


    def convert_tflite_to_kmodel(self):
        path_all_dataset = "All_IMG_Dataset" 
        tflite_path = "Tflite_Model/model.tflite"
        kmodel_path = "K_Models/model.kmodel"
        cmd = "tools/ncc.exe -i tflite -o k210model --dataset " + path_all_dataset + " " + tflite_path + " " + kmodel_path
        os.system(cmd)
        # os.startfile(cmd)


    def draw_training_loss(self, history):
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()

def main():

    train_mobilenet = Train_MobileNet()

    image_training_paht = 'dataset'
    ttv_path = 'ttv_path'  # Train-Test-Validation Path
    image_test_path = 'dataset/valid'
    all_data_set_path = 'dataset-all'
    path_stored_model = 'Keras_Models/'
    test_image = 'ttv_path/test' 
    model_path = "Models"

    print("Rename Image...")
    raname_images(image_training_paht)
    
    print("Resize Image...")
    resize_image(image_training_paht)

    split_folders.ratio(image_training_paht, output=ttv_path, seed=1337, ratio=(.8, .1, .1))

    print("Resize Train Image...")
    resize_image(ttv_path + '\\train')
    print("Resize Test Image...")
    resize_image(ttv_path + '\\test')
    print("Resize Validation Image...")
    resize_image(ttv_path + '\\val')

    class_names = os.listdir(ttv_path + '\\train')
    print(class_names)
    encoder = LabelBinarizer()
    categorical_class_name = encoder.fit_transform(class_names)
    print(categorical_class_name)

    epochs = 6
    train_or_test = True

    print("Creating Model...")
    model = train_mobilenet.create_mobile_net(type_net=0, n_class=len(class_names), dropout_rate=0.001)

    if train_or_test:
        train_mobilenet.remove_old_models(model_path)
        
        train_mobilenet.remove_old_dataset(image_training_paht)
        train_mobilenet.remove_old_dataset(image_test_path)
        train_mobilenet.remove_files(all_data_set_path)
        train_mobilenet.download_data_set(class_names, image_training_paht)
        
        train_mobilenet.remove_corrupted_image(image_training_paht)
        train_mobilenet.raname_images(image_training_paht)
        
        train_mobilenet.resize_image(image_training_paht)
        
        print("Copy Image to all folder...")
        train_mobilenet.copy_image_to_all_class(image_training_paht, all_data_set_path)

        train_mobilenet.print_model_estructure(model)

        print("Training Model...")
        history = train_mobilenet.train_model(model, ttv_path + "\\train", epochs=epochs)

        print("Save Model...")
        train_mobilenet.save_model(path_stored_model, model)

        print("Convert H5 to tflite...")
        train_mobilenet.convert_h5_to_tflite()


        print("Convert tflite to Kmodel...")
        train_mobilenet.convert_tflite_to_kmodel()

        # draw_training_loss(history)
    else:
        model = load_model('E:\Dropbox\Cognitive_Service_Project\Edge_AI_Server\Models\Keras_Models\model_1.h5')
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        for i in os.listdir(test_image):
            path_0 = test_image + "\\" + i
            for j in os.listdir(path_0):
                path = path_0 + "\\" + j
                prediction = train_mobilenet.predic_class(model, path)
                max_provavility = np.argmax(prediction[0])
                print(
                    "File Name: [ " + i + " ] " + " Predicted ConfigFile Name: [ " + class_names[
                        max_provavility] + " ] " + str(
                        prediction))
