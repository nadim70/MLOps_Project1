#!/usr/bin/env python
# coding: utf-8

def load():
    from keras.applications import MobileNet
    model = MobileNet(weights = 'imagenet',include_top = False,input_shape=(224,224,3))
    for layers in model.layers:
        print(layers.__class__.__name__)
    for layers in model.layers:
        layers.trainable = False
    for (i,layer) in enumerate(model.layers):
        print(str(i)+" "+layer.__class__.__name__,layer.trainable)
    return model

def newmodel(model):
    from keras.layers import Dense,Activation, Flatten, GlobalAveragePooling2D
    top_model = model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(2,activation='softmax')(top_model)
    return top_model

def final_model(model,modify_mod):
    from keras.models import Model
    fin_model = Model(inputs=model.input,outputs=modify_mod)
    return(fin_model)


def data_gen(train_data,validation_data):
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=45,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)
    batch_size=32
    train_generator = train_datagen.flow_from_directory(train_data,
                                                        target_size=(224,224),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(validation_data,
                                                                 target_size=(224,224),
                                                                 batch_size=batch_size,
                                                                 class_mode='categorical')
    return(train_generator,validation_generator)

def calc_epoch(train_generator,validation_generator,fin_model):
    from keras.optimizers import RMSprop
    fin_model.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    fin_model.fit_generator(train_generator,
                        steps_per_epoch= 200,
                        epochs=1,
                        validation_data=validation_generator,
                        validation_steps= 50)

def result():
    model = load()
    modify_mod = newmodel(model)
    fin_model = final_model(model,modify_mod)
    train_data = '/root/train'
    validation_data = '/root/validation'
    train_generator,validation_generator=data_gen(train_data,validation_data)
    calc_epoch(train_generator,validation_generator,fin_model)

result()
