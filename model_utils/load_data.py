import config as cfg
from keras_preprocessing.image import ImageDataGenerator

def gen_data(EVAL=False):
    SPLIT = 0.2 if not EVAL else 0.1
    TRAIN_DTGEN = ImageDataGenerator(
            fill_mode='nearest',
            validation_split=SPLIT) # set validation split

    TRAINGEN = TRAIN_DTGEN.flow_from_directory(
        cfg.DATA_PATH,
        target_size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode="categorical",
        interpolation = 'nearest',
        subset='training') # set as training data

    VALGEN = TRAIN_DTGEN.flow_from_directory(
        cfg.DATA_PATH, # same directory as training data
        target_size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode="categorical",
        interpolation = 'nearest',
        subset='validation') # set as validation data
    
    return TRAINGEN, VALGEN