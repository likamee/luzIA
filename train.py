from tensorflow.keras import callbacks
from model_utils import cam_models
from model_utils import filter_img
from model_utils.load_data import gen_data

import config as cfg



def process_train():
    if cfg.GEN_IMG:
        filter_img.apply_filter(cfg.FILTER, cfg.PATO, cfg.SOURCE, cfg.FILESN, cfg.FILESP, cfg.IMGN, cfg.IMGP)

    train_generator, validation_generator = gen_data()
    ##CNN
    MODEL = cam_models.build_vgg16_GAP(cfg.NLAYERS) #best 9 layers15

    NAMETEST = cfg.DS+'_'+cfg.FILTER+'_'+str(cfg.NLAYERS)
    FILENAME=NAMETEST+'.csv'
    CSVLOG=callbacks.CSVLogger(FILENAME, separator=',', append=False)
    #early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
    FPATH=NAMETEST+'.hdf5'
    CHECKP = callbacks.ModelCheckpoint(FPATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    CALLBLIST = [CSVLOG,CHECKP]


    #class_weight = {0:1, 1:2.8}
    MODEL.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // cfg.BATCH_SIZE,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // cfg.BATCH_SIZE,
        epochs = cfg.EPOCHS,
        callbacks=[CALLBLIST])
        #class_weight=class_weight)
    # Evaluating the MODEL.
    del MODEL  # deletes the existing MODEL