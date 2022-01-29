from tensorflow.keras import callbacks
from model_utils import cam_models
from model_utils import filter_img
from model_utils.load_data import gen_data
from model_utils.generic_funcs import remove_files_dir

import config as cfg



def process_train():
    if cfg.GEN_IMG:
        remove_files_dir(cfg.DATA_PATH+'/normais')
        remove_files_dir(cfg.DATA_PATH+'/'+cfg.PATO)
        remove_files_dir(cfg.SOURCE+'/excluidas')
        filter_img.apply_filter(cfg.FILTER, cfg.PATO, cfg.SOURCE, cfg.FILESN, cfg.FILESP, cfg.IMGN, cfg.PROPORTION, cfg.H_RESO, cfg.L_RESO, cfg.TYPEIMG)

    train_generator, validation_generator = gen_data()
    ##CNN
    NAMETEST = cfg.DS+'_'+cfg.FILTER+'_'+str(cfg.NLAYERS)
    MODEL, NAMETEST = cam_models.build_vgg16_GAP(cfg.NLAYERS, cfg.TYPETRAIN, NAMETEST) #best 9 layers15
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