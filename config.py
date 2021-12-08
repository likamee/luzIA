import os 

GEN_IMG = False
DS = 'unifesp_classificadas'
FILTER = 'raw'
NLAYERS = 3
PATO = 'rd'
SOURCE = 'data/'+DS+'/'
DEST = 'data/'+DS+'/algo/'
FILESN = os.listdir(SOURCE+'normais')
FILESP = os.listdir(SOURCE+'alteradas/'+PATO)
IMGN = 48784
IMGP = 17152

PATH = os.getcwd()
# Define data path
DATA_PATH = './' + DEST
EPOCHS=30
BATCH_SIZE = 64
TARGET_SIZE = 299
THRESHOLDS = 200
PROPORTION = 2.82 #eyepacs 2.42



