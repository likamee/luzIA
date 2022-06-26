import os 

GEN_IMG = False
DS = 'refuge_data' #'unifesp_classificadas'
FILTER = 'raw'
NLAYERS = 5
PATO = 'escavacao'
SOURCE = 'data/'+DS+'/'
DEST = 'data/'+DS+'/algo/'
FILESN = os.listdir(SOURCE+'normais')
FILESP = os.listdir(SOURCE+'alteradas/'+PATO)
PROPORTION = 2.82 #2.82
IMGN = len(FILESN)


PATH = os.getcwd()
# Define data path
DATA_PATH = './' + DEST
EPOCHS=30
BATCH_SIZE = 16
TARGET_SIZE = 299
THRESHOLDS = 200
#eyepacs 2.42


# DEVICES NAMES
H_RESO = ['20sus', '021sus', '60sus', '70sus', '80sus', '30_', '60ses', '70ses', '80ses'] # '30_', '60ses', '70ses', '80ses'
L_RESO = ['50_']

TYPETRAIN = 'n' # 'ft' -> fine tunning, or 'n' -> normal or 'tf' -> transfer learning
TYPEIMG = 'h' # 'h' -> high reso, 'l' -> low_reso