import os 

GEN_IMG = True
DS = 'unifesp_classificadas'
FILTER = 'raw'
NLAYERS = 9
PATO = 'rd'
SOURCE = 'data/'+DS+'/'
DEST = 'data/'+DS+'/algo/'
FILESN = os.listdir(SOURCE+'normais')
FILESP = os.listdir(SOURCE+'alteradas/'+PATO)
PROPORTION = 1 #2.82
IMGN = 5000 #len(FILESN)


PATH = os.getcwd()
# Define data path
DATA_PATH = './' + DEST
EPOCHS=30
BATCH_SIZE = 64
TARGET_SIZE = 299
THRESHOLDS = 200
 #eyepacs 2.42


# DEVICES NAMES
H_RESO = ['20sus', '021sus', '60sus', '70sus', '80sus']
L_RESO = ['30_', '50_', '60ses', '70ses', '80ses']

TYPETRAIN = 'n' # 'ft' -> fine tunning, or 'n' -> normal or 'tf' -> transfer learning
TYPEIMG = 'l' # -> high reso, 'l' -> low_reso