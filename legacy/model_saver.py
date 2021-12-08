# -*- coding: utf-8 -*-
from model_utils.cam_models import build_vgg16_GAP
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
from tensorflow import keras
from tensorflow.keras import backend as K
import tarfile



#Em casos onde salvamos apenas os pesos
model = build_vgg16_GAP(input_shape=(256, 256, 3))
model.load_weights('./cam_models_checkpoints/best_retina_net_1.h5')





#################################
#SALVAR O MODELO NORMAL_ALTERADO#
#################################
class_output = model.output[:,1]
output_conv_layer = model.get_layer('block5_conv2')#Esse é o nome do ultimo layer de convoluçõa da rede

grads = keras.backend.gradients(class_output, output_conv_layer.get_output_at(-1))[0]

pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))


##AQUI GERENCIO A PARTE DE SALVAR O MODELO EM TENSORFLOW

# Note: This directory structure will need to be followed - see notes for the next section
model_version = '1'
export_dir = 'export/Servo/' + model_version


# Build the Protocol Buffer SavedModel at 'export_dir'
builder = builder.SavedModelBuilder(export_dir)

# Create prediction signature to be used by TensorFlow Serving Predict API
signature = predict_signature_def(
                                    inputs={"inputs": model.input}, 
                                    
                                    outputs={"score": model.output,
                                             "pooled_grads":pooled_grads,
                                             "last_conv_values": output_conv_layer.get_output_at(-1)}
                                  )


with K.get_session() as sess:
    # Save the meta graph and variables
    builder.add_meta_graph_and_variables(
        sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
    builder.save()
    

with tarfile.open('normal_alterado_cam.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)








##################################
#sALVAR O MODELO vALIDO_NAOVALIDO#
##################################
#model_version = '1'
#export_dir = 'export/Servo/' + model_version
#
#
## Build the Protocol Buffer SavedModel at 'export_dir'
#builder = builder.SavedModelBuilder(export_dir)
#
## Create prediction signature to be used by TensorFlow Serving Predict API
#signature = predict_signature_def(
#                                    inputs={"inputs": model.input}, 
#                                    
#                                    outputs={"score": model.output})
#                                             
#
#
#with K.get_session() as sess:
#    # Save the meta graph and variables
#    builder.add_meta_graph_and_variables(
#        sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
#    builder.save()
#    
#
#with tarfile.open('valido_naovalido.tar.gz', mode='w:gz') as archive:
#    archive.add('export', recursive=True)











































