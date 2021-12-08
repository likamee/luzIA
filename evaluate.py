from tensorflow.python.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import SGD

import config as cfg
import custom_metrics as cm
from model_utils.load_data import gen_data
from model_utils import cam_models

def generate_thresholds(num_thresholds, kepsilon=1e-7):
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    return [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]



THRESHOLDS = generate_thresholds(cfg.THRESHOLDS, 0.0000001)
THRESHOLDS[0] = 0

METRICS_EVAL = [      
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc'),
      TruePositives(thresholds=THRESHOLDS,name='tp'),
      FalsePositives(thresholds=THRESHOLDS,name='fp'),
      TrueNegatives(thresholds=THRESHOLDS,name='tn'),
      FalseNegatives(thresholds=THRESHOLDS,name='fn'),
      Precision(name='pr'),
      Recall(name='recall'),
    ]


def evaluate():
    #evaluate
    MODEL = 'results/eye_raw12.hdf5'
    print('Loading the model')
    MODEL = load_model(MODEL)
    OPT = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    MODEL.compile(loss='binary_crossentropy',optimizer=OPT, metrics=[METRICS_EVAL])

    """ 
    Y_PRED = MODEL.predict(validation_generator)
    Y_PRED = [np.where(np.round(r.tolist())==1)[0][0] for r in Y_PRED]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator.classes, Y_PRED)
    auc_keras = auc(fpr_keras, tpr_keras)


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show() """

    loss, acc, auc, tp, fp, tn, fn, pr, recall = MODEL.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE, verbose=2)

    sensitivities, specificities = [], []
    for i in range(len(thresholds)):
        if tp[i] + fn[i] > 0:
            sensivity = 100 * tp[i] / (tp[i] + fn[i])
        else:
            sensivity = 0
        sensitivities.append(sensivity)

        if tn[i] + fp[i] > 0:
            specificity = 100 * tn[i] / (tn[i] + fp[i])
        else:
            specificity = 0

        specificities.append(specificity)

    x = 0
