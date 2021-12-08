import numpy as np
import validation_metrics as vm
import matplotlib.pyplot as plt











label_map = lambda x: 0 if x.split("\\")[0] == "normal" else 1
goldstandard =  [label_map(x) for x in validation_generator.filenames]






# INPUT
goldstandard = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1]  # List of class defined by the goldstandard




prediction = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
              0]  # List of predicted class by the Machine Learning Model


predictionprob = [-1, 0.7, -0.5, 0.9, -0.4, -0.4, -0.8, -0.1, 0.9, -0.5, -0.6, 0.2, 0.2, -0.3, 0.9, -0.5,
                  -0.6]  # List of predicted probability for the predicted class



# Confusion Matrix
# Which class represents disease
class_map = {
            "disease":1,
            "healthy":0
            }



conflist = vm.confusionlist(prediction, goldstandard,class_map)
confmat = vm.confusionmatrix(conflist,prediction)


#print(
#    '-' * 45 + f'\n Matriz de Confusão (Tabela de Contingência):\n TP: {confmat[TP]} FP: {confmat[FP]}\n FN: {confmat[FN]} TN: {confmat[TN]} \n' + '-' * 45)

SENSIVITY = vm.evaluate(vm.sensv, conflist, 'Sensivity(Recall)', printbootstrap=True)
SPECIFICITY = vm.evaluate(vm.spec, conflist, 'Specificity', printbootstrap=True)
POSITIVE_PREDITIVE_RATE = vm.evaluate(vm.positivepred, conflist, 'Positive Preditive Rate (Precision)', printbootstrap=True)
NEGATIVE_PREDITIVE_RATE = vm.evaluate(vm.negativepred, conflist, 'Negative Preditive Rate', printbootstrap=True)
ACCURACY = vm.evaluate(vm.acur, conflist, 'Accuracy', printbootstrap=True)
LRP = vm.evaluate(vm.LRP, conflist, 'Likelihood Ratio +', False, printbootstrap=True)
LRN = vm.evaluate(vm.LRN, conflist, 'Likelihood Ratio -', False, printbootstrap=True)
TYPE_1_ERROR = vm.evaluate(vm.type1, conflist, 'Type 1 Error', printbootstrap=True)
TYPE_2_ERROR = vm.evaluate(vm.type2, conflist, 'Type 2 Error', printbootstrap=True)



#Aqui irá gerar os dados para a ROC
# ROC Curve/Precision Recall Curve
predictiondyn = prediction[:]
youd = list()
false_positive_rate = list()
true_positive_rate = list()
precision = list()

for cutpoint in range(0, 21, 1):
    
    cutpoint = (cutpoint - 10) / 10
    print(f'Cutpoint: {cutpoint}')
    for i, p in enumerate(predictiondyn):
        
        if class_map['disease'] == prediction[i]:
            if cutpoint > predictionprob[i]:
                predictiondyn[i] = class_map['healthy']
            elif cutpoint <= predictionprob[i]:
                predictiondyn[i] = class_map['disease']
        if class_map['healthy'] == prediction[i]:
            if cutpoint <= predictionprob[i]:
                predictiondyn[i] = class_map['disease']
            elif cutpoint > predictionprob[i]:
                predictiondyn[i] = class_map['healthy']

    #    print(predictiondyn)
    #    print(goldstandard)
    ci = vm.confusionlist(predictiondyn, goldstandard,class_map)
    c = vm.confusionmatrix(ci,prediction)
    #    print(ci)
    #    print(c)

    vm.evaluate(vm.sensv, ci, 'Sensivity(Recall)')
    vm.evaluate(vm.spec, ci, 'Specificity')
    vm.evaluate(vm.positivepred, ci, 'Positive Preditive Rate')
    vm.evaluate(vm.negativepred, ci, 'Negative Preditive Rate')
    vm.evaluate(vm.acur, ci, 'Accuracy')
    vm.evaluate(vm.LRP, ci, 'Likelihood Ratio +', False)
    vm.evaluate(vm.LRN, ci, 'Likelihood Ratio -', False)
    vm.evaluate(vm.type1, ci, 'Type 1 Error')
    vm.evaluate(vm.type2, ci, 'Type 2 Error')

    youden = vm.sensv(ci) + vm.spec(ci) - 100
    print(f'Youden Index: {youden}')

    youd.append(youden)
    false_positive_rate.append(vm.type1(ci))
    true_positive_rate.append(vm.sensv(ci))
    precision.append(vm.positivepred(ci))

    del ci[:]

#ROC Curve/Precision-Recall

false_positive_rate2 = [0.01*x for x in false_positive_rate]
true_positive_rate2 = [0.01*x for x in true_positive_rate]
precision2 = [0.01*x for x in precision]


x = false_positive_rate2
y = true_positive_rate2
z = precision2

# This is the ROC curve
plt.subplot(1,2,1)
plt.title('ROC Curve')
plt.xlabel('1 -Specificity')
plt.ylabel('Sensivity')

plt.plot(x,y,'bo')
plt.plot(x,y)
plt.plot(x,x,'g--')
#plt.savefig('/Users/rafaelscherer/Desktop/ROC.png')

plt.subplot(1,2,2)
plt.title('Precision Recall Curve')
plt.ylabel('PPV (Precision)')
plt.xlabel('Sensivity')
plt.plot(y,z,'bo')
plt.plot(y,z)
#plt.savefig('/Users/rafaelscherer/Desktop/Precion_Recall.png')

# This is the AUC
auc = round(np.trapz(y,x),3)
print(f'Area Under Curve: {auc}')
youdstring = str(round(np.argmax(youd)/10-1,1))
youdmax = round(max(youd),2)
print(f'Cutpoint - Youden Index: (' + youdstring + ') - ('+ str(youdmax) + ')')