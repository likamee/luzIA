# Rafael Scherer
# Métricas IA
"""Este Programa com INPUT sendo as classes de uma lista de dados, as predições
pelo modelo de machine learning da classe de cada valor na lista além da probabilidade
predita para a classe definida pelo modelo, retorna a MATRIZ DE CONFUSÃO, as MÉTRICAS derivadas da matriz,
além de Curva ROC/Precision-Recall e o ponto de corte ótimo do exame com base no Índice de Youden, e os
valores de intervalo de confiança de cada métrica baseados em Escore de Wilson confirmado pela técnica de Bootstrap"""

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# INPUT
""" goldstandard = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1]  # List of class defined by the goldstandard
prediction = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]  # List of predicted class by the Machine Learning Model
predictionprob = [-1, 0.7, -0.5, 0.9, -0.4, -0.4, -0.8, -0.1, 0.9, -0.5, -0.6, 0.2, 0.2, -0.3, 0.9, -0.5,
                  -0.6]  # List of predicted probability for the predicted class """

disease = 1
healthy = 0

# Round Numbers of a list
def arredonda(met):
    for i, c in enumerate(met):
        arred = round(c, 1)
        met[i] = arred
    return met


# Confidence Intervals - Wilson Score
def wilson(p, n, z=1.96):
    p = p / 100
    n = len(n)
    denominator = 1 + z ** 2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    lower_bound = round((centre_adjusted_probability - z * adjusted_standard_deviation) / denominator, 1) * 100
    upper_bound = round((centre_adjusted_probability + z * adjusted_standard_deviation) / denominator, 1) * 100
    return (lower_bound, upper_bound)


matchlist = list()

#Confusion Matrix
def confusionlist(prediction, goldstandard):
    for i, p in enumerate(prediction):
        if disease == p:
            if p == goldstandard[i]:
                matchlist.append(0)  # TP
            elif p != goldstandard[i]:
                matchlist.append(1)  # FP
        if disease != p:
            if p == goldstandard[i]:
                matchlist.append(2)  # TN
            elif p != goldstandard[i]:
                matchlist.append(3)  # FN
    return matchlist

def confusionmatrix(conflist,prediction):
    """
    Função utilizada para o calculo da matriz de confussão.
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, p in enumerate(prediction):
        if conflist[i] == 0:
            TP += 1
        if conflist[i] == 1:
            FP += 1
        if conflist[i] == 2:
            TN += 1
        if conflist[i] == 3:
            FN += 1
    
    return {
            "TP":TP, 
            "FP":FP,
            "TN":TN,
            "FN":FN
            }

# Metrics
#Sensivity
def sensv(lista):
    TP = lista.count(0)
    FN = lista.count(3)
    if TP + FN > 0:
        sensivity = 100 * TP / (TP + FN)
    else:
        sensivity = 0
    return sensivity

#Specificity
def spec(lista):
    TN = lista.count(2)
    FP = lista.count(1)
    if TN + FP > 0:
        specificity = 100 * TN / (TN + FP)
    else:
        specificity = 0
    return specificity

#Positive Predictive Value
def positivepred(lista):
    TP = lista.count(0)
    FP = lista.count(1)
    if TP + FP > 0:
        PPV = 100 * TP / (TP + FP)
    else:
        PPV = 0
    return PPV

#Negative Predictive Value
def negativepred(lista):
    TN = lista.count(2)
    FN = lista.count(3)
    if TN + FN > 0:
        PNV = 100 * TN / (TN + FN)
    else:
        PNV = 0
    return PNV

#Accuracy
def acur(lista):
    TN = lista.count(2)
    FN = lista.count(3)
    TP = lista.count(0)
    FP = lista.count(1)
    Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    return Accuracy

#Likelihood Ratio +
def LRP(lista):
    sensivity = sensv(lista)
    specificity = spec(lista)
    if 100 - specificity > 0:
        LRpos = sensivity / (100 - specificity)
    else:
        LRpos = 0
    return LRpos

#Likelihood Ratio -
def LRN(lista):
    sensivity = sensv(lista)
    specificity = spec(lista)
    if specificity > 0:
        LRneg = (100 - sensivity) / specificity
    else:
        LRneg = 0
    return LRneg

#Type 1 error
def type1(lista):
    specificity = spec(lista)
    type1error = 100 - specificity
    return type1error

#Type 2 Error
def type2(lista):
    sensivity = sensv(lista)
    type2error = 100 - sensivity
    return type2error

#General function
def evaluate(function, lista, namefunction='', printwilson=True, printbootstrap=False):
    
    conf = 0
    s = round(function(lista), 1)
    w = wilson(s, lista)

    print(f'- {namefunction}: {s} ', end='')
    if printwilson == True:
        print(f'- Wilson Score: {w} ', end='')
    if printbootstrap == True:
        print(f'- Bootstrap: {conf}')
    else:
        print('')
    return s, wilson, conf


def metrics_rc(goldstandard, prediction, predictionprob):

	# Confusion Matrix
	# Which class represents disease

	conflist = confusionlist(prediction, goldstandard)
	conflist = [0] * 1292
	#913
	conflist[691:893] = [1] * 202
	conflist[893:1245] = [2] * 352
	conflist[1245:]  = [3] * len(conflist[1245:])

	#confmat = confusionmatrix(conflist, prediction)
	#print('-' * 45 + f'\n Matriz de Confusão (Tabela de Contingência):\n TP: {confmat["TP"]} FP: {confmat["FP"]}\n FN: {confmat["FN"]} TN: {confmat["TN"]} \n' + '-' * 45)

	print('-' * 45 + f'\n Matriz de Confusão (Tabela de Contingência):\n TP: {conflist.count(0)} FP: {conflist.count(1)}\n FN: {conflist.count(3)} TN: {conflist.count(2)} \n' + '-' * 45)

	evaluate(acur, conflist, 'Accuracy')
	evaluate(spec, conflist, 'Specificity')
	evaluate(sensv, conflist, 'Sensivity(Recall)')
	evaluate(positivepred, conflist, 'Positive Preditive Rate (Precision)')
	evaluate(negativepred, conflist, 'Negative Preditive Rate')
	evaluate(LRP, conflist, 'Likelihood Ratio +', False)
	evaluate(LRN, conflist, 'Likelihood Ratio -', False)
	evaluate(type1, conflist, 'Type 1 Error')
	evaluate(type2, conflist, 'Type 2 Error')

	youden = sensv(conflist) + spec(conflist) - 100
	print(youden)

	youd = list()
	false_positive_rate = list()
	true_positive_rate = list()
	precision = list()

	""" # ROC Curve/Precision Recall Curve
	predictiondyn = prediction[:]
	

	for cutpoint in range(0, 21, 1):
		cutpoint = (cutpoint - 10) / 10
		print(f'Cutpoint: {cutpoint}')
		for i, p in enumerate(predictiondyn):
			if disease == prediction[i]:
					if cutpoint > predictionprob[i]:
						predictiondyn[i] = healthy
					elif cutpoint <= predictionprob[i]:
						predictiondyn[i] = disease
			if healthy == prediction[i]:
					if cutpoint <= predictionprob[i]:
						predictiondyn[i] = disease
					elif cutpoint > predictionprob[i]:
						predictiondyn[i] = healthy

		#    print(predictiondyn)
		#    print(goldstandard)
		#    ci = confusionlist(predictiondyn, goldstandard)
		#    print(ci)
		#    print(c)

		evaluate(sensv, ci, 'Sensivity(Recall)')
		evaluate(spec, ci, 'Specificity')
		evaluate(positivepred, ci, 'Positive Preditive Rate')
		evaluate(negativepred, ci, 'Negative Preditive Rate')
		evaluate(acur, ci, 'Accuracy')
		evaluate(LRP, ci, 'Likelihood Ratio +', False)
		evaluate(LRN, ci, 'Likelihood Ratio -', False)
		evaluate(type1, ci, 'Type 1 Error')
		evaluate(type2, ci, 'Type 2 Error')

		youden = sensv(ci) + spec(ci) - 100
		print(f'Youden Index: {youden}')

		youd.append(youden)
		false_positive_rate.append(type1(ci))
		true_positive_rate.append(sensv(ci))
		precision.append(positivepred(ci))

		del ci[:]

	#ROC Curve/Precision-Recall """

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
	""" youdstring = str(round(np.argmax(youd)/10-1,1))
	youdmax = round(max(youd),2)
	print(f'Cutpoint - Youden Index: (' + youdstring + ') - ('+ str(youdmax) + ')') """