import pandas as pd
import numpy as np

# Pre processing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# CART
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image  
import graphviz
import pydotplus

# Kfold
from sklearn.model_selection import StratifiedKFold, KFold

# GS
from sklearn.model_selection import GridSearchCV

# FS
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV

# Resultados
from sklearn import metrics
from sklearn.metrics import classification_report

# Wilcoxon test
from scipy.stats import wilcoxon

# Warnings
import warnings

random_st = 3 # Para reprodutividade
debug = False

########################################### *** PRE PROCESSAMENTO *** ###########################################

# Pre processamento

def pre_proc(data):
	
	# Pre processamento: B->0, M->1

	le = preprocessing.LabelEncoder()
	le.fit(data.diagnosis)

	# print(le.transform(['B','M'])) #Checking which is the positive class

	data.diagnosis = le.transform(data.diagnosis)
	
	# Pre processamento: Separando atributo-classe

	# hasDuplicated = data.duplicated() # Acho que nao serve pra nada

	X = data.drop(columns=['id','diagnosis'])
	Y = data.diagnosis
	
	return X, Y

# Normalizacao
	
def norm(X_train, X_test):
	
	# Scaling
	scaler = MinMaxScaler()
	scaler = scaler.fit(X_train)

	# Copying in order not to affect original data
	X_train_normal = X_train.copy(deep=True)
	Xnormal = X_test.copy(deep=True)
	
	X_train_normal.loc[:, :] = scaler.transform(X_train.loc[:, :])
	Xnormal.loc[:, :] = scaler.transform(X_test.loc[:, :])
	
	return X_train_normal, Xnormal
	
def save_data(data):
	
	X = data.drop(columns=['id','diagnosis'])
	Y = data.diagnosis
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_st) # Separando treinamento e teste
	
	X_train_normal, Xnormal = norm(X_train, X_test)
	
	X_train_normal['diagnosis'] = Y_train
	Xnormal['diagnosis'] = Y_test
	
	X_train_normal.to_csv('train.csv',index=False)
	Xnormal.to_csv('test.csv',index=False)

########################################### *** CART *** ###########################################
    
# CART BASELINE

def CART_baseline(X_train, Y_train, X_test, Y_test, filename):
	
	cart = DecisionTreeClassifier(random_state=random_st)
	cart.fit(X_train, Y_train)
	X_preds = cart.predict(X_test)

	# Calculating metrics
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		CART_accuracy = metrics.accuracy_score(X_preds, Y_test)
		CART_fscore = metrics.f1_score(X_preds, Y_test)
		CART_precision = metrics.precision_score(X_preds, Y_test)
		CART_recall = metrics.recall_score(X_preds, Y_test)

	print(classification_report(X_preds, Y_test))
	print('CART_accuracy: %0.10f' % CART_accuracy)
	print('CART_precision: %0.10f' % CART_precision)
	print('CART_recall: %0.10f' % CART_recall)
	print('CART_fscore: %0.10f' % CART_fscore)
	
	dot_data = StringIO()
	export_graphviz(cart, out_file=dot_data,  
					filled=True, rounded=True, special_characters=True,
					feature_names = list(X_train),class_names=['0','1'])
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	graph.write_png(filename)
	Image(graph.create_png())
	
	return CART_accuracy, CART_precision, CART_fscore, CART_recall

# CART GS

def CART_GS(X_train, Y_train, X_test, Y_test, filename, selected_features=[], n_features=0):

	CART_parameter_space = {
		'min_samples_split': [2, 4, 8, 16, 32, 64],
		'criterion': ['gini', 'entropy'],
		'splitter': ['best', 'random'],
		'random_state': [random_st]
	}

	clf = DecisionTreeClassifier(random_state=random_st)

	CART_accuracy, CART_precision, CART_fscore, CART_recall, CART_Y_pred, CART_selected_feats, CART_best_params, CART_best_estimator = execute_feature_selection(X_train, Y_train, X_test, Y_test, clf, CART_parameter_space, selected_features, n_features)

	if debug is True:
		print('*** CART GS ***')
		print('CART_best_params')
		print(CART_best_params)
		print('CART_selected_feats')
		print(X_train.columns[CART_selected_feats].tolist())
		print(classification_report(Y_test, CART_Y_pred))
		print('CART_accuracy: %0.10f' % CART_accuracy)
		print('CART_precision: %0.10f' % CART_precision)
		print('CART_recall: %0.10f' % CART_recall)
		print('CART_fscore: %0.10f' % CART_fscore)
		
		dot_data = StringIO()
		export_graphviz(CART_best_estimator, out_file=dot_data,
			filled=True, rounded=True, special_characters=True,
			feature_names = X_train.columns[CART_selected_feats].tolist(), class_names=['0','1'])
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
		graph.write_png(filename)
		Image(graph.create_png())
	
	return CART_accuracy, CART_precision, CART_fscore, CART_recall, CART_selected_feats, CART_best_params
	
def execute_CART_GS(X_train, Y_train, filename, selected_features=[], n_features=0):
	
	CART_accuracy = []
	CART_precision = []
	CART_fscore = []
	CART_recall = []
	CART_selected_feats = []
	CART_best_params = []
	
	tree_idx = 0
	
	for X_train_normal, Y_train, X_validation_normal, Y_validation in kfold(X_train, Y_train): # normaliza dentro de cada fold
		
		filename_idx = str(filename)+str(tree_idx+1)+".png"
		accuracy, precision, fscore, recall, selected_feats, best_params = CART_GS(X_train_normal, Y_train, X_validation_normal, Y_validation, filename_idx, selected_features, n_features)
		tree_idx += 1
		
		CART_accuracy.append(accuracy)
		CART_precision.append(precision)
		CART_fscore.append(fscore)
		CART_recall.append(recall)
		CART_selected_feats.append(selected_feats)
		CART_best_params.append(best_params)

	CART_selected_feats_final, CART_best_params_final = get_best_criteria(CART_accuracy, CART_fscore, CART_precision, CART_recall, CART_best_params, CART_selected_feats)
	
	return CART_selected_feats_final, CART_best_params_final
	
########################################### *** KFOLD *** ###########################################

def kfold(X_train_full, Y_train_full):

	kf = KFold(n_splits=10, random_state=random_st)

	for train_idx, validation_idx in kf.split(X_train_full):
		
		X_train, X_validation = X_train_full.iloc[train_idx, :].copy(deep=True), X_train_full.iloc[validation_idx, :].copy(deep=True)
		Y_train, Y_validation = Y_train_full.iloc[train_idx].copy(deep=True), Y_train_full.iloc[validation_idx].copy(deep=True)

		scaler = MinMaxScaler()
		scaler = scaler.fit(X_train)
		
		X_train.loc[:, :] = scaler.transform(X_train.loc[:, :])
		X_validation.loc[:, :] = scaler.transform(X_validation.loc[:, :])
		
		yield X_train, Y_train, X_validation, Y_validation

########################################### *** GRID SEARCH *** ###########################################

# Grid Search

def grid_search(X, Y, model, parameter_space, n_splits=5):
  
	kfold_splitter = StratifiedKFold(n_splits=n_splits,random_state=random_st)

	clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=kfold_splitter, scoring = 'f1')
	clf.fit(X,Y)

	return clf.best_params_, clf.best_estimator_

########################################### *** FEATURE SELECTION *** ###########################################

# Feature Selection

def feature_selection(X,Y, n_features=0):
	
	model = LogisticRegression(solver = 'lbfgs',random_state=random_st)
	
	if n_features == 0:
		
		kfold_splitter = StratifiedKFold(n_splits=5,random_state=random_st)
	
		rfe = RFECV(model, cv=kfold_splitter)
	
	else:
		
		rfe = RFE(model, n_features)
	
	fit = rfe.fit(X, Y)
	criteria = fit.support_
	
	return criteria
	
# Seleciona os melhores atributos e parametros	

def get_best_criteria(accuracy, fscore, precision, recall, best_parameters, selected_features):
	# 0 accuracy	# 1 fscore	# 2 precision	# 3 recall
	metric_idx = 1 #fscore
	
	sorted_selected_features = sorted(list(zip(accuracy, fscore, precision, recall, best_parameters, selected_features)),key=lambda x: x[metric_idx])
	
	return sorted_selected_features[-1][-1], sorted_selected_features[-1][-2] #selected features, selected parameters
		
# Executa o Feature Selection

def execute_feature_selection(X_train, Y_train, X_validation, Y_validation, clf, parameter_space, selected_features, n_features=0):
	
	if not selected_features: #nao usa os atributos selecionados pela visualizacao

		selected_feats = feature_selection(X_train, Y_train, n_features)
		
		X_train = X_train.iloc[:, selected_feats]
		X_validation = X_validation.iloc[:,selected_feats]
			
		best_params, best_estimator = grid_search(X_train, Y_train, clf, parameter_space)

		Y_pred = best_estimator.predict(X_validation)
		
		# Calculating metrics
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			accuracy = metrics.accuracy_score(Y_validation, Y_pred)
			fscore = metrics.f1_score(Y_validation, Y_pred)
			precision = metrics.precision_score(Y_validation, Y_pred)
			recall = metrics.recall_score(Y_validation,Y_pred)
			
		return accuracy, precision, fscore, recall, Y_pred, selected_feats, best_params, best_estimator
	
	#usa os atributos selecionados pela visualizacao
	
	selected_feats = np.isin(X_train.columns, (selected_features))
	
	X_train = X_train.iloc[:, selected_feats]
	X_validation = X_validation.iloc[:,selected_feats]
		
	best_params, best_estimator = grid_search(X_train, Y_train, clf, parameter_space)

	Y_pred = best_estimator.predict(X_validation)
	
	# Calculating metrics
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		accuracy = metrics.accuracy_score(Y_validation, Y_pred)
		fscore = metrics.f1_score(Y_validation, Y_pred)
		precision = metrics.precision_score(Y_validation, Y_pred)
		recall = metrics.recall_score(Y_validation,Y_pred)
		
	return accuracy, precision, fscore, recall, Y_pred, selected_feats, best_params, best_estimator
	
########################################### *** EXECUCAO TREINAMENTO *** ###########################################
				
# Executa todos modelos usando HOLDOUT para treinamento/teste

def execute_baseline(X, Y):
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_st) # Separando treinamento e teste
	
	X_train_normal, Xnormal = norm(X_train, X_test) # normalizando
	
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		
		print("*** CART BASELINE ***")
		
		CART_base = CART_baseline(X_train_normal, Y_train, Xnormal, Y_test, "./img/test/CART_baseline.png")
		
		print(" ")
		
	return CART_base
	
# Executa todos modelos usando HOLDOUT para treinamento/teste e CV=10 para FS+GS

def execute_training_GS(X, Y, CART_base):
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_st) # Separando treinamento e teste
	
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		
		selected_features = []
		
		print("*** CART RFE(CV) ***\ntraining ...\n")
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_RFE(CV)_")
		CART_RFE_CV = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_RFE(CV).png", "*** CART RFE(CV) TEST ***")
		
		#~ ########### **** ##############
		
		#~ # Executa com o n_features = 3
		
		print("*** CART RFE(3) ***\ntraining ...\n")
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_RFE(3)_", selected_features, 3)
		CART_RFE_3 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_RFE(3).png", "*** CART RFE(3) TEST ***")
		
		#~ ########### **** ##############
		
		#~ # Executa com o n_features = 5
		
		print("*** CART RFE(5) ***\ntraining ...\n")
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_RFE(5)_", selected_features, 5)
		CART_RFE_5 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_RFE(5).png", "*** CART RFE(5) TEST ***")
		
		#~ ########### **** ##############
		
		print("*** CART LP(3) ***\ntraining ...\n")
		
		selected_features = ['radius_worst', 'concavity_mean', 'concavity_se']
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_LP(3)_", selected_features, 3)
		CART_LP_3 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_LP(3).png", "*** CART LP(3) TEST ***")
		
		#~ ########### **** ##############
		
		print("*** CART RV(3) ***\ntraining ...\n")
		
		selected_features = ['perimeter_mean', 'area_worst', 'concave points_mean']
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_RV(3)_", selected_features, 3)
		CART_RV_3 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_RV(3).png", "*** CART RV(3) TEST ***")
		
		#~ ########### **** ##############
		
		print("*** CART LP(5) ***\ntraining ...\n")
		
		selected_features = ['radius_worst', 'concavity_mean', 'perimeter_mean', 'concavity_se', 'fractal_dimension_worst']
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_LP(5)_", selected_features, 5)
		CART_LP_5 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_LP(5).png", "*** CART LP(5) TEST ***")
		
		#~ ########### **** ##############
		
		print("*** CART RV(5) ***\ntraining ...\n")
		
		selected_features = ['radius_mean', 'concavity_mean', 'area_worst', 'fractal_dimension_mean', 'concavity_se']
		
		CART_selected_feats_final, CART_best_params_final = execute_CART_GS(X_train, Y_train, "./img/train/CART_RV(5)_", selected_features, 5)
		CART_RV_5 = execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, "./img/test/CART_RV(5).png", "*** CART RV(5) TEST ***")
		
		#~ ########### **** ##############
		
		print("*** Wilcoxon test ***\n")
		
		execute_wilcoxon(CART_RFE_CV, CART_base, "*** CART RFE(CV) vs CART baseline ***")
		execute_wilcoxon(CART_RV_3+CART_RV_5, CART_LP_3+CART_LP_5, "*** CART RV(3) + RV(5) vs CART LP(3) + LP(5) ***")
		execute_wilcoxon(CART_RFE_3+CART_RFE_5, CART_LP_3+CART_LP_5, "*** CART RFE(3) + RFE(5) vs CART LP(3) + LP(5) ***")
		execute_wilcoxon(CART_RFE_3+CART_RFE_5, CART_RV_3+CART_RV_5, "*** CART RFE(3) + RFE(5) vs CART RV(3) + RV(5) ***")

########################################### *** EXECUCAO TESTE *** ###########################################
	
def execute_CART_test(X_train, X_test, Y_train, Y_test, CART_selected_feats_final, CART_best_params_final, filename, cart_type):
	
	X_train_normal, Xnormal = norm(X_train, X_test) # normalizando

	print(cart_type)

	X_train_CART = X_train_normal.iloc[:,CART_selected_feats_final]
	XCART = Xnormal.iloc[:,CART_selected_feats_final]

	clf = DecisionTreeClassifier(**CART_best_params_final)
	clf.fit(X_train_CART, Y_train)

	Y_true, Y_pred = Y_test, clf.predict(XCART)


	print("Selected Features ({})".format(X_train_CART.shape[1]))
	print(X_train_CART.columns.tolist())
	print(" ")
	
	print("CART_best_params_final")
	print(CART_best_params_final)
	print(" ")
	
	print('Results on the test set:')
	print(classification_report(Y_true, Y_pred))
	print(" ")
	
	# Calculating metrics
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		CART_accuracy = metrics.accuracy_score(Y_test, Y_pred)
		CART_fscore = metrics.f1_score(Y_test, Y_pred)
		CART_precision = metrics.precision_score(Y_test, Y_pred)
		CART_recall = metrics.recall_score(Y_test, Y_pred)

	print('CART_accuracy: %0.10f' % CART_accuracy)
	print('CART_precision: %0.10f' % CART_precision)
	print('CART_recall: %0.10f' % CART_recall)
	print('CART_fscore: %0.10f' % CART_fscore)
	print("\n")
	
	dot_data = StringIO()
	export_graphviz(clf, out_file=dot_data,
		filled=True, rounded=True, special_characters=True,
		feature_names = list(XCART), class_names=['0','1'])
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	graph.write_png(filename)
	Image(graph.create_png())
	
	return CART_accuracy, CART_precision, CART_fscore, CART_recall
	
########################################### *** WILCOXON *** ###########################################
	
def execute_wilcoxon(X, Y, wilcoxon_type):
	
	print(wilcoxon_type)
	print(wilcoxon(X, Y))
	print(" ")
		
########################################### *** MAIN *** ###########################################
  
if __name__=="__main__":

	data = pd.read_csv("data.csv") # Leitura
	
	save_data(data)
	
	X, Y = pre_proc(data) # Pre processamento
	
	CART_base = execute_baseline(X, Y)

	execute_training_GS(X, Y, CART_base)
