#!/usr/bin/env python
# coding: utf-8

#import libraries
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns;
import scipy;
import sklearn as sk;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import KFold;
#read in file containing the independent variables
ind_vars = pd.read_stata('independent_vars.dta')
#read in file containing bankruptcy data
data_def = pd.read_stata('bankruptcy_data.dta')

ind_vars.describe()
#only numeric columns and character column
data_def.describe()


#drop null values
#data_def.dropna()
#sort on columns, returns a dataframe, do assign to another dataframe
data_def=data_def.sort_values(['gvkey', 'pyear'])
#set index as a combination of variables
data_def.set_index(['gvkey', 'pyear'])
data_def_ind_assets = data_def[data_def.ind_assets==1]
data_def_ind_assets.head()
data_def_ind_assets=data_def_ind_assets.sort_values(['gvkey','pyear'])
data_def_ind_assets.set_index(['gvkey','pyear'])
data_def_ind_assets.head()

X=data_def_ind_assets[['nita','tlta','rsize','excess_ret','sigma']]
#X.index
Y=data_def_ind_assets[['ind_lopucki']]
print(X.head(), Y.head())

#linear regression
model=LinearRegression()
model.fit(X,Y)
model.coef_
model.intercept_
#R-squared
# 1-ss_residual/ss_total
model.score(X,Y)
#adjusted R-squared
#(1 - (1-R2))*(n-1)/(n-p-1)
1 -(1-model.score(X,Y))*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1)
#predict values
predicted = pd.DataFrame(model.predict(X))
#rename column
predicted.columns=['predicted_value']
predicted.describe()

from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import train_test_split

#logistic regression
#random state fior random number initialiser for k-fold validation
log_model = LogisticRegression(random_state=12)
#number of folds
kf=KFold(n_splits=10)
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25,random_state=12)
log_model.fit(X_train, Y_train)
#accuracies = cross_val_score(log_model,X_train, Y_train)
log_model.coef_
log_model.intercept_
predictions = log_model.predict(X_test)
accuracy_corr_over_data=log_model.score(X_test,Y_test)
print(accuracy_corr_over_data)

#confusion matrix - contingency table
from sklearn import metrics
import numpy as np
cnf_matrix = metrics.confusion_matrix(Y_test,predictions)
cnf_matrix

#classification rate: #correct predictions/#observations
print("Accuracy:",metrics.accuracy_score(Y,predictions))
#precision: 
print("Precision:",metrics.precision_score(Y,predictions))
#correctly categorize default firms/TPR/sensitivity
print("Recall:",metrics.recall_score(Y,predictions))


#plot this contigency table
class_names=['non-default','default']
fig,ax=plt.subplots()
tick_marks=np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks,class_names)
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#ROC curve - plots tpr against fpr
#TPR= #correctly predicted defaults/#obs
#FPR = #incorrectly predicted defaults/#obs
#FPR=1-specificity = 1 - #correctly predicted non-defaults/#obs

y_pred_prob = log_model.predict_proba(X_test)[::,1]
fpr, tpr, _ =metrics.roc_curve(Y_test,y_pred_prob)
auc = metrics.roc_auc_score(Y_test, y_pred_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()

#accuracy ratio or gini coefficient
ar = 2*auc-1
ar

#decision tree
from sklearn.tree import DecisionTreeClassifier
dec_tree_clf =  DecisionTreeClassifier(random_state=0)
dec_tree_clf.fit(X,Y)
dec_tree_pred=dec_tree_clf.predict(X)
dec_tree_prob = dec_tree_clf.predict_proba(X)
cnf_matrix_dt = metrics.confusion_matrix(Y, dec_tree_pred)
cnf_matrix_dt


dec_tree_prob = dec_tree_clf.predict_proba(X)[::,1]
fpr, tpr, _ =metrics.roc_curve(Y,dec_tree_prob)
auc = metrics.roc_auc_score(Y, dec_tree_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()


#random forests
from sklearn.ensemble import RandomForestClassifier
dtree_clf = RandomForestClassifier(n_estimators=10, random_state=0)
dtree_clf.fit(X_train,Y_train)
#predict labels - 0,1
predictions_dtree=dtree_clf.predict(X_test)
#predict probabilities
pred_prob_dtree=dtree_clf.predict_proba(X_test)
#confusion matrix
cnf_matrix_dtree1 = metrics.confusion_matrix(Y_test,predictions_dtree)
print(cnf_matrix_dtree1)
y_pred_prob_dtree = dtree_clf.predict_proba(X_test)[::,1]
fpr, tpr, _ =metrics.roc_curve(Y_test,y_pred_prob_dtree)
auc = metrics.roc_auc_score(Y_test, y_pred_prob_dtree)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()
metrics.accuracy_score(Y_test,predictions_dtree)
metrics.precision_score(Y_test,predictions_dtree)
metrics.recall_score(Y_test,predictions_dtree)

#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_bayes = gnb.fit(X,Y).predict(X)
gnb_proba = gnb.predict_proba(X)
cnf_matrix_nb = metrics.confusion_matrix(Y, y_pred_bayes)
cnf_matrix_nb

gnb_prob = gnb.predict_proba(X)[::,1]
fpr, tpr, _ =metrics.roc_curve(Y,gnb_prob)
auc = metrics.roc_auc_score(Y, gnb_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()

metrics.accuracy_score(Y,y_pred_bayes)
metrics.precision_score(Y,y_pred_bayes)
metrics.recall_score(Y,y_pred_bayes)


#neural network
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=0)
clf_nn.fit(X,Y)
nn_predict=clf_nn.predict(X)
nn_predict_prob=clf_nn.predict_proba(X)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y,nn_predict_prob)
auc=metrics.roc_auc_score(Y, nn_predict_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()





