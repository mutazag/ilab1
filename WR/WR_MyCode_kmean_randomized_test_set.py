#Still a bit disorganized
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import binarize
plt.rcParams.update({'font.size': 20})

#import xgboost as xgb
color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

n_descriptors = 38

#############################################################################################
#############################################################################################
#############################################################################################
    
############################### Machine Learning ###############################


##########################################################################################
#Learning with the k-means clusterered datasets
##########################################################################################
count=1
#count_pred=1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
train_df = pd.read_csv("LASSO_BR2/kmeans_randomized_trainingset_0_unindexed.csv",header=None)
test_df = pd.read_csv("LASSO_BR2/kmeans_randomized_testset_0_unindexed.csv",header=None)

save_values = open('LASSO_BR2/R2_MSE.csv', 'w')


numColumns = train_df.shape[1]
X_train = train_df.iloc[:,0:numColumns-1]
y_train = train_df.iloc[:,numColumns-1]
X_test = test_df.iloc[:,0:numColumns-1]
y_test = test_df.iloc[:,numColumns-1]

#y_train=np.reshape(y_train,(X_train.shape[0],1))
#y_test=np.reshape(y_test,(y_test.shape[0],1))

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)

X_DataSet_scaled = df.merge(X_train_scaled, X_test_scaled)

#scaler = Normalize().fit(X_train)
#X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
#X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)


#################Random Forest ################################################################
#################Random Forest ################################################################
#################Random Forest ################################################################
#################Random Forest ################################################################

from sklearn.ensemble import RandomForestRegressor


#Uncomment this code for 5-fold cross validation

"""
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 150, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 500, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(criterion='mse',random_state=0)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                               cv = 3, verbose=2, random_state=0, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_scaled, y_train)

rf_random.best_params_

regr= rf_random.best_estimator_
"""

# I have tested and got these parameters
regr = RandomForestRegressor(n_estimators= 150, min_samples_split= 0.011,
			 max_depth= 500, random_state=0, criterion='mse',bootstrap=True)
regr.fit(X_train_scaled, y_train)
print(regr.feature_importances_)
#print(regr.predict(x_reduced))
y_predicted=regr.predict(X_test_scaled)

y_predictedRF = y_predicted


# MARE=0
# for i in range(len(y_predicted)):
# #    print(y_test[i],y_predicted[i],abs((y_predicted[i]-y_test[i])/y_test[i]*100))
#     MARE+=abs((y_predicted[i]-y_test[i])/y_test[i]*100)
# print("MARE-RF-TEST=",MARE/len(y_predicted))


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R2:',r2_score(y_test, y_predicted))
print('MSE:',mean_squared_error(y_test, y_predicted))


xPlot=y_test
yPlot=y_predicted


plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(-0,-1)
plt.ylim(-0,-1)
plt.ylabel('RF-TEST ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
plt.savefig('LASSO_BR2/Figs/RF_kmean_randomized_DFT'+str(count), bbox_inches='tight')


y_predicted=regr.predict(X_train_scaled)
MARE=0
MSE=0
for i in range(len(y_train)):
#    print(y_train[i],y_predicted[i],abs((y_predicted[i]-y_train[i])/y_train[i]*100))
    MARE+=abs((y_predicted[i]-y_train[i])/y_train[i]*100)
    MSE+=(y_predicted[i]-y_train[i])**2
print("MARE-RF-TRAIN=",MARE/len(y_predicted))
print("MSE-RF-TRAIN=",(MSE)/len(y_predicted))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R2:',r2_score(y_train, y_predicted))
print('MSE:',mean_squared_error(y_train, y_predicted))
print(MARE/len(y_predicted))


xPlot=y_train
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(-0,-1)
plt.ylim(-0,-1)
plt.ylabel('RF-TRAIN ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
#plt.suptitle(('R^2=', r2_score(y_test, y_predicted)),
                # fontsize=14, fontweight='bold')
plt.savefig('LASSO_BR2/Figs/RF_kmean_train_DFT'+str(count), bbox_inches='tight')




import pickle
s = pickle.dumps(regr)
from sklearn.externals import joblib
joblib.dump(regr, 'LASSO_BR2/RandomForest_kmean_randomized_'+str(count)+'.pkl')  # comment for testing and predicting
#Later..
#regr = joblib.('LASSO_BR2/RandomForest_kmean_randomized_'+str(count)+load'.pkl') #comment for training


#2- Support Vector Machine ###############################################

#===============SEARCH BEST C===============
#from sklearn import svm, grid_search
#def svc_param_selection(X, y, nfolds):
#    C = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#    C=grid_search.best_params_
##print(C)
#=====================================

#from sklearn import svm, grid_search
def svc_model_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVR(epsilon=0.00001, verbose=True), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print (grid_search.best_params_)
    return grid_search.best_estimator_

regr = svc_model_selection(X_train_scaled,y_train,nfolds=5)

#regr = svm.SVR(epsilon=0.01, verbose=True, C=0.1)#, kernel='linear')
#regr = svm.SVR(epsilon=0.01, verbose=True, C=0.1)
a = regr.fit(X_train_scaled, y_train)
y_predicted=regr.predict(X_test_scaled)

y_predictedSVM = y_predicted


MARE=0
for i in range(len(y_test)):
#    print(y_test[i],y_predicted[i],abs((y_predicted[i]-y_test[i])/y_test[i]*100))
    MARE+=abs((y_predicted[i]-y_test[i])/y_test[i]*100)
print("MARE-SVM-TEST=",MARE/len(y_predicted))


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('r2',r2_score(y_test, y_predicted))
print('MSE',mean_squared_error(y_test, y_predicted))
print(MARE/len(y_predicted))

xPlot=y_test
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(-0,-1)
plt.ylim(-0,-1)
plt.ylabel('SVM-TEST ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
#plt.suptitle(('R^2=', r2_score(y_test, y_predicted)),
                # fontsize=14, fontweight='bold')
plt.savefig('LASSO_BR2/Figs/SVM_kmean_randomized_DFT'+str(count), bbox_inches='tight')


y_predicted=regr.predict(X_train_scaled)
MARE=0
MSE=0
for i in range(len(y_train)):
#    print(y_train[i],y_predicted[i],abs((y_predicted[i]-y_train[i])/y_train[i]*100))
    MARE+=abs((y_predicted[i]-y_train[i])/y_train[i]*100)
    MSE+=(y_predicted[i]-y_train[i])**2
print("MARE-SVM-TRAIN=",MARE/len(y_predicted))
print("MSE-SVM-TRAIN=",(MSE)/len(y_predicted))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('r2',r2_score(y_train, y_predicted))
print('MSE',mean_squared_error(y_train, y_predicted))
print(MARE/len(y_predicted))


xPlot=y_train
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(-0,-1)
plt.ylim(-0,-1)
plt.ylabel('SVM-TRAIN ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
#plt.suptitle(('R^2=', r2_score(y_test, y_predicted)),
                # fontsize=14, fontweight='bold')
plt.savefig('LASSO_BR2/Figs/SVM_kmean_train_DFT'+str(count), bbox_inches='tight')



import pickle
s = pickle.dumps(regr)
from sklearn.externals import joblib
joblib.dump(regr, 'LASSO_BR2/SVM_kmean_randomized_'+str(count)+'.pkl') # comment for predictions 
#Later..
#regr = joblib.load('LASSO_BR2/SVM_kmean_randomized_'+str(count)+'.pkl') # load it for predictions


#4- Relevance Vector Machine ###############################################

"""
from skrvm import RVR
regr = RVR(kernel='rbf',n_iter=100000, tol=0.0001, verbose=True)
a = regr.fit(X_train_scaled, y_train)

#clf = joblib.load('RVM_kmean_randomized_'+str(count)+'.pkl') 

y_predicted=regr.predict(X_test_scaled)



MARE=0
for i in range(len(y_test)):
    print(y_test[i],y_predicted[i],abs((y_predicted[i]-y_test[i])/y_test[i]*100))
    MARE+=abs((y_predicted[i]-y_test[i])/y_test[i]*100)
print("MARE=",MARE/len(y_predicted))


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print(r2_score(y_test, y_predicted))
print(mean_squared_error(y_test, y_predicted))
print(MARE/len(y_predicted))

xPlot=y_test
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(2.6,4.4)
plt.ylim(2.6,4.4)
plt.ylabel('RVM ($\AA$)')
plt.xlabel('DFT ($\AA$)')
plt.savefig('LASSO_BR2/Figs/RVM_rbf_kmean_randomized_DFT'+str(count), bbox_inches='tight')


y_predicted=regr.predict(X_train_scaled)
MARE=0
MSE=0
for i in range(len(y_train)):
    print(y_train[i],y_predicted[i],abs((y_predicted[i]-y_train[i])/y_train[i]*100))
    MARE+=abs((y_predicted[i]-y_train[i])/y_train[i]*100)
    MSE+=(y_predicted[i]-y_train[i])**2
print("MARE=",MARE/len(y_predicted))
print("MSE=",(MSE)/len(y_predicted))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print(r2_score(y_train, y_predicted))
print(mean_squared_error(y_train, y_predicted))
print(MARE/len(y_predicted))

import pickle
s = pickle.dumps(regr)
from sklearn.externals import joblib
joblib.dump(regr, 'LASSO_BR2/RVM_kmean_randomized_'+str(count)+'.pkl')
#Later..
regr = joblib.load('LASSO_BR2/RVM_kmean_randomized_'+str(count)+'.pkl') 

"""
##################Keras########################################################################
##################Keras########################################################################
##################Keras########################################################################
##################Keras########################################################################

from sklearn.metrics import roc_auc_score
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'




#sgd = Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False)
sgd = SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False)
#sgd = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#sgd = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#sgd = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

##define base model
def base_model():
    #start
    # For a multi-class classification problem

# For a mean squared error regression problem
    import keras.backend as K

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)
    
    model = Sequential()
    
    #end
    model.add(Dense(1024, activation='relu', input_shape=(n_descriptors,))) ###### CHANGE BASED ON INPUT SPACE
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
#    model.add(Dense(64, activation='relu'))
#    model.add(Dense(32, activation='relu'))
#    model.add(Dense(16, activation='relu'))
#    model.add(Dense(30, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    #model.compile(loss='mean_squared_error',  optimizer=sgd)
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy', mean_pred])

    model.compile(optimizer='rmsprop',
              loss='mse')

    return model

seed=1

class EarlyStoppingByLossVal(Callback):
    #def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
    def __init__(self, monitor='val_loss', value=0.000001, verbose=0):    
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        a = 10
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch) 
            self.model.stop_training = True

callbacks = [
    #EarlyStoppingByLossVal(monitor='val_loss', value=0.035, verbose=1)  #value is the loss function error
    EarlyStoppingByLossVal(monitor='val_loss', value=0.000050, verbose=1)
    #EarlyStoppingByLossVal(monitor='val_loss', value=0.000001, verbose=1)
]
#hist = model.fit(X_train, y_train, batch_size=1, epochs=200, validation_data=(X_test, y_test), verbose=2)
estimator = KerasRegressor(build_fn=base_model, nb_epoch=1000, batch_size=16)
# /**/

estimator.fit(X_train_scaled, y_train, batch_size=32, shuffle=True, epochs = 200 , 
              callbacks=callbacks,validation_data=(X_test_scaled, y_test),  verbose=2)

#estimator.fit(X_train, y_train)
#results = cross_val_score(estimator, x_reduced, y_reduced, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, estimator.predict(X_test))
#score1 = mean_squared_error(y_test, estimator.predict(X_test_scaled))

y_predicted=estimator.predict(X_test_scaled)


y_predictedNN = y_predicted

MARE=0
MSE=0
for i in range(len(y_test)):
 #   print(y_test[i],y_predicted[i],abs((y_predicted[i]-y_test[i])/y_test[i]*100))
    MARE+=abs((y_predicted[i]-y_test[i])/y_test[i]*100)
    MSE+=(y_predicted[i]-y_test[i])**2
print("MARE-NN-TEST=",MARE/len(y_predicted))
print("MSE-NN-TEST=",(MSE)/len(y_predicted))
#print(score)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R2:',r2_score(y_test, y_predicted))
print('MSE:',mean_squared_error(y_test, y_predicted))
print(MARE/len(y_predicted))
#

xPlot=y_test
yPlot=y_predicted
#print("y_test")
#print(y_test)
#print("y_predicted")
#print(y_predicted)

#x2Plot =x_test
#y2Plot= y_test

plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(0,-1)
plt.ylim(0,-1)
plt.ylabel('NN-TEST ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
plt.savefig('LASSO_BR2/Figs/NN_kmean_randomized_DFT_'+str(count), bbox_inches='tight')


y_predicted=estimator.predict(X_train_scaled)
MARE=0
MSE=0
for i in range(len(y_train)):
    #print(y_train[i],y_predicted[i],abs((y_predicted[i]-y_train[i])/y_train[i]*100))
    MARE+=abs((y_predicted[i]-y_train[i])/y_train[i]*100)
    MSE+=(y_predicted[i]-y_train[i])**2
print("MARE-NN-TRAIN=",MARE/len(y_predicted))
print("MSE-NN-TRAIN=",(MSE)/len(y_predicted))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R2:',r2_score(y_train, y_predicted))
print('MSE:',mean_squared_error(y_train, y_predicted))
print(MARE/len(y_predicted))

#========================PLOT TRAIN SET========================================
xPlot=y_train
yPlot=y_predicted
#print("y_train")
#print(y_train)
#print("y_predicted")
#print(y_predicted)

#x2Plot =x_test
#y2Plot= y_test

plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(0,-1)
plt.ylim(0,-1)
plt.ylabel('NN-TRAIN ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
plt.savefig('LASSO_BR2/Figs/NN_kmean_train_DFT_'+str(count), bbox_inches='tight')

#==============================================================================

from keras.models import model_from_json
model_json = estimator.model.to_json()
with open("LASSO_BR2/k-mean_clustered_randomized_dataset_model_"+str(count)+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save("LASSO_BR2/k-mean_clustered_randomized_dataset_model_random_test_set_"+str(count)+".h5")
print("Saved model to disk")


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


########################### AVG RESULTS  #########################################################################


y_predictedAVG = (y_predictedRF + y_predictedSVM + y_predictedNN)/3


print('R2:',r2_score(y_test, y_predictedAVG))
print('MSE:',mean_squared_error(y_test, y_predictedAVG))

xPlot=y_test
yPlot=y_predictedAVG

plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.xlim(0,-1)
plt.ylim(0,-1)
plt.ylabel('AVG-TEST ($J/m^2$)')
plt.xlabel('DFT ($J/m^2$)')
plt.savefig('LASSO_BR2/Figs/AVG_DFT_'+str(count), bbox_inches='tight')




# later...
#from keras.models import load_model
#
#
#
#model = load_model("LASSO_BR2/SavedModels/k-mean_clustered_randomized_dataset_model_random_test_set_"+str(count)+".h5")
# 
#y_predicted=model.predict(X_test_scaled)
#
#MARE=0
#MSE=0
#for i in range(len(y_test)):
#    print(y_test[i],y_predicted[i],abs((y_predicted[i]-y_test[i])/y_test[i]*100))
#    MARE+=abs((y_predicted[i]-y_test[i])/y_test[i]*100)
#    MSE+=(y_predicted[i]-y_test[i])**2
#print("MARE-NN-TRAIN=",MARE/len(y_predicted))
#print("MSE-NN-TRAIN=",(MSE)/len(y_predicted))
#
#
#
#
#from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#print(r2_score(y_test, y_predicted))
#print(mean_squared_error(y_test, y_predicted))
#print(MARE/len(y_predicted))
#
#
#xPlot=y_test
#yPlot=y_predicted
#plt.figure(figsize=(10,10))
#plt.plot(xPlot,yPlot,'ro')
#plt.plot(xPlot,xPlot)
#plt.xlim(0,-1)
#plt.ylim(0,-1)
#plt.ylabel('NN ($J/m^2$)')
#plt.xlabel('DFT ($J/m^2$)')
#plt.savefig('LASSO_BR2/Figs/NN_kmean_randomized_DFT_'+str(count), bbox_inches='tight')
#
#
#y_predicted=model.predict(X_train_scaled)
#
#
#
#from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#print(r2_score(y_train, y_predicted))
#print(mean_squared_error(y_train, y_predicted))
#print(MARE/len(y_predicted))


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

#
#
#lasso_monolayer_data = pd.read_csv("LASSO/lasso_monolayer_data.csv",header=0)
#
#
#
##Query based on column value
#ar=lasso_monolayer_data.loc[lasso_monolayer_data.iloc[:,0]=='CdO']
#
##the average of the monolayer data
#ar.iloc[0,1:].mean()
#
#for i in range(310):                ###### NUMBER OF MONOLAYERS
#    ar=lasso_monolayer_data.iloc[i,1:]
# #   print(lasso_monolayer_data.iloc[i,0],ar.mean(),ar.std(),ar.var())  #ONLY VISUALIZZATION 
#
