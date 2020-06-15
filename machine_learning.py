#Tennis matches classification

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import binom_test
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import graphviz
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# Beolvasás
data = pd.read_csv('Stats.csv')

#Adatok első 5 sorának kiírása
data.head()

# Cimkék kinyerése
y = data['winner']
y = 1 * y

# Jellemzők kinyerése
df=pd.read_csv('Stats.csv', sep=',',header=None)
a = np.delete(df.values, 0, axis=1) # match_id törlése
a = np.delete(a, 0, axis=1) # player_id törlése
a = np.delete(a, 2, axis=1) # winner oszlop törlése - mivel ezt már kinyertük
x = np.delete(a, 0, axis=0) # első sor(fejléc) törlése
x = x.astype(np.float)  # float típusra konvertálás
x = np.nan_to_num(x) # a nan értékek 0-ra állítása

# Dimenziók
# 20240 adat
# 21 jellemző
print(x.shape)
print(y.shape)

# Train és test-re bontás
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=8, shuffle=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15, shuffle=True)

#Baseline - Nearest centroid
tic = timeit.default_timer()
nc=NearestCentroid()
nc.fit(x_train,y_train)
print('Train-acc:',nc.score(x_train,y_train))
print('Test-acc:',nc.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
NC_time = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

#Gaussian Naive Bayes - Gaus eloszlást feltételezünk
tic = timeit.default_timer()
model = GaussianNB()
model.fit(x_train,y_train)
toc = timeit.default_timer()
print('Train-acc:',model.score(x_train,y_train))
print('Test-acc:',model.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
GNB_time = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

#Konfidencia intervallum - ellenőrizni
predicted_test = model.predict(x_test)
test_acc = accuracy_score(y_test,predicted_test)
n_success = np.sum(y_test==predicted_test)

p = 0.91

interval = binom_test(n_success,y_test.shape[0],p=p)
print("Test-acc: ",test_acc,"+/-",interval,"(",p*100,"%)")

#!nproc # rendelkezésre álló processzorok kiíratása

# K-nearest neighbors with one processor
k=5
#knn = KNeighborsClassifier(n_neighbors=k)
tic = timeit.default_timer()
knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2,algorithm='ball_tree',leaf_size=30,n_jobs=1) # simple
knn.fit(x_train,y_train)
print('train-acc: ',knn.score(x_train,y_train))
print('test-acc: ',knn.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
KNN_time_simple = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

# konfúziós mátrix
predicted = knn.predict(x_test)
print(confusion_matrix(y_test, predicted))

# K-nearest neighbors with all processors
k=5
#knn = KNeighborsClassifier(n_neighbors=k)
tic = timeit.default_timer()
knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2,algorithm='auto',leaf_size=50,n_jobs=-1) # improvement
knn.fit(x_train,y_train)
print('train-acc: ',knn.score(x_train,y_train))
print('test-acc: ',knn.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
KNN_time_with_improvement = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

# konfúziós mátrix
predicted = knn.predict(x_test)
print(confusion_matrix(y_test, predicted))

# Grid Search for KNN - sokáig tart
tic = timeit.default_timer()
tuned_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12]}]
base_model = KNeighborsClassifier(weights='distance')
clf = GridSearchCV(base_model, tuned_parameters, cv=5, scoring="accuracy",return_train_score=True,refit=True)
clf.fit(x_train,y_train)
print(clf.best_params_)
print('test accuracy: ', clf.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
Grid_KNN_time = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

#Decision tree
tic = timeit.default_timer()
clf = tree.DecisionTreeClassifier(max_features='auto',max_depth=2) # ha kicsi a max depth nem kapunk jó eredményt
clf.fit(x_train,y_train)
print("train acc:",clf.score(x_train,y_train))
print("test acc:",clf.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
DT_time = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

# Vizualization of Decision tree
temp = pd.read_csv('Stats.csv', sep=',',header=None)
valami = np.delete(temp.values, 0, axis=1)
valami = np.delete(valami, 0, axis=1)
valami = np.delete(valami, 2, axis=1)

#features and labels
features = []
labels = ['winner','looser']
a = valami[0:1,:]
for t in(a[0]):
  features.append(str(t))

dot_data = tree.export_graphviz(clf, feature_names=features,  class_names=labels,  filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

#Grid search for Decision tree
tic = timeit.default_timer()
params = [{'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'min_samples_leaf':[1,2,3,4,5], 'max_features': ['auto', 'sqrt', 'log2']}]
clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(),param_grid=params,scoring='accuracy',cv=10,return_train_score=True)
clf.fit(x_train,y_train)
print('Best-params:',clf.best_params_)
print('Best-score:',clf.best_score_)
print('Test score: ', clf.score(x_test,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
Grid_DT_time = elapsed_time
print('Elapsed time: ',elapsed_time,'seconds')

# Grid serach visualization
df = pd.DataFrame(clf.cv_results_)
f = plt.figure()
f.set_figheight(10)
f.set_figwidth(10)
line_styles = [':','-','-.','--']
for ms,ls in zip(df['param_min_samples_leaf'].unique(),line_styles):
  subd_df = df[df['param_min_samples_leaf']==ms]
  depth = subd_df['param_max_depth']
  train_means = subd_df['mean_train_score'].values
  val_means = subd_df['mean_test_score'].values
  plt.plot(depth,train_means,'b'+ls,label='train-acc '+str(ms))
  plt.plot(depth,val_means,'y'+ls,label='val-acc '+str(ms))
plt.plot(clf.best_params_['max_depth'],clf.best_score_,'ro',label='best')
plt.legend()

#Decision tree - improving with polinomyal convertation
tic = timeit.default_timer()
poly = PolynomialFeatures(degree=3,interaction_only=True,include_bias=False) # 6 már sok, nincs elég ram hozzá
x_train_transformed = poly.fit_transform(x_train)
x_test_transformed = poly.transform(x_test)

clf = tree.DecisionTreeClassifier(max_features='auto',max_depth=10, min_samples_leaf=2)
clf.fit(x_train_transformed,y_train)
print("train acc:",clf.score(x_train_transformed,y_train))
print("test acc:",clf.score(x_test_transformed,y_test))
toc = timeit.default_timer()
elapsed_time = toc-tic
print('Elapsed time: ',elapsed_time,'seconds')

# az input convertálásnak köszönhetően jobb eredményt kaptunk

#Konfidencia intervallum a döntési fára, ami az eddi modellek közül a legjobb ereményt nyújtotta
predicted_test = clf.predict(x_test_transformed)
test_acc = accuracy_score(y_test,predicted_test)
n_success = np.sum(y_test==predicted_test)

p = 0.98

interval = binom_test(n_success,y_test.shape[0],p=p)
print("Test-acc: ",test_acc,"+/-",interval,"(",p*100,"%)")

# Futási idők összehasonlítása
print('Nearest centroid: ', NC_time)
print('Gaussian Naive Bayes: ', GNB_time)
print('K-nearest neighbors (simple): ', KNN_time_simple)
print('K-nearest neighbors (with improvement): ', KNN_time_with_improvement)
print('Decision tree: ', DT_time)
print('Grid search for KNN: ', Grid_KNN_time)
print('Grid serach for Decision tree: ', Grid_DT_time)

plt.plot(NC_time, 'o', label='Nearest centroid')
plt.plot(GNB_time, 'o', label= 'Gaussian Naive Bayes')
plt.plot(DT_time, 'o', label= 'Decision tree')
plt.legend()

plt.plot(KNN_time_simple, 'o', label= 'K-nearest neighbors (simple)' )
plt.plot(KNN_time_with_improvement, 'o', label= 'K-nearest neighbors (with improvement)' )
plt.plot(Grid_KNN_time, 'o', label= 'Grid search for KNN')
plt.plot(Grid_DT_time, 'o', label='Grid serach for Decision tree')
plt.legend()

tic = timeit.default_timer()
mms = MinMaxScaler() # feature vektort normalizáljuk
x_train_neural = mms.fit_transform(x_train)
x_test_neural = mms.fit_transform(x_test)

model = Sequential([
    Dense(32, input_shape=(x_train_neural.shape[1],)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dropout(0.25), # Regularizáció
    Dense(2),
    Activation('softmax'),
])

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

y_train_one_hot = keras.utils.to_categorical(y_train)
y_test_one_hot = keras.utils.to_categorical(y_test)

model.fit(x_train_neural, y_train_one_hot, epochs=5, batch_size=32)
toc = timeit.default_timer()

loss_and_metrics = model.evaluate(x_test_neural, y_test_one_hot, batch_size=128)
print('loss: ',loss_and_metrics[0])
print('accuracy: ',loss_and_metrics[1])
print()

probs = model.predict(x_test_neural, batch_size=128)
print(probs[0])

classes = np.argmax(probs,axis=1)
#print('Predicted class: ',classes[0],'and probability: ',probs[0,classes[0]])

print(confusion_matrix(y_test,classes))
print('time:', toc-tic)

tic = timeit.default_timer()

mms = MinMaxScaler()
x_train_neural = mms.fit_transform(x_train)
x_test_neural = mms.fit_transform(x_test)

# a tanító halmaz 10%-ának felhasználása az korai leálláshoz
x_train_neural,x_val,y_train_neural,y_val = train_test_split(x_train_neural,y_train,test_size=0.1)
y_train_one_hot = keras.utils.to_categorical(y_train_neural)
y_val_one_hot = keras.utils.to_categorical(y_val)

model = Sequential([
    Dense(32, input_shape=(x_train_neural.shape[1],)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(2),
    Activation('softmax'),
])

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

batch_size = 10
callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=1*batch_size,restore_best_weights=True,verbose=1)]
model.fit(x_train_neural,y_train_one_hot,callbacks=callbacks,validation_data=(x_val,y_val_one_hot),epochs=100,batch_size=batch_size)
toc = timeit.default_timer()

probs = model.predict(x_test_neural, batch_size=128)
print(probs[0])

classes = np.argmax(probs,axis=1)
#print('Predicted class: ',classes[0],'and probability: ',probs[0,classes[0]])

print('time:', toc-tic)

print(confusion_matrix(y_test,classes))

def avg_prediction(clfs, x):
  probs = clfs[0].predict_proba(x)
  for i in range(1, len(clfs)):
    probs = probs + clfs[i].predict_proba(x)
  probs = probs / len(clfs)
  pred = np.argmax(probs, axis=1)
  return pred

def max_prediction(clfs, x):
  probs = clfs[0].predict_proba(x)
  for i in range(1, len(clfs)):
    probs = np.maximum(probs, clfs[i].predict_proba(x))
  pred = np.argmax(probs, axis=1)
  return pred

tic = timeit.default_timer()
mms = MinMaxScaler()
x_train_ensemble = mms.fit_transform(x_train)
x_test_ensemble = mms.fit_transform(x_test)

tuned_params = {'random_state':[1,2,3,5,9]}

classifiers = []
for k in tuned_params['random_state']:
  svm = MLPClassifier(random_state=k, max_iter=400)
  svm.fit(x_train_ensemble, y_train)
  classifiers.append(svm)

toc = timeit.default_timer()

p_test = avg_prediction(classifiers, x_test_ensemble)
print(accuracy_score(y_test, p_test))
print(confusion_matrix(y_test, p_test))

p_test = max_prediction(classifiers, x_test_ensemble)
print(accuracy_score(y_test, p_test))
print(confusion_matrix(y_test, p_test))

print(toc - tic)

tic = timeit.default_timer()
mms = MinMaxScaler()
x_train_ensemble = mms.fit_transform(x_train)
x_test_ensemble = mms.fit_transform(x_test)

tuned_params = {
    'n_estimators':[1, 3, 5, 8, 10, 13],
    'max_samples':[1.0],
    'max_features':[1.0],
    'base_estimator__alpha':[0],
    'base_estimator__hidden_layer_sizes':[(200,)]
}
mlp = MLPClassifier(max_iter=10000, random_state=3)
b_mlp = BaggingClassifier(mlp, random_state=3)
gs_bagging = GridSearchCV(b_mlp, tuned_params, cv=4, return_train_score=True)
gs_bagging.fit(x_train_ensemble, y_train)
toc = timeit.default_timer()
print(gs_bagging.score(x_test_ensemble, y_test))
print(gs_bagging.best_params_)

#print(confusion_matrix(y_test, p_test))
print(toc - tic)


