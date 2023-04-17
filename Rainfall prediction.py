import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()
import io
data=pd.read_csv(io.BytesIO(uploaded['Rainfall- Dataset.csv']))

"""DATA EXPLORATION"""

data=data.fillna(data.mean())
data.info()

data.head()

data.describe()

plt.figure(figsize=(11,4))
sns.heatmap(data[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']].corr(),annot=True)
plt.show()

plt.figure(figsize=(11,4))
sns.heatmap(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
plt.show()

data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(10,6));

data[['YEAR','AUG','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8));

data[['YEAR','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']].groupby("YEAR").sum().plot(figsize=(13,8));

data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(13,8));

data[['SUBDIVISION','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(16,8));

data.hist(figsize=(20,20));

"""DATA NORMALIZATION"""

data.isnull().sum()

#Function to plot the graphs
def plot_graphs(groundtruth,prediction, title):
  N = 9
  ind = np.arange(N) #the x Locations for the groups
  width = 0.27
# the width of the bars
  fig=plt.figure()
  fig.suptitle(title, fontsize=12)
  ax=fig.add_subplot (111)
  rects1 = ax.bar (ind, groundtruth, width, color='prediction')
  rects2 = ax.bar (ind+width, prediction, width, color='ground truth')
  ax: set_ylabel("Amount of rainfall")
  ax.set_xticks(ind+width)
  ax.set_xticklabels(('APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')) 
  ax.legend( (rects1[0], rects2[0]), ('Ground truth", "Prediction'))
#   autolabel(rects1)
  for rect in rects1:
    h=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')
  for rect in rects2:
    h=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')
#autolabel(rects2)
  plt.show()

"""SEPARATION OF TRAINING AND TESTING DATA"""

# seperation of training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_absolute_error

division_data = np.asarray(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']])

X = None; y = None;
for i in range(division_data.shape[1]-3):
  if X is None:
    X = division_data[:, i:i+3]
    y = division_data[:, i+3]
  else:
    X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
    y = np.concatenate((y, division_data[:, i+3]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

#test 2005

temp=data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC' ]].loc[data['YEAR']==2005]

data_2005 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION']=='PUNJAB'])

X_year_2005 = None; y_year_2005 = None 
for i in range(data_2005.shape[1]-3):
  if X_year_2005 is None:
    X_year_2005 = data_2005[:, i:i+3]
    y_year_2005= data_2005[:, i+3]

  else:

    X_year_2005 = np.concatenate((X_year_2005, data_2005[:, i:i+3]), axis=0)
    y_year_2005 = np.concatenate((y_year_2005, data_2005[:, i+3]), axis=0)

#test 2010

temp=data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC' ]].loc[data['YEAR']==2010]

data_2010 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION']=='PUNJAB'])

X_year_2010 = None; y_year_2010 = None 
for i in range(data_2010.shape[1]-3):
  if X_year_2010 is None:
    X_year_2010 = data_2010[:, i:i+3]
    y_year_2010= data_2010[:, i+3]

  else:

    X_year_2010 = np.concatenate((X_year_2010, data_2010[:, i:i+3]), axis=0)
    y_year_2010 = np.concatenate((y_year_2010, data_2010[:, i+3]), axis=0)

#test 2015

temp=data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC' ]].loc[data['YEAR']==2015]

data_2015 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION']=='PUNJAB'])

X_year_2015 = None; y_year_2015 = None 
for i in range(data_2015.shape[1]-3):
  if X_year_2015 is None:
    X_year_2015 = data_2015[:, i:i+3]
    y_year_2015= data_2015[:, i+3]

  else:

    X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i+3]), axis=0)
    y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i+3]), axis=0)

from keras.metrics.metrics import mean_absolute_percentage_error
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Linear model

reg = linear_model.ElasticNet(alpha=0.5)
reg.fit(X_train, y_train) 
y_pred=reg.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

#2005
y_year_pred_2005=reg.predict(X_year_2005)
#2010
y_year_pred_2010 = reg.predict(X_year_2010)
#2015
y_year_pred_2015 = reg.predict(X_year_2015)

print("MEAN 2005")
print(np.mean(y_year_2005),np.mean(y_year_pred_2005))
print("Standard deviation 2005")
print(np.sqrt(np.var(y_year_2005)),np.sqrt(np.var(y_year_pred_2005)))
print("MEAN 2010")
print (np.mean(y_year_2010), np.mean(y_year_pred_2010))
print("Standard deviation 2010") 
print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))
print("MEAN 2015")
print (np.mean(y_year_2015),np.mean(y_year_pred_2015))
print("Standard deviation 2015") 
print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))
plot_graphs(y_year_2005,y_year_pred_2005,"Year-2005")
plot_graphs(y_year_2010,y_year_pred_2010,"Year-2010") 
plot_graphs(y_year_2015,y_year_pred_2015,"Year-2015")

#NN model
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Flatten

inputs = Input(shape=(3,1))
x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
x = Conv1D(128, 2, padding='same', activation='elu')(x)
x = Flatten()(x)
x = Dense(128, activation='elu')(x)
x = Dense(54, activation='elu')(x)
x = Dense(32, activation='elu')(x)
x = Dense(1, activation='linear')(x)
model = Model (inputs=[inputs], outputs=[x])
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
model.summary()

model.fit(x=np.expand_dims(X_train,axis=2),y=y_train,batch_size=64,epochs=10,verbose=1,validation_split=0.1,shuffle=True)
y_pred=model.predict(np.expand_dims(X_test,axis=2))
print(mean_absolute_error(y_test,y_pred))

# spliting training and testing data only for Punjab
punjab = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['SUBDIVISION'] == 'PUNJAB'])

X = None; y = None
for i in range (punjab.shape[1]-3):
  if X is None:
    X = punjab[:, i:i+3]
    y = punjab[:, i+3]
  else:
    X = np.concatenate((X, punjab[:, i:i+3]), axis=0)
    y = np.concatenate((y, punjab[:, i+3]), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

from sklearn import linear_model
# Linear model
reg = linear_model.ElasticNet(alpha=0.5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

#2005
y_year_pred_2005 = reg.predict(X_year_2005)
#2010
y_year_pred_2010 = reg.predict(X_year_2010)
#2015
y_year_pred_2015 = reg.predict(X_year_2015)
print("MEAN 2005")
print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
print("Standard deviation 2005")
print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))
print("MEAN 2010")
print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
print("Standard deviation 2010")
print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))
print("MEAN 2015")
print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
print("Standard deviation 2015")
print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))
plot_graphs (y_year_2005,y_year_pred_2005, "Year-2005")
plot_graphs (y_year_2010,y_year_pred_2010, "Year-2010")
plot_graphs (y_year_2015,y_year_pred_2015, "Year-2015")

# SVR model
from sklearn.svm import SVR
clf = SVR(kernel='rbf', gamma='auto', C=0.5, epsilon=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

#2005
y_year_pred_2005 = reg.predict(X_year_2005)
#2010
y_year_pred_2010 = reg.predict(X_year_2010)
#2015
y_year_pred_2015 = reg.predict(X_year_2015)
print("MEAN 2005")
print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
print("Standard deviation 2005")
print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))
print("MEAN 2010")
print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
print("Standard deviation 2010")
print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))
print("MEAN 2015")
print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
print("Standard deviation 2015")
print(np.sqrt(np.var(y_year_2015)), np. sqrt(np.var(y_year_pred_2015)))
plot_graphs (y_year_2005,y_year_pred_2005, "Year-2005")
plot_graphs (y_year_2010,y_year_pred_2010, "Year-2010")
plot_graphs (y_year_2015,y_year_pred_2015, "Year-2015")

model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
y_pred = model.predict(np.expand_dims(X_test, axis=2))
print(mean_absolute_error(y_test, y_pred))
