#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np # linear algebra
import seaborn as sns
from ggplot import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#%% import dataset 
data = pd.read_csv("/Users/ayushsharma/repos/GIT/Breast-Cancer-Analysis/data.csv")

# Violin Plot

# ax = sns.violinplot(x = "diagnosis", y = "radius_mean" , hue = "concavity_mean" , data = data , split = True)

data.head()
# Decribing data - statistics features
data.describe()



## to figure out the integrity of the dataset
null_count = data.isnull().sum()
percentage = null_count / len(data) *100

plt.figure( figsize=(4,4) )
percentage.plot( kind='bar',label='the percentage of NULL values' )
plt.ylim(0.0,100.0)
plt.legend()
plt.show( )


# to check the distribution of the diagnose feature
label = data[ 'diagnosis' ]
plt.figure( figsize=[10,5] )
plt.subplot(121)
plt.pie( x= label.value_counts(),labels=label.unique(),colors=['b','r'],explode=[0.1,0.1],autopct='%.2f' )
plt.title( 'the distribution of labels' )
plt.subplot(122)
plt.bar( x = [ 0.2,1 ],height =label.value_counts() ,width=0.6,color=['lightskyblue','gold'] )
plt.xticks( range(2),label.unique() )
plt.title( 'the number of labels' )
plt.legend()
plt.show()


# As you can see there are  columns, like "id" and "Unnamed: 32". Let's drop them. Also We need to change categorical data to numeric data.
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]


# to visualize the distribution of different labels
B = data[ data['diagnosis']==1 ]
M = data[ data['diagnosis']==0 ]
def plot_distribution ( feature ):
    global B
    global M
    b = B[feature]
    m = M[feature]
    group_labels = ['benign','malignant']
    colors = ['#FFD700', '#7EC0EE']
    plt.figure( figsize=[4,4] )
    sns.distplot( b,color=colors[0],label=group_labels[0] )
    sns.distplot( m,color=colors[1],label=group_labels[1] )
    plt.title(feature)
    plt.legend(  )
    plt.show()

plot_distribution('radius_mean')
plot_distribution('texture_mean')
plot_distribution('perimeter_mean')
plot_distribution('area_mean')


#correlation map
sns.set(style="white")
fig,ax=plt.subplots(figsize=(16,16))
corr=data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,vmin=-1,vmax=1,fmt = ".1f",annot=True,cmap="coolwarm", mask=mask, square=True)

# Dropping variables due to high correaltion between variables.
data=data.drop(['area_mean', 'perimeter_mean', 'radius_worst', 'area_worst', 'perimeter_worst','texture_worst','concavity_mean','perimeter_se', 'area_se'],axis=1)



# Train and test split
x_train, x_test, y_train, y_test = train_test_split(x_data,y, test_size = 0.3, random_state = 1)


# Seperating dependant and indepedant variables
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)


# to normalize the dataset with standardscale
conv = StandardScaler()
std_data = conv.fit_transform( x_data )



# use PCA to reduce dimensionality
pca = PCA(n_components=20,svd_solver='full')
transformed_data = pca.fit_transform( std_data )
print( transformed_data.shape )
print( pca.explained_variance_ratio_*100 )
print( pca.explained_variance_ )

threshold = 0.80
for_test = 0
order = 0
for index,ratio in  enumerate (pca.explained_variance_ratio_):
    if threshold>for_test:
        for_test+= ratio
    else:
        order = index + 1
        break
        
        
print( 'the first %d features could represent 85 percents of the viarance' % order )
print( pca.explained_variance_ratio_[:order].sum() )
com_col = [ 'com'+str(i+1) for i in range(order) ]
com_col.append('others')
com_value = [ i for i in pca.explained_variance_ratio_[:order] ]
com_value.append( 1-pca.explained_variance_ratio_[:order].sum() )
com_colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgrey', 'orange', 'white']
plt.figure( figsize=[4,4] )
plt.pie( x=com_value,labels=com_col,colors=com_colors,autopct='%.2f' )
plt.title( 'the first 6 components' )
plt.show()


def plot_confusion_matrix (  label,pred,classes = [0,1] ,cmap = plt.cm.Blues,title='confusion matrix' ):
    con_m = confusion_matrix( label,pred )
    plt.imshow( con_m,interpolation = 'nearest',cmap=cmap )
    plt.title(title)
    plt.colorbar()
    thres = con_m.max() / 2
    for j in range( con_m.shape[0] ):
        for i in range( con_m.shape[1] ):
            plt.text( i,j,con_m[j,i],
                      horizontalalignment = 'center',
                      color='white' if con_m[i,j]>thres else 'black')

    plt.ylabel( 'true label' )
    plt.xlabel( 'predicted label' )
    plt.xticks(  classes,classes )
    plt.yticks(  classes,classes )
    plt.tight_layout()

def print_matrix(  label,pred ):
    tn, fp, fn, tp = confusion_matrix( label,pred ).ravel()
    print( 'Accuracy rate = %.2f' %(( tp+tn )/( tn+fp+fn+tp )) )
    print('Precision rate = %.2f' % ((tp ) / (fp + tp)))
    print('Recall rate = %.2f' % ((tp ) / (fn + tp)))
    print('F1 score = %.2f' % ( 2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn)))) ))
    
    
def plot_ROC( label,pred ):
    from sklearn.metrics import roc_curve
    fpr, tpr,t = roc_curve( label,pred )
    plt.plot(fpr, tpr, label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ')
    print( 'the threshold is ', t )
    plt.show()
    
def plot_learning_curve( estimator,title,x,y,train_sizes = np.linspace(.1, 1.0, 5),n_job = 1 ):
    plt.figure( figsize=[4,4] )
    plt.title(title)
    plt.xlabel( 'Training examples' )
    plt.ylabel( 'Score' )

    train_size,train_score,test_score = learning_curve(estimator,x,y,n_jobs=n_job,train_sizes=train_sizes)


    train_scores_mean = np.mean(train_score, axis = 1)
    train_scores_std = np.std(train_score, axis = 1)
    test_scores_mean = np.mean(test_score, axis = 1)
    test_scores_std = np.std(test_score, axis = 1)
    plt.grid()
    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_size, train_scores_mean, 'o-', color = "r",
             label = "Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color = "g",
             label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt


random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(transformed_data, y, test_size = 0.12, random_state = random_seed)
logistic_reg = LogisticRegression( random_state=random_seed )
para_grid = {
            'penalty':['l1','l2'],
            'C':[0.001,0.01,0.1,1.0,10,100,1000]
            }
CV_log_reg = GridSearchCV( estimator=logistic_reg,param_grid=para_grid,n_jobs=-1 )
CV_log_reg.fit( X_train,y_train )
best_para = CV_log_reg.best_params_
print( 'the best parameters are ',best_para )


# now using the best parameters to log the regression model
logistic_reg = LogisticRegression( C=best_para['C'],penalty=best_para['penalty'],random_state=random_seed )
logistic_reg.fit( X_train,y_train )
y_pred = logistic_reg.predict( X_test )

plot_confusion_matrix( y_test,y_pred )
plt.show( )
print_matrix(y_test,y_pred)
plot_ROC(y_test,y_pred)
plt.show( )
