import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, InstanceHardnessThreshold, TomekLinks, CondensedNearestNeighbour, AllKNN
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
sns.set()

# Method: df_column_names
# Inputs: df:Table
# Output: res:list
def df_column_names(df):
    return df.columns

# Method: eda_nullvalues
# Inputs: df:Table
# Output:
def eda_nullvalues(df):
    total = df.isnull().sum().sort_values(ascending = False)
    total = total[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = total * 100 / len(df)
    percent = percent[df.isnull().sum().sort_values(ascending = False) != 0]
    concat = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
    print (concat)
    print ( "-------------")

# Method: eda_stats
# Inputs: df:Table
# Output:
def eda_stats(df):
    print('-------------------------------------------------------------')
    print('')
    # Dataframe columns
    print('Dataframe Columns:')
    print(df.columns)
    print('')
    print('-------------------------------------------------------------')
    print('')
    #Concise summary of the dataframe
    print('Dataframe Summary:')
    print(df.info())
    print('')
    print('-------------------------------------------------------------')
    print('')
    #Returns the dimensions of the array rows & columns
    print('Dataframe Dimensions:')
    print('# of rows:',df.shape[0])
    print('# of columns:',df.shape[1])
    print('')
    print('-------------------------------------------------------------')
    print('')
    #Descriptive or summary statistics of numeric columns
    print('Dataframe Summary Statistics:')
    print(df.describe())
    print('')
    print('-------------------------------------------------------------')
    #Returns missing values
    print('Dataframe Missing Values:')
    print(df.isna().mean().round(4) * 100)
    print('')
    print('-------------------------------------------------------------')

# Method: eda_datavisualization
# Inputs: df:Table
# Output:
def eda_datavisualization(df, n=5):
    print('-------------------------------------------------------------')
    print('')
    #Return first rows of dataframe
    print('Dataframe Head:')
    print(df.head(n))
    print('')
    print('-------------------------------------------------------------')
    print('')
    #Return last rows of dataframe
    print('Dataframe Tail:')
    print(df.tail(n))
    print('')
    print('-------------------------------------------------------------')
    print('')
    #Return random sample of rows from dataframe
    print('Dataframe Sample:')
    print(df.sample(n))
    print('')
    print('-------------------------------------------------------------')

# Method: eda_datadistribution
# Inputs: df:Table
# Output:
def eda_datadistribution(df):
    print('-------------------------------------------------------------')
    print('')
    print('Representation of the distribution of numeric data:')
    df.boxplot(widths = 0.6, rot=45, figsize=[20,10])
    print('')
    df.hist(figsize=[15,10])
    print('')
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(10,10)) 
    ax = sns.heatmap(corrMatrix, square=True, annot=True, ax=ax)
    ax.set_title('Heatmap Correlation Matrix of numeric data')
    plt.show()
    print('-------------------------------------------------------------')

# Method: eda_labelinformation
# Inputs: df:Table, column_names:list
# Output:
def eda_labelinformation(df, column_names):
    for column_name in column_names:
        print('-------------------------------------------------------------')
        print('')
        print('Value counts of the Target Feature')
        print(df[column_name].value_counts())
        print('')    
        print('-------------------------------------------------------------')
        print('')
        print('Percentage of Imbalance Property: {:.2f}%'.format((df[column_name].value_counts()[1] * 100 / df[column_name].value_counts()[0])))
        print('')
        print('-------------------------------------------------------------')
        print('')
        ax = sns.countplot(x = df[column_name])
        ax.set_title('Number of Observations per Class', fontsize=18)
        ax.set_xlabel(column_name, fontsize=16)
        ax.set_ylabel('count', fontsize=16)
        plt.show()
        print('')
        print('-------------------------------------------------------------')

# Method: read_df
# Inputs: path:text
# Output: df:Table
def read_df(path):
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    return df

# Method: remove_null_values
# Inputs: df:Table
# Output: df:Table
def remove_null_values(df):
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].mode()[0])
    if 'smoking_status' in df.columns:
        df['smoking_status'] = df['smoking_status'].fillna('unknown')
    return df

# Method: encode_categorical_features
# Inputs: df:Table
# Output: df:Table
def encode_categorical_features(df):
    df = df.apply(lambda x: x.astype('category').cat.codes)
    return df

# Method: split_and_scale
# Inputs: df:Table
# Output: 
def split_and_scale(df, test_size=0.2, random_state=42):
    target_column = 'stroke'
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        shuffle=True, 
                                                        stratify=y,
                                                        random_state=random_state)
    rs = RobustScaler()
    X_train = rs.fit_transform(X_train)
    X_test = rs.transform(X_test)
    X_train.to_csv('X_train.csv', encoding='utf-8', index=False)
    X_test.to_csv('X_test.csv', encoding='utf-8', index=False)
    y_train.to_csv('y_train.csv', encoding='utf-8', index=False)
    y_test.to_csv('y_test.csv', encoding='utf-8', index=False)   
    
    print("Done!")
    
# Method: evaluate_classificationmodel
# Inputs: X_train:Table, X_test:Table, y_train:Table, y_test:Table, model:Cla_Model
# Output: result:dict
def evaluate_classificationmodel(X_train, X_test, y_train, y_test, model):
    # model fitting
    model.fit(X_train, y_train)
    # prediction for the evaluation set
    predictions = model.predict(X_test)
    # accuracy
    accuracy = metrics.accuracy_score(y_test, predictions)   
    # precision, recall and f1 score
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
    # confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()    
    # fpr and tpr values for various thresholds 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label = 1) 
    # area under the curve
    auc_ = metrics.auc(fpr, tpr)    
    # gather results
    result = {'accuracy': accuracy,
              'precision':precision,
              'recall':recall,
              'fscore':fscore, 
              'n_occurences':support,
              'predictions_count': Counter(predictions),
              'tp':tp, 
              'tn':tn, 
              'fp':fp,
              'fn':fn, 
              'auc':auc_}
    
    return result

# Method: resampling_techniques_pipeline
# Inputs: X_train:Table, X_test:Table, y_train:Table, y_test:Table, model:Cla_Model
# Output: results:dict 
def resampling_techniques_pipeline(X_train, X_test, y_train, y_test, model):
    results = {'ordinary':{},
               'class_weight':{},
               'oversample':{},
               'undersample':{},
               'hybrid':{}}
    
    # ------- Ordinary ----------
    results['ordinary'] = evaluate_classificationmodel(X_train, X_test, y_train, y_test, model)
    
    # ------- Class weight -------
    if 'class_weight' in model.get_params().keys():
        model.set_params(class_weight='balanced')
        results['class_weight'] = evaluate_classificationmodel(X_train, X_test, y_train, y_test, model)
        
    # ------ OverSampling techniques -----
    print('   Oversampling methods:')
    techniques = [RandomOverSampler(), SMOTE(), SMOTENC(categorical_features=[1,2]), BorderlineSMOTE(), SVMSMOTE(), 
                  KMeansSMOTE(cluster_balance_threshold=0.01), ADASYN()]
    for sampler in techniques:
        technique = sampler.__class__.__name__
        print(f'Technique:{technique}')
        print(f'Before resampling: {sorted(Counter(y_train).items())}')
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f'After resampling: {sorted(Counter(y_resampled).items())}')        
        results['oversample'][technique] = evaluate_classificationmodel(X_resampled, X_test, y_resampled, y_test, model)
        
    # ------ UnderSampling techniques --------
    print('   Undersampling methods:')
    techniques = [RandomUnderSampler(), ClusterCentroids(), NearMiss(), InstanceHardnessThreshold(), 
                  TomekLinks(), CondensedNearestNeighbour(), AllKNN(),EditedNearestNeighbours(),
                  RepeatedEditedNearestNeighbours(), OneSidedSelection(), NeighbourhoodCleaningRule()]
    for sampler in techniques:
        technique = sampler.__class__.__name__
        print(f'Technique:{technique}')
        print(f'Before resampling: {sorted(Counter(y_train).items())}')
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f'After resampling: {sorted(Counter(y_resampled).items())}')        
        results['undersample'][technique] = evaluate_classificationmodel(X_resampled, X_test, y_resampled, y_test, model)
        
    # ------ Hybrid techniques --------
    print(   'Hybrid methods:')
    techniques = [SMOTEENN(), SMOTETomek()]
    for sampler in techniques:
        technique = sampler.__class__.__name__
        print(f'Technique:{technique}')
        print(f'Before resampling: {sorted(Counter(y_train).items())}')
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f'After resampling: {sorted(Counter(y_resampled).items())}')        
        results['hybrid'][technique] = evaluate_classificationmodel(X_resampled, X_test, y_resampled, y_test, model)
    
    return results

# Method: evaluate_resamplingmethod
# Inputs: results:dict, method:text
# Output:
def evaluate_resamplingmethod(results, method, metrics = ['precision', 'recall', 'fscore']):
    fig, ax = plt.subplots(1, 3, sharey = True, figsize=(20,6)) 
    
    for i, metric in enumerate(metrics):
        ax[i].axhline(results['ordinary'][metric][0], label = 'No Resampling')
        if results['class_weight']:
            ax[i].bar(0, results['class_weight'][metric][0], label = 'Adjust Class Weight')
        ax[0].legend()
        for j, (technique, result) in enumerate(results[method].items()):
            ax[i].bar(j+1, result[metric][0], label = technique)
        ax[i].set_title(f'Cerebral Stroke:\n{metric}')

# Method: metrics_dataframe
# Inputs: results:dict
# Output: results:Table
def metrics_dataframe(results):
    # ordinary results
    ordinary = results.get('ordinary', {})
    df_ordinary = pd.DataFrame.from_dict(ordinary, orient='index').T
    df_ordinary.insert(0, 'method','ordinary')
    
    #class weight results
    class_weight = results.get('class_weight',{})
    df_classweight = pd.DataFrame.from_dict(class_weight, orient='index').T
    df_classweight.insert(0, 'method','class_weight')
    
    # sampling techniques results
    sampling = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in results.items()}, axis=0)
    sampling = sampling.drop(0, axis=1)
    sampling = sampling.dropna(how='all')
    sampling = sampling.droplevel(0, axis=0)
    sampling.index.name = 'method'
    sampling = sampling.reset_index()
    
    #concat dataframes
    output = pd.concat([df_ordinary, df_classweight], ignore_index=True)
    results = pd.concat([output, sampling], ignore_index=True)
    
    return results

# Method: print_done
# Inputs: any_df:Table
# Output:
def print_done(any_df):
    print('done')

# Method: model_object_selector
# Inputs: choice:Num
# Output: model:Cla_Model
def model_object_selector(choice):
    if choice == 0:
        model = SVC()
    elif choice == 1:
        model = DecisionTreeClassifier()
    elif choice == 2:
        model = GaussianNB()
    elif choice == 3:
        model = KNeighborsClassifier()
    elif choice == 4:
        model = LogisticRegression()
    else:
        raise ValueError('Model not found')
    return model