import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer,ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.metrics import f1_score,precision_score,recall_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB

import imblearn
from imblearn.under_sampling import RandomUnderSampler


# Evaluation metrics, stratified k-fold cross validation, and models

# Custom metrics
def custom_eval_metric(y_true, y_pred,weight=None,classtype=None):   
    f1_score_ = f1_score(y_true,y_pred,average=weight)
    precision_ = precision_score(y_true,y_pred,average=weight)
    recall_ = recall_score(y_true,y_pred,average=weight)
    
    if weight:
        return f1_score_,precision_,recall_
    return f1_score_[classtype],precision_[classtype],recall_[classtype]

# Word frequencies
def cross_val_scores_freq(X_train, y_train, num_folds, model, ngrams, nontext=False, undersampling=False):     
    strat_kf = StratifiedKFold(n_splits=num_folds, shuffle=True) 
    f1_neg = np.empty(num_folds)
    precision_neg = np.empty(num_folds)
    recall_neg = np.empty(num_folds)
    f1_pos = np.empty(num_folds)
    precision_pos = np.empty(num_folds)
    recall_pos = np.empty(num_folds)
    f1_w = np.empty(num_folds)
    precision_w = np.empty(num_folds)
    recall_w = np.empty(num_folds)
    
    index = 0
    model = model
    t = X_train.Removed
    
    for train, test in strat_kf.split(np.zeros(len(t)), t):        
        x_train_, x_test_ = X_train.reindex(index=train), X_train.reindex(index=test)
        y_train_, y_test_ = y_train[train], y_train[test]
        
        if undersampling:
            undersample = RandomUnderSampler(sampling_strategy='majority') 
            x_train_, y_train_ = undersample.fit_resample(x_train_, y_train_)         
        if nontext:
            column_trans = ColumnTransformer(
            [('child_rem', OneHotEncoder(dtype='int'),['child_rem_flag']),
             ('sec_rem', OneHotEncoder(dtype='int'),['sec_child_rem_flag']),
             ('third_rem', OneHotEncoder(dtype='int'),['third_child_rem_flag']),
             ('fourth_rem', OneHotEncoder(dtype='int'),['fourth_child_rem_flag']),
             ('fifth_rem', OneHotEncoder(dtype='int'),['fifth_child_rem_flag']),
             ('run_prop',StandardScaler(),['run_prop_prev_rem']), 
             ('run_tot',StandardScaler(),['run_prev_tot']), 
             ('run_rem',StandardScaler(),['run_prev_rem']),
             ('tfidf', TfidfVectorizer(ngram_range=(1,ngrams)), 'body_clean_no_stop')], remainder='drop')
            x_train_= column_trans.fit_transform(x_train_)
            x_test_ = column_trans.transform(x_test_)
        else:
            column_trans = ColumnTransformer(
            [('tfidf', TfidfVectorizer(ngram_range=(1,ngrams)), 'body_clean_no_stop')], remainder='drop')
            x_train_= column_trans.fit_transform(x_train_)
            x_test_ = column_trans.transform(x_test_)            
            
        model.fit(x_train_, y_train_)
        pred_test = model.predict(x_test_)
        f1_neg[index],precision_neg[index],recall_neg[index] = custom_eval_metric(y_test_,pred_test,classtype=0)
        f1_pos[index],precision_pos[index],recall_pos[index] = custom_eval_metric(y_test_,pred_test,classtype=1)
        f1_w[index],precision_w[index],recall_w[index] = custom_eval_metric(y_test_,pred_test,weight='weighted')
        index += 1
    
    d = dict()
    neg_vals = [np.mean(f1_neg),np.mean(precision_neg),np.mean(recall_neg)]
    neg_keys = ['f1_neg','precision_neg','recall_neg']
    pos_vals = [np.mean(f1_pos),np.mean(precision_pos),np.mean(recall_pos)]
    pos_keys = ['f1_pos','precision_pos','recall_pos']
    w_vals = [np.mean(f1_w),np.mean(precision_w),np.mean(recall_w)]
    w_keys = ['f1_w','precision_w','recall_w']
    
    for key,val in zip(neg_keys,neg_vals):
        d[key] = val
    for key,val in zip(pos_keys,pos_vals):
        d[key] = val  
    for key,val in zip(w_keys,w_vals):
        d[key] = val 
    return d,model



# Tune precision-recall tradeoff to maximize F1 scores 

# set threshold
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# Word frequencies
def model_thresh_freq(X_train, y_train, num_folds, model, ngrams,nontext=False,undersampling=False):
    thresholds = np.arange(0, 1, 0.001)       
    strat_kf = StratifiedKFold(n_splits=num_folds, shuffle=True) 
    f1_neg_ = np.empty((num_folds,thresholds.shape[0]))
    precision_neg_ = np.empty((num_folds,thresholds.shape[0]))
    recall_neg_ = np.empty((num_folds,thresholds.shape[0]))
    f1_pos_ = np.empty((num_folds,thresholds.shape[0]))
    precision_pos_ = np.empty((num_folds,thresholds.shape[0]))
    recall_pos_ = np.empty((num_folds,thresholds.shape[0]))
    f1_w_ = np.empty((num_folds,thresholds.shape[0]))
    precision_w_ = np.empty((num_folds,thresholds.shape[0]))
    recall_w_ = np.empty((num_folds,thresholds.shape[0]))
    
    index = 0
    model = model
    t = X_train.Removed
    
    for train, test in strat_kf.split(np.zeros(len(t)), t):        
        x_train_, x_test_ = X_train.reindex(index=train), X_train.reindex(index=test)
        y_train_, y_test_ = y_train[train], y_train[test]
        
        if undersampling:
            undersample = RandomUnderSampler(sampling_strategy='majority') 
            x_train_, y_train_ = undersample.fit_resample(x_train_, y_train_)         
        if nontext:
            column_trans = ColumnTransformer(
            [('child_rem', OneHotEncoder(dtype='int'),['child_rem_flag']),
             ('sec_rem', OneHotEncoder(dtype='int'),['sec_child_rem_flag']),
             ('third_rem', OneHotEncoder(dtype='int'),['third_child_rem_flag']),
             ('fourth_rem', OneHotEncoder(dtype='int'),['fourth_child_rem_flag']),
             ('fifth_rem', OneHotEncoder(dtype='int'),['fifth_child_rem_flag']),
             ('run_prop',StandardScaler(),['run_prop_prev_rem']), 
             ('run_tot',StandardScaler(),['run_prev_tot']), 
             ('run_rem',StandardScaler(),['run_prev_rem']),
             ('tfidf', TfidfVectorizer(ngram_range=(1,ngrams)), 'body_clean_no_stop')], remainder='drop')
            x_train_= column_trans.fit_transform(x_train_)
            x_test_ = column_trans.transform(x_test_)
        else:
            column_trans = ColumnTransformer(
            [('tfidf', TfidfVectorizer(ngram_range=(1,ngrams)), 'body_clean_no_stop')], remainder='drop')
            x_train_= column_trans.fit_transform(x_train_)
            x_test_ = column_trans.transform(x_test_)             

        model.fit(x_train_, y_train_)      
        yhat = model.predict_proba(x_test_)
        probs = yhat[:, 1]
        
        scores_pos = [f1_score(y_test_, to_labels(probs, t),average=None)[1] for t in thresholds]
        precision_pos = [precision_score(y_test_, to_labels(probs, t),average=None)[1] for t in thresholds]
        recall_pos = [recall_score(y_test_, to_labels(probs, t),average=None)[1] for t in thresholds]

        scores_neg = [f1_score(y_test_, to_labels(probs, t),average=None)[0] for t in thresholds]
        precision_neg = [precision_score(y_test_, to_labels(probs, t),average=None)[0] for t in thresholds]
        recall_neg = [recall_score(y_test_, to_labels(probs, t),average=None)[0] for t in thresholds]

        scores_w = [f1_score(y_test_, to_labels(probs, t),average='weighted') for t in thresholds]
        precision_w = [precision_score(y_test_, to_labels(probs, t),average='weighted') for t in thresholds]
        recall_w = [recall_score(y_test_, to_labels(probs, t),average='weighted') for t in thresholds]
        
        f1_neg_[index] = scores_neg
        precision_neg_[index] = precision_neg
        recall_neg_[index] = recall_neg   
        
        f1_pos_[index] = scores_pos
        precision_pos_[index] = precision_pos
        recall_pos_[index] = recall_pos 
        
        f1_w_[index] = scores_w
        precision_w_[index] = precision_w
        recall_w_[index] = recall_w
        index += 1    
            
    f1_neg_avg = np.average(f1_neg_,axis=0)
    precision_neg_avg = np.average(precision_neg_,axis=0)
    recall_neg_avg = np.average(recall_neg_,axis=0)

    f1_pos_avg = np.average(f1_pos_,axis=0)
    precision_pos_avg = np.average(precision_pos_,axis=0)
    recall_pos_avg = np.average(recall_pos_,axis=0)
    
    f1_w_avg = np.average(f1_w_,axis=0)
    precision_w_avg = np.average(precision_w_,axis=0)
    recall_w_avg = np.average(recall_w_,axis=0)

    ix_neg = np.argmax(f1_neg_avg) 
    ix_pos = np.argmax(f1_pos_avg)
    ix_w = np.argmax(f1_w_avg)
    
    neg_keys = ['f1_neg','precision_neg','recall_neg']
    pos_keys = ['f1_pos','precision_pos','recall_pos']
    w_keys = ['f1_w','precision_w','recall_w']
    
    d = {}
    f1_neg_lst = f1_neg_avg[ix_neg],precision_neg_avg[ix_neg],recall_neg_avg[ix_neg]
    f1_pos_lst = f1_pos_avg[ix_pos],precision_neg_avg[ix_pos],recall_neg_avg[ix_pos]
    f1_w_lst = f1_w_avg[ix_w],precision_w_avg[ix_w],recall_w_avg[ix_w]
    
    for key,val in zip(neg_keys,f1_neg_lst):
        d[key] = val
    for key,val in zip(pos_keys,f1_pos_lst):
        d[key] = val
    for key,val in zip(w_keys,f1_w_lst):
        d[key] = val
        
    return d,thresholds,f1_neg_avg,f1_pos_avg,f1_w_avg 

def plot_confusion_matrix(test_labels, predicted_labels):
    confusion = confusion_matrix(test_labels, predicted_labels)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))
    sns.heatmap(cm/np.sum(cm), annot=True, 
                fmt='.3%', cmap='Blues',cbar_kws={"format": comma_fmt})
    ax.set_xticklabels(["Intact", "Removed"])
    ax.set_yticklabels(["Intact", "Removed"])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    return
