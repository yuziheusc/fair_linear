import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

def get_corr(x,y):
    return np.corrcoef(x, y)[0,1]

def bayesian_bernoulli(x):
    n = np.shape(x)[0]
    sn = np.sum(x)
    a = sn+1.0
    b = n-sn+1.0
    mean = a/(a+b)
    var = a*b/((a+b)**2*(a+b+1))
    return mean, var**0.5

#x = np.array([0,0,0,0,0,0,0,0,0,0,0])
#print(bayesian_bernoulli(x))   

def infer_score(x,y):
## use random fortest to infer the scores
## x is the scores and y is the target or protected
## high acc on target is good, high acc on protected is bad
    def random_forest_infer(Xtrain, Ytrain, Xtest):
        clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=6)
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        return Ypred

    def logistic_infer(Xtrain, Ytrain, Xtest):
        clf = LogisticRegression(penalty='l2', C=1e64, solver='lbfgs',max_iter=100)
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        return Ypred 
        
    n_fold = 5
    kf = KFold(n_splits=n_fold, shuffle=True)
    label_a = np.array([])
    pred_a = np.array([])
    for train_index, test_index in kf.split(x):
        x_train,y_train = x[train_index], y[train_index]
        x_test,y_test = x[test_index], y[test_index]
        
        x_train = np.reshape(x_train, (-1,1))
        x_test = np.reshape(x_test, (-1,1))
        
        #y_pred = random_forest_infer(x_train, y_train, x_test)
        y_pred = logistic_infer(x_train, y_train, x_test)
        label_a = np.append(label_a, y_test)
        pred_a = np.append(pred_a, y_pred)
    acc = 1. - np.sum(np.abs(label_a - pred_a))/pred_a.shape[0]
    return acc
        
        
    

def do_evaluate_score(df, score_name, outcome, protect_list, n_bins):
    ## segement data into groups
    df_group_a = df[df[protect_list[0]]<=0.5]
    df_group_b = df[df[protect_list[0]]>0.5]
    df_group_a_nega = df_group_a[df_group_a[outcome]<=0.5]
    df_group_a_posi = df_group_a[df_group_a[outcome]>0.5]
    df_group_b_nega = df_group_b[df_group_b[outcome]<=0.5]
    df_group_b_posi = df_group_b[df_group_b[outcome]>0.5]  
    
    ## calculate majority base line
    ratio_one_y = (df[outcome]*1.0).mean()
    maj_y = max(ratio_one_y, 1.0-ratio_one_y)
    ratio_one_protect = (df[protect_list[0]]*1.0).mean()
    maj_protect = max(ratio_one_protect, 1.0-ratio_one_protect)

    ## calculate delta score
    delta_score_nega = df_group_a_nega[score_name].mean() - df_group_b_nega[score_name].mean()
    delta_score_posi = df_group_a_posi[score_name].mean() - df_group_b_posi[score_name].mean()
    delta_score = df_group_a[score_name].mean() - df_group_b[score_name].mean()
    
    ## calculate delta score binary
    Ypred_group_a = (df_group_a[score_name]>0.5)*1.0
    Ypred_group_b = (df_group_b[score_name]>0.5)*1.0
    delta_score_binary = np.mean(Ypred_group_a) - np.mean(Ypred_group_b)

    ## correlation
    corr = get_corr(df[score_name], df[protect_list[0]])

    ## correlation of linear score
    corr_linear = -1
    if(False):
        score_linear = -np.log(1./df[score_name].values - 1.)
        corr_linear = get_corr(score_linear, df[protect_list[0]])
    
    ## calculate accuracy, percentage and l2 loss
    Y = df[outcome]
    Ypred_r = df[score_name]
    Ypred = (Ypred_r>0.5)*1.0
    acc_y = accuracy_score(Y, Ypred)
    l2_y = np.mean((Y - Ypred_r)**2)
    
    ## calculate prec, rcal and f1 for pred
    prec_y = precision_score(Y, Ypred)
    rcal_y = recall_score(Y, Ypred)
    f1_y = f1_score(Y, Ypred, labels=np.unique(Ypred))

    ## calculate multual information between pred and protect
    mutual_info = mutual_info_score(df[protect_list[0]], Ypred)
    
    ## logistic log-likelihood function
    log_like = log_loss(Y, Ypred_r)
   
    ## infer the protected feature using naive logistic regression
    acc_infer_protect_post = -1
    acc_infer_target_post = -1
    if(True):
        ## infer_scores
        acc_infer_protect_post = infer_score(df[score_name].values, df[protect_list[0]].values)
        acc_infer_target_post = infer_score(df[score_name].values, df[outcome].values)
    
    ## infer the protected feature using nn, with cross validation
    acc_p = accuracy_score(df[protect_list[0]], df["_my_infer_protect"])
    prec_p = precision_score(df[protect_list[0]], df["_my_infer_protect"])
    rcal_p = recall_score(df[protect_list[0]], df["_my_infer_protect"])
    f1_p = f1_score(df[protect_list[0]], df["_my_infer_protect"], labels=np.unique(df["_my_infer_protect"]))
    
    ## infer the protect feature using random forest, with corss validation
    acc_p_rf = accuracy_score(df[protect_list[0]], df["_my_infer_protect_rf"])

    ## infer the protect feature using NB
    acc_p_nb_ber = -1
    if("_my_infer_protect_nb_ber" in df.columns):
        acc_p_nb_ber = accuracy_score(df[protect_list[0]], df["_my_infer_protect_nb_ber"])

    
    ## calibration with inverse variance weighting method
    cali_a = []; cali_b = []
    dcali_a = []; dcali_b = []
    count_a = []; count_b = []
    bins= []
    
    min_score = df[score_name].min()
    max_score = df[score_name].max()
    for i in range(n_bins):
        s1 = i * 1.0/n_bins
        s2 = (i+1) * 1.0/n_bins
        if(i==0): s1 -= 1.0
        if(i==n_bins-1): s2 += 1.0
            
        ## edge case
        #if(s2<=min_score or s1>max_score): continue
        
        df_bin_a = df_group_a[(df_group_a[score_name]>=s1) & (df_group_a[score_name]<s2)]
        df_bin_b = df_group_b[(df_group_b[score_name]>=s1) & (df_group_b[score_name]<s2)]
        
        if(df_bin_a.shape[0]<2 or df_bin_b.shape[0]<2): continue
        
        ## outdated, frequentist approach
        #mean_a = df_bin_a[outcome].mean()
        #mean_b = df_bin_b[outcome].mean()
        #std_a = df_bin_a[outcome].std()/df_bin_a.shape[0]**0.5
        #std_b = df_bin_b[outcome].std()/df_bin_b.shape[0]**0.5
        
        ## new bayesian approach
        mean_a, std_a = bayesian_bernoulli(df_bin_a[outcome].values)
        mean_b, std_b = bayesian_bernoulli(df_bin_b[outcome].values)
        
        bins.append((i+0.5)*1.0/n_bins)
        cali_a.append(mean_a)
        cali_b.append(mean_b)
        dcali_a.append(std_a)
        dcali_b.append(std_b)
        count_a.append(1.0*df_bin_a.shape[0]) # number of sample in bins
        count_b.append(1.0*df_bin_b.shape[0])
    
    bins = np.array(bins)
    cali_a = np.array(cali_a); cali_b = np.array(cali_b)
    dcali_a = np.array(dcali_a); dcali_b = np.array(dcali_b)
    count_a = np.array(count_a); count_b = np.array(count_b)
    
    cali_weight = (dcali_a**2 + dcali_b**2)**(-1)
    cali_err_l2 = (cali_a-cali_b)**2
    cali_err_total = np.sum(cali_err_l2*cali_weight)/np.sum(cali_weight)
    
    cali_err_fixed = (cali_a-bins)**2*count_a + (cali_b-bins)**2*count_b
    cali_err_fixed = np.sum(cali_err_fixed)/(np.sum(count_a+count_b))
    
    
    #print(df_group_a_nega[score_name].mean(), df_group_b_nega[score_name].mean())
    #print(df_group_a_posi[score_name].mean(), df_group_b_posi[score_name].mean())
    #print(delta_score_nega, delta_score_posi, cali_err_total)
    #print(cali_weight, cali_a, cali_b)
    #print(acc)
    return {
        "acc_y":acc_y,
        "l2_y":l2_y,
        "prec_y":prec_y,
        "rcal_y":rcal_y,
        "f1_y":f1_y,
        "acc_p":acc_p,
        "prec_p":prec_p,
        "rcal_p":rcal_p,
        "f1_p":f1_p,
        "acc_p_rf":acc_p_rf,
        "acc_p_nb_ber":acc_p_nb_ber,
        "log_like":log_like,
        "corr":corr,
        "corr_linear":corr_linear,
        "delta_score_nega":delta_score_nega,
        "delta_score_posi":delta_score_posi,
        "delta_score":delta_score,
        "delta_score_binary":delta_score_binary,
        "cali_err":cali_err_total,
        "cali_a":[bins.tolist(), cali_a.tolist(), dcali_a.tolist()],
        "cali_b":[bins.tolist(), cali_b.tolist(), dcali_b.tolist()],
        "cali_err_fixed":cali_err_fixed,
        "acc_infer_protect_post":acc_infer_protect_post,
        "acc_infer_target_post":acc_infer_target_post,
        #"acc_infer_protect_cross":acc_infer_protect_cross,
        "maj_y":maj_y,
        "maj_protect":maj_protect,
        "mutual_info":mutual_info,
    }

