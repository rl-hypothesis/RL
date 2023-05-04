import pandas as pd
import numpy as np
from itertools import product,chain,combinations
from multiprocessing import Pool
from functools import partial
import pickle
import random
import time
import math
import sys
import os

import config
from config import THRESHOLD_INDEPENDENT_TEST
from source_code.core.hypothesis_evaluation.test_evaluation import StatisticalTestHandler
from source_code.core.pivot_handler import PeriodHandler
from source_code.core.group_handler import GroupHandler
from source_code.models.models import get_data
from source_code.utils.tools import add_index, name_groups
import warnings

idx2method = {-1:'TRAD_BY', 0:'TRAD_BN',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',\
5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',8:'Ɛ-Hybrid',9:'Ψ-Support', 10:'Min_VAL', 11:'SMT_cov', 12:'SMT'}

pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore', category=UserWarning)

def test_groups(group, group_name, split_attribute, grp_type, names, hierarchy, dimension, top_n, num_hyps, approach,\
agg_type, test_arg, users_1, support, alpha, verbose=False):
    
    """
    group : [group]
    top_n : integer - Maximum of returned results (n)
    approach : list of Integer - p value / coverage based / Alpha investinf
    grp_type : list - features for clustering
    agg_type : string - Aggregation function
    dimension : string - variable used by aggregate function
    test_arg : Pair - [Type of test (one sample, two samples ...), Value for One Sample]
    support : integer - Minimum number of points in a group
    alpha : float - Threshold of p-value
    verbose : Boolean - If True, print of steps
    """

    test_handler = StatisticalTestHandler()
    #group_handler = GroupHandler()

    groups = []
    nameGrp_2_index = []
    group_2_users = []

    for j,grp in enumerate(group_name):

        liste = []
        dicte = dict()
        dicte_2 = dict()

        key = (grp, split_attribute)
        
        if key in hierarchy:
            for i in hierarchy[ key ]:
                df = names[j][i]

                liste.append(i)
                dicte[i] = df
                dicte_2[i] = set(df.cust_id.unique())

        nameGrp_2_index.append( dicte )
        group_2_users.append( dicte_2 )
        groups.append(liste)

    
    if len(groups[0]) == 0:
        return [],[], nameGrp_2_index
    
    cases = []

    #Get the list of users of each group | dict = key: Group description - value: List of users 
    group1_2_users = group_2_users[0]

    #Get the list of users of each period
    users1 = set()
    users1 = users_1

    users2 = set()

    if len(groups) < 2: #If ONE-SAMPLE == TRUE
        groups.append( [test_arg[1]] )
        one_sample = True

    len_e = len(groups[0])
    len_h = len(groups[1])

    if verbose==True:
        print(f'Grouping Data Done - All condidates keeping ones above the support : {len_e * len_h}')

    # Cartesian product +  Evaluate all cases
    if one_sample == True:
        sizes = [len(perd) for perd in groups[0]]
        sizes = int(sum(sizes)/len(sizes))
    else:
        sizes = [ [len(perd) for perd in perds] for perds in groups ]
        sizes = [ sum(perds)/len(perds) for perds in sizes]
        sizes = int(sum(sizes)/len(sizes))

    cases = list(product(*groups)) #Cartesian product

    len_groups = len(groups)

    #Get independant and normally distributed samples
    result = test_handler.evaluate_mult(agg_type, cases, nameGrp_2_index, test_arg[0])
    #result = result[result["chi-squared test"] > THRESHOLD_INDEPENDENT_TEST]

    nb__ = len(result)

    if verbose==True:
        print( 'All independant cases : ',nb__ )
    
    col = [f'grp{i+1}' for i in range(len_groups)]

    col2 = []
    for i in range(len_groups):
        col2.append(f'grp{i+1}')
        col2.append(f'grp{i+1}_size')

    col2.append('p-value')
    
    pairs = result.copy()
    nb__ = len(pairs)

    if verbose == True:
        print('Ground Truth Done !')

    pairs_ = dict()
    
    #Create samples
    for num_hyp in num_hyps:
        pairs = pairs.sample(frac=1)

        leng_sample = int(num_hyp*nb__)

        if num_hyp < 1:
            sampled_pairs = pairs.head(leng_sample)
            sampled_pairs = sampled_pairs.sample(frac=1)
        else:
            sampled_pairs = pairs.copy()
            sampled_pairs = sampled_pairs.sample(frac=1)
        
        pairs_[num_hyp] = sampled_pairs.copy()
    
    pairs_['nb__'] = nb__

    args = product(num_hyps,top_n)

    f = partial(worker, pairs=pairs_, col=col, col2=col2, test_arg=test_arg, nameGrp_2_index=nameGrp_2_index, agg_type=agg_type, approach=approach,\
    group1_2_users=group1_2_users, group2_2_users=group1_2_users, users1=users1, users2=users2, sizes=sizes, alpha=alpha, one_sample=one_sample, len_groups=len_groups,\
    time_groups=0, verbose=verbose) 

    pool = Pool()      
    res = pool.map(f, args)
    pool.close()

    numHypo_2_stats = dict()
    numHypo_2_results = dict()

    #Unify the results
    for posi, i in enumerate(zip(*res)):
        len_ = len(i)
        dict_inter = dict()

        for idx in range(len_):
            samples = i[idx].keys()

            for sample in samples:
                if sample not in dict_inter:
                    dict_inter[sample] = i[idx][sample]
                else:
                    dict_inter[sample] = {**dict_inter[sample], **i[idx][sample]}

        if posi == 0:
            numHypo_2_stats = dict_inter
        else:
            numHypo_2_results = dict_inter

    return numHypo_2_stats, numHypo_2_results, nameGrp_2_index

def worker(args, pairs, col, col2, test_arg, nameGrp_2_index, agg_type, approach, group1_2_users, group2_2_users, users1, users2, sizes, alpha, one_sample,\
 len_groups, time_groups, verbose):

    #columns_stat = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value',
    #'Best_pairs_BY','Min_p_value_BY','Max_p_value_BY','Sum_p_value_BY','Cov_period_1_BY','Cov_period_2_BY','Cov_total_BY','p_value_adjustement_BY_time',
    #'Best_pairs_B','Min_p_value_B','Max_p_value_B','Sum_p_value_B','Cov_period_1_B','Cov_period_2_B','Cov_total_B','p_value_adjustement_B_time','grouping_time']

    columns_stat = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value',
    'Best_pairs_B','Min_p_value_B','Max_p_value_B','Sum_p_value_B','Cov_period_1_B','Cov_period_2_B','Cov_total_B','p_value_adjustement_B_time','grouping_time']

    columns_stat_2 = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value',
    'Best_pairs_BY','Min_p_value_BY','Max_p_value_BY','Sum_p_value_BY','Cov_period_1_BY','Cov_period_2_BY','Cov_total_BY','p_value_adjustement_BY_time','grouping_time']

    columns_stat_Alpha_investing_ = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value']
    
    template_colum = ['Best_pairs','Min_p_value','Max_p_value','Sum_p_value','Cov_period_1','Cov_period_2','Cov_total','p_value_adjustement_time']

    top_n = [args[1]] #Get the number of results
    hyp = [args[0]] #Get the sample size

    nb__ = pairs['nb__']

    test_handler = StatisticalTestHandler()
    
    numHypo_2_stats = dict()
    numHypo_2_results = dict()

    for num_hyp in hyp:
        leng_sample = int(num_hyp*nb__)

        sampled_pairs = pairs[num_hyp]
        
        result_alpha = sampled_pairs[col].copy() #Data for Cover alpha
        result_alpha_ = sampled_pairs[col2].copy() #Data for other alpha investing
        
        result = sampled_pairs
        
        time_process_all_cases = 0# time.time() - start_time

        if verbose==True:
            print('p-values and chi2-test Done')

        #result = result[~result["p-value"].isna()]

        all_candidates = len(result)
        nb_under_p_value = len(result)

        if 'grp1_members' in result.columns:
            columns = [f'grp{i+1}_members' for i in range(len_groups)]
            result.drop(columns=columns,inplace=True)

        all_stats = dict()
        all_results = dict()

        if len(result)>0:
            #min_p_value = result["p-value"].min()
            #max_p_value = result["p-value"].max()

            result1 = result.drop(["chi-squared test"], axis=1).copy()

            del result
            
            for top in top_n: # Thresholds (n)
                results = dict()
                stats = []

                if verbose==True:
                    print('n = ',top)

                for appr in approach: # Methods
                    res = []
                    time_exec = []

                    if appr == 4: #Alpha investing support
                        #methods = [20,50,100,200,300,500] #Gammas
                        methods = [500]
                        process_time = 0

                        columns_stat_Alpha_investing = columns_stat_Alpha_investing_

                        for mth in methods:
                            columns_stat_Alpha_investing = columns_stat_Alpha_investing+[column+f'_' for column in template_colum]
                        
                        columns_stat_Alpha_investing = columns_stat_Alpha_investing + ['grouping_time']
                        
                        for mtd in methods :

                            if 'coverage' in result_alpha.columns:
                                result_alpha.drop(columns=['coverage','cov_1','cov_2'], inplace=True)

                            start_time = time.time()

                            res.append(alpha_investing_cov(top, result_alpha, nameGrp_2_index, agg_type,\
                            test_arg[0], group1_2_users, group2_2_users, users1, users2, alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)
                            
                            if verbose==True:
                                print(f'Top {top} (Sample : {num_hyp*100}) - Alpha investing : 4,{mtd} - time processing = {time_exec[-1]} s')
                    
                    elif appr > 4 and appr < 10:
                        if appr == 5:
                            #methods = [0.25, 0.5, 0.75, 0.9]
                            methods = [0.9]
                        elif appr in [6,7]:
                            #methods = [20,50,100,200,300,500]
                            methods = [500]
                        elif appr == 8:
                            #methods = [ (0.25,500,500), (0.5,500,500), (0.75,500,500), (0.25,100,100), (0.5,100,100), (0.75,100,100),\
                            #(0.25,200,200), (0.5,200,200), (0.75,200,200) ]
                            methods = [(0.75,200,200)]
                        elif appr == 9 :
                            #methods = [1/2, 1/3, 1/4, 1/5, 1/6]
                            methods = [1/2]

                        process_time = 0

                        columns_stat_Alpha_investing = columns_stat_Alpha_investing_

                        for mth in methods:
                            columns_stat_Alpha_investing = columns_stat_Alpha_investing+[column+f'_' for column in template_colum]
                        
                        columns_stat_Alpha_investing = columns_stat_Alpha_investing + ['grouping_time']
                        
                        for mtd in methods :
                            start_time = time.time()

                            res.append(alpha_investing(top, appr, result_alpha_, nameGrp_2_index, agg_type,\
                            test_arg[0], group1_2_users, group2_2_users, users1, users2, sizes, alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)
                            
                            if verbose==True:
                                print(f'Top {top} (Sample : {num_hyp*100}) - Alpha investing : {appr},{mtd} - time processing = {time_exec[-1]} s')
                    
                    elif appr == 0:
                        #methods = ["fdr_by", "fdr_b"]
                        methods = ["fdr_b"]

                        process_time = time_process_all_cases

                        #for mtd in ["fdr_by", "fdr_b"]:
                        for mtd in ["fdr_b"]:
                            start_time = time.time()

                            res.append(trad_and_cover_g(top, appr, result1, group1_2_users, group2_2_users, users1, users2, alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)

                            if verbose==True:
                                print(f'Top {top} (Sample : {num_hyp*100}) - Method : {appr} - Adjust : {mtd} - time processing = {time_exec[-1]+time_process_all_cases} s')
                    
                    elif appr == -1:
                        #methods = ["fdr_by", "fdr_b"]
                        methods = ["fdr_by"]

                        process_time = time_process_all_cases

                        #for mtd in ["fdr_by", "fdr_b"]:
                        for mtd in ["fdr_by"]:
                            start_time = time.time()

                            res.append(trad_and_cover_g(top, appr, result1, group1_2_users, group2_2_users, users1, users2, alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)

                            if verbose==True:
                                print(f'Top {top} (Sample : {num_hyp*100}) - Method : {appr} - Adjust : {mtd} - time processing = {time_exec[-1]+time_process_all_cases} s')
                            
                    stat = [int(num_hyp*nb__), all_candidates, nb_under_p_value,\
                    result1['p-value'].min(), result1['p-value'].max(), result1['p-value'].sum()]

                    if one_sample == False:
                        for i in range(len(methods)):
                            stat = stat + [len(res[i]),res[i]['p-value'].min(), res[i]['p-value'].max(), res[i]['p-value'].sum() ,\
                            res[i]['coverage_gained_1'].max(), res[i]['coverage_gained_2'].max(),\
                            (res[i]['coverage_gained_1'].max()+res[i]['coverage_gained_2'].max())/2,time_exec[i]+process_time]
                    else:
                        for i in range(len(methods)):
                            stat = stat + [len(res[i]),res[i]['p-value'].min(), res[i]['p-value'].max(), res[i]['p-value'].sum() ,\
                            res[i]['coverage_gained_1'].max(), res[i]['coverage_gained_2'].max(),\
                            res[i]['coverage_gained_1'].max(),time_exec[i]+process_time]

                    if appr == -1:
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat_2 ).T)
                    elif appr < 4 :
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat ).T)
                    elif appr==10 or appr==11:
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat_min_val ).T)
                    else:
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat_Alpha_investing ).T)
                    
                    results[ idx2method[appr] ] = res
                
            all_stats[top] = stats
            all_results[top] = results
        
        numHypo_2_stats[num_hyp*100] = all_stats
        numHypo_2_results[num_hyp*100] = all_results

    return [numHypo_2_stats,numHypo_2_results]

def trad_and_cover_g(top_n, approach, result, group1_2_users, group2_2_users, users1, users2, alpha=0.05, method="fdr_by"):
    """
    approach :
    0 = Traditionnal
    1 = Sum of Coverage
    2 = Coverage Grp 1
    3 = Coverage Grp 2
    """
    top = top_n

    if top == -1:
        top = 1000000000
    
    coverage_grp1 = 0
    coverage_grp2 = 0

    users_1 = users1
    users_2 = users2

    len_u1 = len(users1)
    len_u2 = len(users2)

    one_sample = False

    if len_u2 == 0:
        one_sample = True
        len_u2 = 1

    res = []

    if result.grp2.dtype == float:
        result.grp2 = result.grp2.astype(str)

    m = result.shape[0]
    nb_columns = len(result.columns)

    grp1_duplicated = set(result.grp1.unique())

    if len(group2_2_users)==0:
        grp2_duplicated = set()
        group2_2_users[ list(result.grp2.unique())[0] ]={}
    else:
        grp2_duplicated = set(result.grp2.unique())

    result = result.sort_values("p-value").reset_index(drop=True)

    if approach > 0: #Cover_G
        d = result[['grp1','grp2']]
        d['key'] = d.apply(lambda x : x[0]+' '+str(x[1]),axis=1)
        d.drop(columns=['grp1','grp2'],inplace=True)
        d['val'] = d.index
        d = d.set_index('key').T.to_dict('list')

    for n in range(len(result)):
        if approach > 0: #Cover_G
            if approach == 2:
                result['coverage'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )
            if approach == 3:
                result['coverage'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )
            if approach == 1 :
                result['cov_1'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )
                result['cov_2'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )
                result['coverage'] = result['cov_1']+result['cov_2']
                result.drop(columns=['cov_1','cov_2'],inplace=True)

            result = result[result.coverage > 0]

            if len(result)>0:
                best = result['coverage'].idxmax()
            
            result.drop(columns=['coverage'],inplace=True)
            
        if (len(result)==0) or (top==0):
            break

        if approach > 0: #Cover_G
            if nb_columns > 5: #ANOVA CASE
                grp1 = result.loc[[best]].values.tolist()[0]
                p = grp1[-1]
                grp1 = grp1[:-1]
                gr = [ grp1[jj] for jj in range(0,len(grp1),2)]
                grp1, grp2 = gr[0],gr[1]

            else:
                grp1, grp1_size, grp2, grp2_size, p = result.loc[[best]].values.tolist()[0]

            result = result.drop(best)
            pos = d[grp1+' '+str(grp2)][0]
        else: #TRAD
            if nb_columns > 5: #ANOVA CASE
                grp1 = result.head(1).values.tolist()[0]
                p = grp1[-1]
                grp1 = grp1[:-1]
                gr = [ grp1[jj] for jj in range(0,len(grp1),2)]
                grp1, grp2 = gr[0],gr[1]
            else:
                grp1, grp1_size, grp2, grp2_size, p = result.head(1).values.tolist()[0]

            result = result.iloc[1:]
            pos = n

        limit = compute_limit(alpha, pos + 1, m, method)

        if p >= limit:
            if approach == 0:
                break
            else:
                continue
        
        if grp1 in grp1_duplicated:
            grp1_duplicated.remove(grp1)

            inter = users_1.intersection( group1_2_users[grp1] )
            users_1 = {i for i in users_1 if i not in inter} 
            coverage_grp1 += len( inter )

        if one_sample == False:
            if grp2 in grp2_duplicated:
                grp2_duplicated.remove(grp2)

                inter = users_2.intersection( group2_2_users[grp2] )
                users_2 = {i for i in users_2 if i not in inter} 
                coverage_grp2 += len( inter )
        else:
            coverage_gained_2 = 0

        res.append( (grp1, grp2, coverage_grp1/len_u1, coverage_grp2/len_u2, p) )

        top = top-1

    df_results = pd.DataFrame(res, columns=["group1","group2","coverage_gained_1","coverage_gained_2","p-value"])

    return df_results

def alpha_investing_cov(top_n, result, nameGrp_2_index, test_type, test_arg, group1_2_users, group2_2_users, users1, users2, alpha=0.05, method="fdr_by"):        
    
    test_handler = StatisticalTestHandler()

    top = top_n

    if top == -1:
        top = 1000000000
    
    coverage_grp1 = 0
    coverage_grp2 = 0

    users_1 = users1
    users_2 = users2

    len_u1 = len(users1)
    len_u2 = len(users2)

    one_sample = False

    if len_u2 == 0:
        one_sample = True
        len_u2 = 1

    res = []

    if result.grp2.dtype == float:
        result.grp2 = result.grp2.astype(str)

    m = result.shape[0]
    nb_columns = len(result.columns)

    grp1_duplicated = set(result.grp1.unique())

    if len(group2_2_users)==0:
        grp2_duplicated = set()
        group2_2_users[ list(result.grp2.unique())[0] ]={}
    else:
        grp2_duplicated = set(result.grp2.unique())

    w0 = (1-alpha)*alpha
    w_next = w0

    if isinstance(method,str):
        gamma = 20
    else:
        gamma = method
    
    alpha_0 = w0 / (gamma+w0)

    cov_calcul = True
    n_refus = 0

    while (w_next > 0) and (top > 0):
        if cov_calcul == True:
            result['cov_1'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )

            if one_sample:
                result['cov_2'] = result.grp2.apply(lambda x : 0 )
            else:
                result['cov_2'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )

            result['coverage'] = result['cov_1']+result['cov_2']
            result = result[result.coverage > 0]

        cov_calcul = False

        if len(result) > 0 :
            best = result['coverage'].idxmax()
        else:
            result.drop(columns=['cov_1','cov_2','coverage'],inplace=True)
            break

        case = []

        if nb_columns > 2: #ANOVA CASE
            grp1 = result.loc[[best]].values.tolist()[0]
            cov1,cov2 = grp1[-3], grp1[-2]
            grp1 = grp1[:-3]
            case = []
            for idx_grp, grp_i in enumerate(grp1):
                case.append(nameGrp_2_index[idx_grp][grp_i])

            grp2 = grp1[1]
            grp1 = grp1[0]
        else:
            grp1, grp2, cov1, cov2, cov_tot  = result.loc[[best]].values.tolist()[0]
            if len(nameGrp_2_index)==1: #One-Sample test
                case = [ nameGrp_2_index[0][grp1],grp2 ]
            else:
                case = [ nameGrp_2_index[0][grp1], nameGrp_2_index[1][grp2] ]

        result = result.drop(best)

        p = test_handler.evaluate_one(test_type, case, test_arg)

        if len(p)==0:
            continue
        
        p = p['p-value'].values[0]

        cov_j = (cov1/len_u1) + (cov2/len_u2)
    
        if one_sample == False:
            cov_j = cov_j / 2

        alpha_j = alpha_0 * math.sqrt(cov_j)
        err = w_next - ( alpha_j/(1-alpha_j) )

        if err >= 0:
            if p <= alpha_j:
                w_next = w_next + alpha
            else:
                w_next = err
                n_refus = n_refus + 1
                continue
        else:
            n_refus += 1
            continue

        cov_calcul = True

        if grp1 in grp1_duplicated:
            grp1_duplicated.remove(grp1)
            inter = users_1.intersection( group1_2_users[grp1] )
            users_1 = {i for i in users_1 if i not in inter} 

            coverage_grp1 += len(inter)

        if one_sample:
            coverage_grp2 = 0
        else:
            if grp2 in grp2_duplicated:
                grp2_duplicated.remove(grp2)
                inter = users_2.intersection( group2_2_users[grp2] )
                users_2 = {i for i in users_2 if i not in inter} 

                coverage_grp2 += len(inter)

        res.append( (grp1, grp2, coverage_grp1/len_u1, coverage_grp2/len_u2, p , w_next-alpha, alpha_j, n_refus) )
        top = top-1    

        n_refus = 0

    df_results = pd.DataFrame(res, columns=["group1","group2","coverage_gained_1","coverage_gained_2","p-value","wealth","alpha_j","nb_refus_before"])

    return df_results

def alpha_investing(top_n, approach, result, nameGrp_2_index, test_type, test_arg, group1_2_users, group2_2_users, users1, users2, total_size, alpha=0.05, method="fdr_by"):
    """
    approach : 
    5 = beta Farsighted
    6 = gamma Fixed
    7 = delta Hopeful
    8 = epsilon Hybrid
    9 = psi Support
    """

    test_handler = StatisticalTestHandler()

    top = top_n

    if top == -1:
        top = 1000000000
    
    coverage_grp1 = 0
    coverage_grp2 = 0

    users_1 = users1
    users_2 = users2

    len_u1 = len(users1)
    len_u2 = len(users2)

    one_sample = False

    if len_u2 == 0:
        one_sample = True
        len_u2 = 1

    res = []

    if result.grp2.dtype == float:
        result.grp2 = result.grp2.astype(str)

    m = result.shape[0]
    nb_columns = len(result.columns)

    grp1_duplicated = set(result.grp1.unique())

    if len(group2_2_users)==0:
        grp2_duplicated = set()
        group2_2_users[ list(result.grp2.unique())[0] ]={}
    else:
        grp2_duplicated = set(result.grp2.unique())

    w0 = (1-alpha)*alpha
    w_next = w0

    if isinstance(method,str):
        if approach == 5:
            param = 0.5
        elif approach in [6,9]:
            param = 20
        elif approach == 7:
            param = 50
        elif approach == 8:
            param = 0.5
            param_gamma = 20
            param_delta = 50
    else:
        if approach == 8:
            param = method[0]
            param_gamma = method[1]
            param_delta = method[2]
        else:
            param = method

    if approach ==9:
        alpha_0 = w0/(500+w0)
    else:
        alpha_0 = w0/(param+w0)

    k_0 = w_next

    slides = np.array([])
    distance_slide = 50

    n_refus = 0
    
    while (w_next > 0) and (top > 0):

        if len(result) == 0 :
            break

        case = []

        if nb_columns > 5: #ANOVA CASE
            grp1 = result.head(1).values.tolist()[0]
            grp1 = grp1[:-1]

            gr = [ grp1[jj] for jj in range(0,len(grp1),2)]
            sizes = [ grp1[jj] for jj in range(1,len(grp1),2)]

            for idx_grp, grp_i in enumerate(gr):
                case.append(nameGrp_2_index[idx_grp][grp_i])

            grp1, grp2 = gr[0],gr[1]

            local_size = sum(sizes)/len(sizes)
        else:
            grp1, grp1_size, grp2, grp2_size, p = result.head(1).values.tolist()[0]

            if len(nameGrp_2_index)==1: #One-Sample test
                case = [ nameGrp_2_index[0][grp1],grp2 ]
                local_size = grp1_size
            else:
                case = [ nameGrp_2_index[0][grp1], nameGrp_2_index[1][grp2] ]
                local_size = (grp1_size + grp2_size)/2

        result = result.iloc[1:]

        p = test_handler.evaluate_one(test_type, case, test_arg)

        if len(p)==0:
            continue
        
        p = p['p-value'].values[0]

        if approach == 5:
            #Beta Farsighted
            alpha_j = min(alpha, (w_next * (1-param))/(1+w_next*(1-param)) )

            if p < alpha_j:
                w_next = w_next + alpha
            else:
                w_next = param * w_next
                n_refus += 1
                continue
        elif approach == 6:
            # Gamma Fixed
            alpha_j = alpha_0
            err = w_next - ( alpha_0/(1-alpha_0) )

            if err < 0 :
                break
            elif p < alpha_0:
                w_next = w_next + alpha
            else:
                w_next = w_next - w0/param
                n_refus += 1
                continue
        elif approach == 7:
            # Delta Hopeful
            alpha_j = alpha_0
            err = w_next - ( alpha_0/(1-alpha_0) )

            if err < 0:
                break
            elif p < alpha_0:
                w_next = w_next + alpha
                alpha_0 = min(alpha, w_next/(param+w_next))
            else:
                w_next = err
                n_refus += 1
                continue
        elif approach == 8:
            # Epsilon Hybrid
            if len( np.where(slides[-distance_slide:]==1)[0] ) <= len(slides[-distance_slide:]) * param :
                alpha_j = w0/(param_gamma+w0)
            else:    
                alpha_j = min( alpha, k_0/(param_delta+k_0) )

            err = w_next - ( alpha_j/(1-alpha_j) )

            if err >= 0:
                if p < alpha_j:
                    w_next = w_next + alpha
                    k_0 = w_next
                    slides = np.append(slides,1)
                else:
                    w_next = err
                    slides = np.append(slides,0)
                    n_refus += 1
                    continue
            else:
                n_refus += 1
                continue
        elif approach == 9:
            #Psi Support
            alpha_j = alpha_0 * math.pow(local_size/total_size, param)

            err = w_next - ( alpha_j/(1-alpha_j) )

            if err >= 0:
                if p < alpha_j:
                    w_next = w_next + alpha
                else:
                    w_next = err
                    n_refus += 1
                    continue
            else:
                n_refus += 1
                continue

        if grp1 in grp1_duplicated:
            grp1_duplicated.remove(grp1)
            inter = users_1.intersection( group1_2_users[grp1] )
            users_1 = {i for i in users_1 if i not in inter} 

            coverage_grp1 += len(inter)

        if one_sample:
            coverage_grp2 = 0
        else:
            if grp2 in grp2_duplicated:
                grp2_duplicated.remove(grp2)
                inter = users_2.intersection( group2_2_users[grp2] )
                users_2 = {i for i in users_2 if i not in inter} 

                coverage_grp2 += len(inter)

        res.append( (grp1, grp2, coverage_grp1/len_u1, coverage_grp2/len_u2, p , w_next-alpha, alpha_j, n_refus) )
        top = top-1    
        n_refus = 0

    df_results = pd.DataFrame(res, columns=["group1","group2","coverage_gained_1","coverage_gained_2","p-value","wealth","alpha_j","nb_refus_before"])

    return df_results

def compute_limit(alpha, n, m, method="fdr_by"):
    if method == "fdr_by":
        return alpha * n / (m * sum(1 / ii for ii in range(1, m+1)))
    if method == "fdr_b":
        return alpha / m
