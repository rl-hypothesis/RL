import pandas as pd
import numpy as np
import statistics
import itertools
from functools import partial
from multiprocessing import Pool
import sys
import random

from scipy import stats
from scipy.stats import chi2_contingency, kstest, f_oneway, pearsonr
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ztest

from config import THRESHOLD_NORMAL_DIST,THRESHOLD_INDEPENDENT_TEST
from source_code.utils.tools import name_groups

class StatisticalTestHandler:

    # EVALUATION OF MANY CASES USING MULTI PROCESSING
    def evaluate_mult(self, test_type, cases, nameGrp_2_index, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_evaluation_mult(cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_evaluation_mult(cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_evaluation_mult(cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_evaluation_mult(cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")

    def distribution_evaluation_mult(self, cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_indep
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, nameGrp_2_index = nameGrp_2_index, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()
        
        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def mean_evaluation_mult(self, cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            #test_func = self.one_sample_indep
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_indep
        elif test_arg == "Paired":
            test_func = self.one_sample_indep
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, nameGrp_2_index = nameGrp_2_index, seg_type=segmentation_type, seg_arg=segmentation_arg)  

        pool = Pool()      
        res = pool.map(f, cases)
        pool.close()

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def variance_evaluation_mult(self, cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.two_sample_indep
        elif test_arg == "One-Sample":
            #test_func = self.one_sample_indep
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, nameGrp_2_index = nameGrp_2_index, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def anova_evaluation_mult(self, cases, test_arg, nameGrp_2_index, segmentation_type, segmentation_arg):
        f = partial(compute_test_parallel, test_func=self.anova_indep, nameGrp_2_index = nameGrp_2_index, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()
            
        len_groups = len(cases[0])        
        columns = []

        for i in range(len_groups):
            columns = columns + [f"grp{i+1}", f"grp{i+1}_members", f"grp{i+1}_size"]

        len_groups = len(res[0]) - len(columns) - 1
        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        res = pd.DataFrame(res, columns=columns)
        res = res[~res["p-value"].isna()]

        return res

    # EVALUATION OF MANY CASES (WITHOUT MULTI PROCESSING)
    def evaluate(self, test_type, cases, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")

    def distribution_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_dist_test
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg
        
        res = []
        for case in cases:
            res.append( test_func(case[0], case[1], segmentation_type) )

        res = pd.DataFrame(res, columns=["p-value", "chi-squared test"])

        ll = res[~res["p-value"].isna()]
        ll = ll[ ll["p-value"] < 0.05 ]
        print( len(ll) )

        return res

    def mean_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_mean_test
        elif test_arg == "Paired":
            test_func = self.paired_mean_test
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        
        self.test_arg = test_arg

        res = []
        for case in cases:
            res.append( test_func(case[0], case[1], segmentation_type) )

        res = pd.DataFrame(res, columns=["p-value", "chi-squared test"])

        return res

    def variance_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.F_test
        elif test_arg == "One-Sample":
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg

        res = []
        for case in cases:
            res.append( test_func(case[0], case[1], segmentation_type) )

        res = pd.DataFrame(res, columns=["p-value", "chi-squared test"])

        return res

    def anova_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        
        res = []
        for case in cases:
            res.append( self.anova(case, segmentation_type) )

        len_groups = len(cases[0])
        
        columns = []
        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        res = pd.DataFrame(res, columns=columns)

        return res

    # EVALUATION OF ONE CASE
    def evaluate_one(self, test_type, cases, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")
 
    def mean_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_mean_test
        elif test_arg == "Paired":
            test_func = self.paired_mean_test
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        self.test_arg = test_arg
        
        res = test_func(cases[0], cases[1], segmentation_type)
        res = pd.DataFrame([res], columns=["p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def distribution_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_dist_test
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg
        
        res = test_func(cases[0], cases[1], segmentation_type)
        res = pd.DataFrame([res], columns=["p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]
        
        return res

    def variance_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.F_test
        elif test_arg == "One-Sample":
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg
        
        res = test_func(cases[0], cases[1], segmentation_type)
        res = pd.DataFrame([res], columns=["p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def anova_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):

        res = self.anova(cases, segmentation_type)

        len_groups = len(cases)
        
        columns = []
        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        l = len(res)
        res = np.array(res).reshape((1,l))

        res = pd.DataFrame(res, columns=columns)
        res = res[~res["p-value"].isna()]


        return res

    # INDEPENDANCE TEST
    def one_sample_indep(self,i,j,segmentation_type):
        return 0.001, 1

    def two_sample_indep(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)


        return 0.001, p

    def anova_indep(self,i, segmentation_type):
        values_s = []
        for v in i:
            values_s.append( aggregate_values(v,segmentation_type) )
        
        t_normal = True
        for v in values_s:
            t_normal = t_normal and self.test_normal_distribution(v) 
        
        if not (t_normal):
            for j in range(len(values_s)):
                values_s[j] = (values_s[j] - min(values_s[j])) / (max(values_s[j]) - min(values_s[j]))

        indep_cases = list(itertools.combinations(values_s,2))

        indep_tests = []

        for i in indep_cases:
            contingency_df = self.generate_contingency_table(i[0], i[1])
            stat, p, dof, expected = chi2_contingency(contingency_df)
            indep_tests.append(p)
            #indep_tests.append(1)

        return [0.001,*indep_tests]

    # P-VALUE TESTS
    def one_sample_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        return stats.ttest_1samp(values_1, float(j))[1], 1

    def one_sample_dist_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        return stats.kstest(values_1, j)[1], 1

    def two_sample_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        return stats.ttest_ind(values_1, values_2, equal_var=False)[1], 1

    def paired_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        if len(values_1)==len(values_2):
            k = 1
        else:
            if len(values_1) > len(values_2):
                values_1 = values_1[:len(values_2)]
            else:
                values_2 = values_2[:len(values_1)]

        return stats.ttest_rel(values_1, values_2)[1], 1

    def two_sample_dist_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        return kstest(values_1, values_2)[1], 1

    def z_test(self, i, j, freq="D"):
        values_1 = aggregate_values(df=i, freq=freq) / self.test_arg[1]
        values_2 = aggregate_values(df=j, freq=freq) / self.test_arg[1]
        
        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        return ztest(values_1, values_2)[1], 1
    
    def F_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        F = statistics.variance(values_1) / statistics.variance(values_2)
        df1 = len(values_1)-1
        df2 = len(values_2)-1
        
        return stats.f.sf(F, df1, df2), 1

    def one_sample_variance_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        df = len(values_1)-1
        chi = df * statistics.variance(values_1) / float(j)
        return 1-stats.chi2.sf(chi, df), 1

    def anova(self,i, segmentation_type):
        values_s = []
        for v in i:
            values_s.append( aggregate_values(v,segmentation_type) )
        
        t_normal = True
        for v in values_s:
            t_normal = t_normal and self.test_normal_distribution(v) 
        
        if not (t_normal):
            for j in range(len(values_s)):
                values_s[j] = (values_s[j] - min(values_s[j])) / (max(values_s[j]) - min(values_s[j]))

        indep_cases = list(itertools.combinations(values_s,2))
        indep_tests = [1 for i in indep_cases]

        return [f_oneway(*values_s)[1],*indep_tests]

    def test_normal_distribution(self, values_1, threshold_normal_dist=THRESHOLD_NORMAL_DIST):
        return len(values_1) >= 8 and stats.normaltest(values_1)[1] >= threshold_normal_dist

    def generate_contingency_table(self, values_1, values_2):

        data = pd.DataFrame([values_1, values_2]).T.fillna(0).astype(bool)
        return pd.DataFrame([data[0].value_counts(), data[1].value_counts()]).T

def aggregate_values(df, segmentation_type, freq="D", col="rating"):
    if segmentation_type is None:
        return df[col].values
    else:
        return df.groupby(pd.Grouper(freq=freq)).agg({"purchase": "sum"})["purchase"].values #####ICI

def compute_test_parallel(args, test_func, nameGrp_2_index, seg_type, seg_arg):
    #columns = ['cust_id','article_id','rating','purchase', 'transaction_date']
    columns = ['cust_id','article_id','rating',]

    if len(args) < 3 :
        name_1 = args[0]
        i = nameGrp_2_index[0][name_1]
        i = i[['cust_id','article_id','rating']]
        
        j = args[1]

        #if (isinstance(j,int) == False) and (isinstance(j,str) == False) and (isinstance(j,float) == False) :
        if len(nameGrp_2_index) > 1:
            #Two samples
            name_2 = j
            j = nameGrp_2_index[1][name_2]
            j = j[['cust_id','article_id','rating']]
            set_2_group = set(j.cust_id.unique())
            len_2_group = len(j)
        else:
            #One sample
            name_2 = j
            set_2_group = {}
            len_2_group = 0

        return [name_1, set(i.cust_id.unique()), len(i), name_2, set_2_group, len_2_group, *test_func(i, j, seg_type)]
    else: #ANOVA
        periods = args

        periods_2 = periods
        periods = [ i[['cust_id','article_id','rating','purchase', 'transaction_date']] for i in periods ]

        periods_2 = [ (name_groups(i), set(i.cust_id.unique()), len(i)) for i in periods_2]
        periods_2 = list(itertools.chain(*periods_2))

        return [*periods_2, *test_func(periods, seg_type)]
