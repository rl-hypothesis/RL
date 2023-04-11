import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool
import random
import math
from collections import deque, namedtuple
import itertools
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_DIR+'/../Group_testing/'

sys.path.append(os.path.dirname(SCRIPT_DIR))

from source_code.models.models import get_data
from source_code.core.pivot_handler import PeriodHandler
from source_code.core.group_handler import GroupHandler
from source_code.utils.tools import name_groups
from source_code.core.hypothesis_evaluation.test_handler_2 import test_groups
from config import THRESHOLD_INDEPENDENT_TEST

torch.autograd.set_detect_anomaly(True)

class Agent(nn.Module):
    def __init__(self, state_size, hidden_size, num_groups_per_step, num_attr, num_agg):

        super(Agent, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.num_groups_per_step = num_groups_per_step
        self.num_agg = num_agg
        self.num_attr = num_attr

        self.fc_1 = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, num_groups_per_step*num_attr).double()
        #torch.nn.init.xavier_uniform_(self.fc_3.weight)
        
        
        self.fc_agg = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_agg.weight)
        self.relu_agg = nn.ReLU()

        self.fc_agg_2 = nn.Linear(hidden_size, num_agg).double()
        #torch.nn.init.xavier_uniform_(self.fc_agg_2.weight)

    def forward(self, input_state):
        out = self.fc_1(input_state.clone())
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.relu2(out_2)

        out_agg = self.fc_agg(input_state.clone())
        out_agg = self.relu_agg(out_agg)

        return self.fc_3(out_2), self.fc_agg_2(out_agg)

class Agent_explore_exploit(nn.Module):
    def __init__(self, state_size, hidden_size):

        super(Agent_explore_exploit, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.fc_1 = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, 2).double()
        #torch.nn.init.xavier_uniform_(self.fc_3.weight)

    def forward(self, input_state):
        out = self.fc_1(input_state.clone())
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.relu2(out_2)

        return self.fc_3(out_2)


class GetRegions(object):
    def __init__(self, policies, dataset, num_attr=6, num_groups_per_step=4, num_agg=3):
        self.support = 5

        self.num_agg = num_agg
        self.num_attr = num_attr
        self.num_groups_per_step = num_groups_per_step
        
        self.hidden_size = 128
        self.group_vector_size = 69
        self.state_size = self.num_groups_per_step*self.group_vector_size+self.num_agg
        
        self.policies = {}
        self.expl_policies = {}

        for policy in policies:

            policy_net = Agent(self.state_size, self.hidden_size, self.num_groups_per_step, self.num_attr, self.num_agg)
            policy_net.load_state_dict(torch.load(f'policy_net_{policy}.pth'))
            policy_net.eval()

            policy_net_explore_exploit = Agent_explore_exploit(self.state_size, self.hidden_size)
            policy_net_explore_exploit.load_state_dict(torch.load(f'policy_net_explore_exploit_{policy}.pth'))
            policy_net_explore_exploit.eval()

            self.policies[policy] = policy_net
            self.expl_policies[policy] = policy_net_explore_exploit

        self.__prepare_data(dataset)

        self.idx2agg = {0:'mean', 1:'variance', 2:'distribution'}
        self.agg2idx = {'mean':np.array([1,0,0]), 'variance':np.array([0,1,0]), 'distribution':np.array([2,0,1])}

        self.idx2method = {-1:'TRAD_BY', 0:'TRAD_BN',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',\
        5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',8:'Ɛ-Hybrid',9:'Ψ-Support'}

    def worker(self, args):
        key1 = args[0]
        key2 = args[1]

        if key1 in self.all_names_2:
            liste = []
            key = list(key1)

            for val in self.column_to_values[key2]:
                a = set( key+[val] )
                if a in self.all_names_2:
                    aa = self.all_names[self.all_names_2.index(a)]
                    liste.append(aa)
            
            name_key = self.all_names[self.all_names_2.index(key1)]

            return (name_key,key2),liste

        return None

    def __name_groups_2(self, df):
        index = df.index.names

        if (isinstance(df,int)) or (isinstance(df,float)) or (isinstance(df,str)):
            return str(df)
        
        df = df.drop(columns=['article_id','rating','cust_id'])

        #columns = [col for col in df.columns if len(df[col].unique())==1]
        columns = list(index)
        columns.sort()

        return ['_'.join(i) for i in df.reset_index()[columns].drop_duplicates().values][0]

    def __prepare_data(self, dataset):
        group_handler = GroupHandler()
        period_handler = PeriodHandler()

        df = pd.read_csv(f'{dataset}.csv')
        df.index = pd.to_datetime(pd.to_datetime(df.timestamp).dt.date)

        self.users = set(df.cust_id.unique())

        columns = df.columns
        columns = [col for col in columns if col not in ('purchase','transaction_date','timestamp')]

        self.columns_2 = [col for col in columns if col not in ('cust_id','article_id','rating')]

        self.types_columns = dict()
        self.one_hot_columns = dict()

        self.values_to_columns = dict()
        self.column_to_values = dict()

        for col in columns:
            if col in ['article_id','cust_id']:
                continue
            
            df_loc = df[col]

            typ = df_loc.dtype
            self.types_columns[col] = typ

            if col == 'genre':
                l = []
                df_loc.drop_duplicates().apply(lambda x: l.extend( x.split('|') ) )
                all_values = list(set(l))
            else:
                all_values = df_loc.unique()

            if col != 'rating':
                self.column_to_values[col] = list(all_values)

                for val in all_values:
                    self.values_to_columns[val] = col

            if typ == 'object':
                one = nn.functional.one_hot(torch.tensor(range(len(all_values))), num_classes=len(all_values))

                for i,val in enumerate(all_values):
                    self.one_hot_columns[col,val] = one[i,:].numpy()

        hierarchy_groups = []

        for i in range(1,len(self.columns_2)):
            a = list(itertools.combinations(self.columns_2, i))
            for j in a:
                split_ons = set(self.columns_2) - set(j)
                
                vals = [self.column_to_values[m] for m in set(j)]
                vals = list(itertools.product(*vals))

                for m in vals:
                    for k in split_ons:
                        hierarchy_groups.append( [set(m), k] )

        self.idx2columns = {idx:attr for idx,attr in enumerate(self.columns_2)}

        first_date = df.index.min() #Timestamp is the index

        df = df[columns]
        df_2 = period_handler.period(df, 'One-Sample', 'time', [first_date], [0,2])

        groups = None

        if __name__ == '__main__':
            groups = [ group_handler.groups(d, ['']) for i,d in enumerate(df_2) ] #Create all possible groups

        groups = [ [df for df in grp if len(df)>self.support  ] for grp in groups ]

        self.nameGrp_2_index = [ {self.__name_groups_2(df):df.reset_index() for idx,df in enumerate(grp)} for grp in groups ]

        self.all_names = list(self.nameGrp_2_index[0].keys())
        self.all_names_2 = [set(sorted(set(name.split('_')))) for name in self.all_names]

        pool = Pool()      
        res = pool.map(self.worker, hierarchy_groups)
        pool.close()

        res = [a for a in res if a is not None]

        self.hierarchy_groups = dict()

        for a in res:
            self.hierarchy_groups[ a[0] ] = a[1]

        del res

        self.all_names_2 = list(self.all_names)

        self.all_names_2 = [name for name in self.all_names_2 if len(name.split('_'))==1]
        self.all_names_2 = sorted(self.all_names_2, key=lambda x:len(self.nameGrp_2_index[0][x].cust_id.unique()), reverse=True)

        #self.start_cases = []

        #for i in range(1,3):
        #    a = list(itertools.combinations(self.all_names_2, i))
        #    self.start_cases.extend(a)
    
    def __data2state(self, group_state):
        state = np.array([])

        for grp in group_state:
            grp = grp.reset_index()
            for col,typ in self.types_columns.items():

                if typ == 'object':
                    val_counts = grp[[col]].value_counts()
                    rep = self.one_hot_columns[ col,val_counts.index[0][0] ] * val_counts[0]

                    for i in val_counts.index[1:]:
                        rep += self.one_hot_columns[ col,i[0] ] * val_counts[i]

                else:
                    rep = grp[[col]].describe().to_numpy().reshape(1,8)
                    rep = rep[0]

                state = np.concatenate( (state,rep) )

        return state

    def __get_mask(self, group_state_names, filters_attributes):
        mask_action_2 = [set(filters_attributes)-set([self.values_to_columns[val] for val in grp]) for grp in group_state_names]
        mask_action = []

        for i in range(self.num_groups_per_step*self.num_attr):
            grp_idx = i//self.num_attr

            if grp_idx < len(group_state_names):
                idx = i%self.num_attr

                if self.idx2columns[idx] in mask_action_2[grp_idx]:
                    mask_action.append(0)
                else:
                    mask_action.append(-np.Inf)
            else:
                mask_action.append(-np.Inf)

        return mask_action

    def __get_mask_2(self, group_state_names):
        mask_action_2 = [set(self.columns_2)-set([self.values_to_columns[val] for val in grp]) for grp in group_state_names]
        mask_action = []

        for i in range(self.num_groups_per_step*self.num_attr):
            grp_idx = i//self.num_attr

            if grp_idx < len(group_state_names):
                idx = i%self.num_attr

                if self.idx2columns[idx] in mask_action_2[grp_idx]:
                    mask_action.append(0)
                else:
                    mask_action.append(-np.Inf)
            else:
                mask_action.append(-np.Inf)

        return mask_action

    def __name_to_attributes(self, name):
        l = [self.values_to_columns[i] for i in name.split('_')]
        return '_'.join(l)

    def get_results(self, policy_name, prev_selected=None, list_regions=None, agg_function=None, filters_functions=None, filters_attributes=None):
        policy = self.policies[policy_name]
        policy_expl = self.expl_policies[policy_name]

        done = True
        first_ite = False

        #Apply a mask of Agg

        if filters_functions is None:
            filters_functions = ['mean','variance','distribution']

        if filters_attributes is None:
            filters_attributes = self.columns_2
        
        mask_agg = [0 if self.idx2agg[i] in filters_functions else -np.Inf for i in range(self.num_agg)]

        if list_regions is None:
            shuffled_cases = list(self.all_names_2)
            random.shuffle(shuffled_cases)
            group_state_names = shuffled_cases[:1]
        else:
            if isinstance(list_regions,str):
                group_state_names = [list_regions]
            else:
                group_state_names = list_regions

        if prev_selected is None:
            first_ite = True
            prev_selected = group_state_names[0]

            names_to_dict = set(sorted(set(group_state_names[0].split('_'))))
            names_to_dict = set([self.values_to_columns[val] for val in names_to_dict])
        else:
            names_to_dict = set(sorted(set(prev_selected.split('_'))))
            names_to_dict = set([self.values_to_columns[val] for val in names_to_dict])

        group_state = [self.nameGrp_2_index[0][i] for i in group_state_names]

        remaining = self.num_groups_per_step - len(group_state)
        zeros = np.zeros(remaining*self.group_vector_size)

        state_groups = self.__data2state(group_state)
        state_groups = np.concatenate( (state_groups,zeros) )

        if agg_function is None:
            agg_function = random.choice(filters_functions)
        
        state_hypo = self.agg2idx[agg_function]

        state = np.concatenate( (state_groups,state_hypo) )
        state = torch.tensor(state).double().view(1,-1)

        action_explore_exploit = policy_expl(state)
        action_explore_exploit = action_explore_exploit.argmax(1).view(-1,1)#torch.distributions.Categorical(logits=action_explore_exploit).sample().view(-1,1)
        action_explore_exploit = action_explore_exploit[0]

        if action_explore_exploit[0]==0 and first_ite == False:
            #Explore
            group_state_names = [name for name in self.all_names_2 if self.values_to_columns[name] not in names_to_dict]
            random.shuffle(group_state_names)
            group_state_names = group_state_names[:self.num_groups_per_step]

            group_state = [self.nameGrp_2_index[0][i] for i in group_state_names]
            state_2 = self.__data2state(group_state)

            state_2 = np.concatenate( (state_2,state_hypo) )
            state_2 = torch.tensor(state_2).double().view(1,-1)

            explore = True

        else:
            #Exploit
            state_2 = state.clone().detach()
            explore = False
        
        names_to_dict = [set(sorted(set(name.split('_')))) for name in group_state_names]
        mask = self.__get_mask(names_to_dict, filters_attributes)

        action_group, action_agg = policy(state_2)

        mask = torch.Tensor(mask).float().view(action_group.size())
        action_group = torch.add(action_group, mask)
        action_group = action_group.argmax(1).view(-1,1)
        #action_group = torch.distributions.Categorical(logits=action_group).sample().view(-1,1)

        mask_agg = torch.Tensor(mask_agg).float().view(action_agg.size())
        action_agg = torch.add(action_agg, mask_agg)
        action_agg = action_agg.argmax(1).view(-1,1)
        #action_agg = torch.distributions.Categorical(logits=action_agg).sample().view(-1,1)
        
        action_group = action_group[0]
        action_agg = action_agg[0][0].item()

        input_data_region = action_group[0].item()//self.num_attr
        split_attribute = action_group[0].item()%self.num_attr

        selected_group = group_state[input_data_region]
        selected_group_name = group_state_names[input_data_region]

        split_attribute = self.idx2columns[split_attribute]

        agg_type = self.idx2agg[ action_agg ]
        new_state_hypo = self.agg2idx[ agg_type ]

        test_arg = ['One-Sample']

        if action_agg < 2:
            test_arg.append(2.5)
        else:
            test_arg.append('uniform')

        if len(selected_group_name.split('_'))+1 == len(self.columns_2):
            done = False

        top_n = [self.num_groups_per_step]
        num_hyps = [1]
        approaches = [-1,0] #Alpha investing
        alpha = 0.05
        dimension = 'rating'
        
        users = set(selected_group.cust_id.unique())
        
        stats, results, names = test_groups([selected_group],[selected_group_name], split_attribute, None, self.nameGrp_2_index, self.hierarchy_groups, dimension,\
        top_n, num_hyps, approaches, agg_type, test_arg, users, self.support, alpha, verbose=False)

        names = names[0]

        if len(results) == 0:
            group_state_names = []
            groups_results = []

            min_p_val = -1
            max_p_val = -1
            sum_p_val = -1
            cov_total = 0

            fdr = 0
            power = 0

            done = False
        else:
            groups_results = results[num_hyps[0]*100][self.num_groups_per_step][self.idx2method[approaches[0]]][0]
            ground_truth = results[num_hyps[0]*100][self.num_groups_per_step][self.idx2method[approaches[1]]][0]

            p_values = groups_results['p-value'].values

            groups_results = groups_results.group1.unique()
            groups_results = list(groups_results)

            ground_truth = ground_truth.group1.unique()
            ground_truth = list(ground_truth)

            if len(ground_truth) != 0:
                power = len( set(groups_results).intersection(set(ground_truth)) )/len( set(ground_truth) )
            else:
                power = 0

            group_state_names = groups_results
        
            if len(group_state_names) == self.num_groups_per_step:
                stats = stats[num_hyps[0]*100][self.num_groups_per_step][0]

                min_p_val = stats.Min_p_value_BY.values[0]
                max_p_val = stats.Max_p_value_BY.values[0]
                sum_p_val = stats.Sum_p_value_BY.values[0]
                cov_total = stats.Cov_total_BY.values[0]

                fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

            elif len(group_state_names) != 0:
                stats = stats[num_hyps[0]*100][self.num_groups_per_step][0]

                min_p_val = stats.Min_p_value_BY.values[0]
                max_p_val = stats.Max_p_value_BY.values[0]
                sum_p_val = stats.Sum_p_value_BY.values[0]
                cov_total = stats.Cov_total_BY.values[0]

                fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

            else:

                min_p_val = -1
                max_p_val = -1
                sum_p_val = -1
                cov_total = 0
                
                fdr = 0

                done = False
        
        attribute_data_region = self.__name_to_attributes(selected_group_name)
        users_data_regions = set(selected_group.cust_id.unique())
        hypotheses = test_arg+[agg_type]
        sizes = [len(self.nameGrp_2_index[0][i]) for i in group_state_names]

        if explore:
            explor_exploit = 'Explore'
        else:
            explor_exploit = 'Exploit'

        attribute_set_output_data_regions = [self.__name_to_attributes(i) for i in group_state_names]
        users_set_output_data_regions = [set(self.nameGrp_2_index[0][i].cust_id.unique()) for i in group_state_names]

        dic={'power':[power],
        'fdr':[fdr],
        'max_pval':[max_p_val],
        'min_pval':[min_p_val],
        'sum_pval':[sum_p_val],
        'coverage':[cov_total],
        'firs_data_region':[prev_selected],
        'input_data_region':[selected_group_name],
        'attributes_combination_input_data_region':[attribute_data_region],
        'cust_ids_input_data_region':[users_data_regions],
        'hypothesis':[hypotheses],
        'action':[explor_exploit],
        'size_output_set':[len(group_state_names)],
        'output_data_regions':[group_state_names],
        'attributes_combination_output_data_regions':[attribute_set_output_data_regions],
        'cust_ids_output_data_regions':[users_set_output_data_regions],
        'size_ouptput_data_regions':[sizes],
        'done':[not done]}

        results = pd.DataFrame(data=dic)

        return results


policies = ['power_only','cov_only', 'fdr_only', 'cov_power']

#Call the class before starting the loop
get_regions = GetRegions(policies, 'MovieLens')

regions = []
actions = []
outputs = []
functions = []

#This loop is to simulate a fully guided

for i in range(3):
    print(f'---------- {i}\n')
    if i == 0:
        df = get_regions.get_results('power_only')
    else:
        df = get_regions.get_results('power_only', prev_selected=selected_group, list_regions=list_regions, agg_function=agg_function)

    prev_selected = df.firs_data_region.values[0]
    selected_group = df.input_data_region.values[0]
    list_regions = df.output_data_regions.values[0]
    agg_function = df.hypothesis.values[0][2]
    action = df.action.values[0]
    done = df.done.values[0]

    regions.append(selected_group)
    actions.append(action)
    outputs.append(list_regions)
    functions.append(agg_function)

    if done == False:
        break


print(f'Start region: {regions[0]}')
print(f'Exploit using {functions[0]}')
print(f'Having outputs: {outputs[0]}')
print()

for i in range(len(actions[1:])):
    print(f'{actions[1+i]} based on {regions[1+i]} using {functions[1+i]}')
    print(f'Having outputs: {outputs[1+i]}')
    print()
