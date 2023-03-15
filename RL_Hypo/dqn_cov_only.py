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


Transition = namedtuple( 'Transition', ('state_explore_exploit', 'state', 'action_explore_exploit', 'action_group', 'action_agg', 'reward', 'next_state', 'done') )

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append( Transition(*args) )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = deque([],maxlen=self.capacity)

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


def select_action(state, policy_agent, mask, epsilon_start, num_groups_per_step, num_attr, num_agg, epsilon_end, epsilon_decay, steps_done, steps_add=True):

    sample = random.random()

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
    math.exp(- steps_done * epsilon_decay)

    if steps_add:
        steps_done += 1

    if sample > epsilon:
        with torch.no_grad():

            action_group, action_agg = policy_agent(state)
            mask = torch.Tensor(mask).float().view(action_group.size())
            a = torch.add(action_group, mask)

            action_grp = torch.distributions.Categorical(logits=a).sample().view(-1,1)
            action_fun = torch.distributions.Categorical(logits=action_agg).sample().view(-1,1)

            return steps_done, epsilon, action_grp, action_fun
    else:
        mask = [1 if i == 0 else 0 for i in mask]
        sum_ = sum(mask)
        mask = [i/sum_ for i in mask]

        action_grp = [np.random.choice(np.arange(num_groups_per_step*num_attr), p=mask)]
        action_fun = [np.random.choice(np.arange(num_agg), p=[1/3,1/3,1/3])]

        return steps_done, epsilon, torch.tensor(action_grp).long().view(-1,1), torch.tensor(action_fun).long().view(-1,1)

def select_explore_exploit(state, policy_agent, epsilon_start, epsilon_end, epsilon_decay, steps_done, do_exploit = None, steps_add=True):

    sample = random.random()

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
    math.exp(- steps_done * epsilon_decay)
    
    if sample > epsilon:
        with torch.no_grad():

            action_explore_exploit = policy_agent(state)
            
            if do_exploit == True:
                action_explore_exploit[:,0] = torch.tensor(-np.Inf)
            
            return torch.distributions.Categorical(logits=action_explore_exploit).sample().view(-1,1)
            #return action_explore_exploit.argmax(1).view(-1,1)
    else:
        
        if do_exploit == True:
            a = [np.random.choice(np.arange(2), p=[0, 1])]
        else:
            a = [np.random.choice(np.arange(2), p=[0.5, 0.5])]

        return torch.tensor(a).long().view(-1,1)


def data2state(group_state, types_columns, one_hot_columns):
    state = np.array([])

    for grp in group_state:
        grp = grp.reset_index()
        for col,typ in types_columns.items():

            if typ == 'object':
                val_counts = grp[[col]].value_counts()
                rep = one_hot_columns[ col,val_counts.index[0][0] ] * val_counts[0]

                for i in val_counts.index[1:]:
                    rep += one_hot_columns[ col,i[0] ] * val_counts[i]

            else:
                rep = grp[[col]].describe().to_numpy().reshape(1,8)
                rep = rep[0]

            state = np.concatenate( (state,rep) )

    return state

def reward_p_value(p_value):
    if p_value > 0.05:
        return 0
    else:
        return 1-(p_value/0.05)

def name_to_attributes(name, values2attribute):
    l = [values2attribute[i] for i in name.split('_')]
    return '_'.join(l)

def get_mask(group_state_names, columns_2, values_to_columns, idx2columns, num_groups_per_step, num_attr):
    mask_action_2 = [set(columns_2)-set([values_to_columns[val] for val in grp]) for grp in group_state_names]
    mask_action = []

    for i in range(num_groups_per_step*num_attr):
        grp_idx = i//num_attr

        if grp_idx < len(group_state_names):
            idx = i%num_attr

            if idx2columns[idx] in mask_action_2[grp_idx]:
                mask_action.append(0)
            else:
                mask_action.append(-np.Inf)
        else:
            mask_action.append(-np.Inf)

    return mask_action

def name_groups_2(df):
    index = df.index.names

    if (isinstance(df,int)) or (isinstance(df,float)) or (isinstance(df,str)):
        return str(df)
    
    df = df.drop(columns=['article_id','rating','cust_id'])

    #columns = [col for col in df.columns if len(df[col].unique())==1]
    columns = list(index)
    columns.sort()

    return ['_'.join(i) for i in df.reset_index()[columns].drop_duplicates().values][0]


idx2method = {-1:'TRAD_BY', 0:'TRAD_BN',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',\
5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',8:'Ɛ-Hybrid',9:'Ψ-Support', 10:'Min_VAL', 11:'SMT_cov', 12:'SMT'}

num_groups_per_step = 4
num_agg = 3

batch_size = 64
group_vector_size = 69
state_size = num_groups_per_step*group_vector_size+num_agg #69 is the vector of each groupe vector, 2 is the vector of hypothesis
hidden_size = 128

epiodes = 2

target_update = 10
gamma = 0.99
learning_rate = 0.0003
steps_done = 0
epsilon_start, epsilon_end, epsilon_decay = 0.9, 0.01, 0.01

support = 5

alpha = 0.5
beta = 0.5

powers = []
fdrs = []
rewards = []
max_pvalues = []
min_pvalues = []
sum_pvalues = []
covs = []
epsilons = []
episodes = []
episodes_un = []
data_regions = []
attribute_data_region = []
attribute_set_output_data_regions = []
regions = []
users_data_regions = []
users_set_output_data_regions = []
sizes = []
sizes_data_region = []
hypotheses = []
explor_exploit = []

losses1 = []
losses2 = []
losses3 = []

q_values1 = []
q_values2 = []
q_values3 = []

replay_memory = ReplayMemory(batch_size*2)

#Data
df = pd.read_csv('data_df.csv')
df.index = pd.to_datetime(pd.to_datetime(df.timestamp).dt.date)

columns = df.columns
columns = [col for col in columns if col not in ('purchase','transaction_date','timestamp')]

columns_2 = [col for col in columns if col not in ('cust_id','article_id','rating')]
num_attr = len(columns_2)

#Models

policy_net = Agent(state_size, hidden_size, num_groups_per_step, num_attr, num_agg)
target_net = Agent(state_size, hidden_size, num_groups_per_step, num_attr, num_agg)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

policy_net_explore_exploit = Agent_explore_exploit(state_size, hidden_size)
target_net_explore_exploit = Agent_explore_exploit(state_size, hidden_size)

target_net_explore_exploit.load_state_dict(policy_net_explore_exploit.state_dict())
target_net_explore_exploit.eval()

optimizer3 = torch.optim.Adam(policy_net_explore_exploit.parameters(), lr=learning_rate)

dataset='MovieLens'

group_handler = GroupHandler()
period_handler = PeriodHandler()

#df, feature2values = get_data(dataset=dataset)
#df.to_csv('data_df.csv')

#Get here the one hot encoding of each attribute-value
types_columns = dict()
one_hot_columns = dict()

values_to_columns = dict()
column_to_values = dict()

for col in columns:
    if col in ['article_id','cust_id']:
        continue
    
    df_loc = df[col]

    typ = df_loc.dtype
    types_columns[col] = typ

    if col == 'genre':
        l = []
        df_loc.drop_duplicates().apply(lambda x: l.extend( x.split('|') ) )
        all_values = list(set(l))
    else:
        all_values = df_loc.unique()

    if col != 'rating':
        column_to_values[col] = list(all_values)

        for val in all_values:
            values_to_columns[val] = col

    if typ == 'object':
        one = nn.functional.one_hot(torch.tensor(range(len(all_values))), num_classes=len(all_values))

        for i,val in enumerate(all_values):
            one_hot_columns[col,val] = one[i,:].numpy()

hierarchy_groups = []

for i in range(1,len(columns_2)):
    a = list(itertools.combinations(columns_2, i))
    for j in a:
        split_ons = set(columns_2) - set(j)
        
        vals = [column_to_values[m] for m in set(j)]
        vals = list(itertools.product(*vals))

        for m in vals:
            for k in split_ons:
                hierarchy_groups.append( [set(m), k] )

idx2columns = {idx:attr for idx,attr in enumerate(columns_2)}

first_date = df.index.min() #Timestamp is the index

df = df[columns]
df_2 = period_handler.period(df, 'One-Sample', 'time', [first_date], [0,2])

groups = None

if __name__ == '__main__':
    groups = [ group_handler.groups(d, ['']) for i,d in enumerate(df_2) ] #Create all possible groups

#groups = [ [df.reset_index() for df in grp if len(df)>support  ] for grp in groups ]
groups = [ [df for df in grp if len(df)>support  ] for grp in groups ]

nameGrp_2_index = [ {name_groups_2(df):df.reset_index() for idx,df in enumerate(grp)} for grp in groups ]

all_names = list(nameGrp_2_index[0].keys())
all_names_2 = [set(sorted(set(name.split('_')))) for name in all_names]

def worker(args):
    key1 = args[0]
    key2 = args[1]

    if key1 in all_names_2:
        liste = []
        key = list(key1)

        for val in column_to_values[key2]:
            a = set( key+[val] )
            if a in all_names_2:
                aa = all_names[all_names_2.index(a)]
                liste.append(aa)
        
        name_key = all_names[all_names_2.index(key1)]

        return (name_key,key2),liste

    return None

pool = Pool()      
res = pool.map(worker, hierarchy_groups)
pool.close()

res = [a for a in res if a is not None]

hierarchy_groups = dict()

for a in res:
    hierarchy_groups[ a[0] ] = a[1]

del res

all_names_2 = list(all_names)

print(len(all_names_2))
print(len(all_names))

aa = 0
all_names_2 = [name for name in all_names_2 if len(name.split('_'))==1]
all_names_2 = sorted(all_names_2, key=lambda x:len(nameGrp_2_index[0][x].cust_id.unique()), reverse=True)

start_cases = []

for i in range(1,3):
    a = list(itertools.combinations(all_names_2, i))
    start_cases.extend(a)

len_start = len(start_cases)

for ep in range(epiodes):
    for j, group_state_names in enumerate(start_cases):
        group_state_names = list(group_state_names)

        done = True

        group_state = [nameGrp_2_index[0][i] for i in group_state_names]
        remaining = num_groups_per_step - len(group_state)

        random_idx = random.randint(0,2)
        if random_idx==0:
            # The agg is Mean
            state_hypo = np.array([1,0,0])
        elif random_idx==1:
            # The agg is Variance
            state_hypo = np.array([0,1,0])
        else:
            # The agg is Variance
            state_hypo = np.array([0,0,1])

        state_groups = data2state(group_state, types_columns, one_hot_columns)

        zeros = np.zeros(remaining*group_vector_size)
        state_groups = np.concatenate( (state_groups,zeros) )

        state = np.concatenate( (state_groups,state_hypo) )
        state = torch.tensor(state).double().view(1,-1)

        step = 0

        while done :
            print(f'Episode : {ep*len_start+j} - {step}')

            avg_nb_attr = [len(name.split('_')) for name in group_state_names]
            avg_nb_attr = sum(avg_nb_attr)/len(avg_nb_attr)

            if step == 0 or avg_nb_attr < 3:
                do_exploit = True
            else:
                do_exploit = None

            explore_exploit_action = select_explore_exploit(state, policy_net_explore_exploit, epsilon_start, epsilon_end, epsilon_decay, steps_done, do_exploit=do_exploit)
            explore_exploit_action = explore_exploit_action[0]

            if explore_exploit_action[0]==0:
                #Explore
                names_to_dict = set(sorted(set(selected_group_name.split('_'))))
                names_to_dict = set([values_to_columns[val] for val in names_to_dict])
                group_state_names = [name for name in all_names_2 if values_to_columns[name] not in names_to_dict]
                group_state_names = group_state_names[:num_groups_per_step]

                group_state = [nameGrp_2_index[0][i] for i in group_state_names]
                state_2 = data2state(group_state, types_columns, one_hot_columns)

                state_2 = np.concatenate( (state_2,state_hypo) )
                state_2 = torch.tensor(state_2).double().view(1,-1)

                explore = True

            else:
                #Exploit
                state_2 = state.clone().detach()
                explore = False
        
            names_to_dict = [set(sorted(set(name.split('_')))) for name in group_state_names]
            mask = get_mask(names_to_dict, columns_2, values_to_columns, idx2columns, num_groups_per_step, num_attr)

            steps_done, epsilon, group_action, agg_action = select_action(state_2, policy_net, mask, epsilon_start, num_groups_per_step, num_attr, num_agg, epsilon_end, epsilon_decay, steps_done)

            group_action = group_action[0]

            input_data_region = group_action[0].item()//num_attr
            split_attribute = group_action[0].item()%num_attr

            selected_group = group_state[input_data_region]
            selected_group_name = group_state_names[input_data_region]

            split_attribute = idx2columns[split_attribute]

            if agg_action[0][0] == 0:
                agg_type = 'mean'
                new_state_hypo = np.array([1,0,0])
                test_arg = ['One-Sample', 2.5]
            elif agg_action[0][0] == 1:
                agg_type = 'variance'
                new_state_hypo = np.array([0,1,0])
                test_arg = ['One-Sample', 2.5]
            else:
                agg_type = 'distribution'
                new_state_hypo = np.array([0,0,1])
                test_arg = ['One-Sample', 'uniform']
            

            if len(selected_group_name.split('_'))+1 == len(columns_2):
                done = False

            top_n = [num_groups_per_step]
            num_hyps = [1]
            approaches = [-1,0] #Alpha investing
            alpha = 0.05
            dimension = 'rating'

            theName = selected_group_name
            users = set(selected_group.cust_id.unique())

            stats, results, names = test_groups([selected_group],[selected_group_name], split_attribute, None, nameGrp_2_index, hierarchy_groups, dimension,\
            top_n, num_hyps, approaches, agg_type, test_arg, users, support, alpha, verbose=False)

            names = names[0]

            if len(results) == 0:
                zeros = np.zeros(num_groups_per_step*group_vector_size)
                new_state = np.concatenate( (zeros,new_state_hypo) )
                new_state = torch.tensor(new_state).double().view(1,-1)

                group_state = []
                group_state_names = []
                groups_results = []

                reward_1 = 0

                min_p_val = -1
                max_p_val = -1
                sum_p_val = -1
                cov_total = 0

                fdr = 0
                power = 0

                done = False
            else:
                groups_results = results[num_hyps[0]*100][num_groups_per_step][idx2method[approaches[0]]][0]
                ground_truth = results[num_hyps[0]*100][num_groups_per_step][idx2method[approaches[1]]][0]

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
                group_state = [names[gr] for gr in group_state_names]
            
                if len(group_state) == num_groups_per_step:
                    new_state = data2state(group_state, types_columns, one_hot_columns)
                    new_state = np.concatenate( (new_state,new_state_hypo) )
                    new_state = torch.tensor(new_state).double().view(1,-1)

                    stats = stats[num_hyps[0]*100][num_groups_per_step][0]

                    min_p_val = stats.Min_p_value_BY.values[0]
                    max_p_val = stats.Max_p_value_BY.values[0]
                    sum_p_val = stats.Sum_p_value_BY.values[0]
                    cov_total = stats.Cov_total_BY.values[0]

                    reward_1 = cov_total#reward_p_value(sum_p_val/num_groups_per_step)

                    fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

                elif len(group_state) != 0:
                    remaining = num_groups_per_step - len(group_state_names)
                    new_state = data2state(group_state, types_columns, one_hot_columns)

                    zeros = np.zeros(remaining*group_vector_size)
                    new_state = np.concatenate( (new_state,zeros) )
                    new_state = np.concatenate( (new_state,new_state_hypo) )
                    new_state = torch.tensor(new_state).double().view(1,-1)

                    stats = stats[num_hyps[0]*100][num_groups_per_step][0]

                    min_p_val = stats.Min_p_value_BY.values[0]
                    max_p_val = stats.Max_p_value_BY.values[0]
                    sum_p_val = stats.Sum_p_value_BY.values[0]
                    cov_total = stats.Cov_total_BY.values[0]

                    reward_1 = cov_total#reward_p_value(sum_p_val/len(group_state_names))

                    fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

                else:
                    zeros = np.zeros(num_groups_per_step*group_vector_size)
                    new_state = np.concatenate( (zeros,new_state_hypo) )
                    new_state = torch.tensor(new_state).double().view(1,-1)

                    reward_1 = 0

                    min_p_val = -1
                    max_p_val = -1
                    sum_p_val = -1
                    cov_total = 0
                    
                    fdr = 0

                    done = False
        
            rwd = reward_1
            reward = torch.Tensor([rwd]).long()

            rewards.append(rwd)
            powers.append(power)
            fdrs.append(fdr)
            max_pvalues.append(max_p_val)
            min_pvalues.append(min_p_val)
            sum_pvalues.append(sum_p_val)
            covs.append(cov_total)

            episodes.append(len_start*ep+j) #Change HERE AFTER
            episodes_un.append(step)

            if explore:
                explor_exploit.append('Explore')
            else:
                explor_exploit.append('Exploit')
            
            data_regions.append(theName)
            attribute_data_region.append( name_to_attributes(theName, values_to_columns) )

            sizes.append( len(group_state_names) )
            sizes_data_region.append( [len(i) for i in group_state] )

            regions.append( group_state_names )
            attribute_set_output_data_regions.append( [name_to_attributes(i, values_to_columns) for i in group_state_names] )

            users_data_regions.append( set(selected_group.cust_id.unique()) )
            users_set_output_data_regions.append( [set(i.cust_id.unique()) for i in group_state] )

            hypotheses.append( test_arg+[agg_type] )

            step += 1

            replay_memory.push(state, state_2, explore_exploit_action, group_action, agg_action[0], reward, new_state, torch.Tensor([done]) )

            if len(replay_memory) >= batch_size:
                aa += 1

                state = new_state.clone().detach()

                sample = replay_memory.sample(batch_size)
                batch = Transition(*zip(*sample))

                state_batch = torch.cat(batch.state).view(batch_size,-1) #State_2
                next_state_batch = torch.cat(batch.next_state).view(batch_size,-1)
                
                explore_exploit_state_batch = torch.cat(batch.state_explore_exploit).view(batch_size, -1) #State

                action_group_batch = torch.cat(batch.action_group).view(batch_size,-1)
                action_agg_batch = torch.cat(batch.action_agg).view(batch_size,-1)
                action_explore_exploit_batch = torch.cat(batch.action_explore_exploit).view(batch_size, -1)

                reward_batch = torch.cat(batch.reward).view(-1)

                done_batch = torch.cat(batch.done).view(-1)
                done_batch = (done_batch == 1)

                ######
                state_action_group_values, state_action_agg_values = policy_net(state_batch)

                state_action_group_values = torch.gather( state_action_group_values, 1, action_group_batch ).view(batch_size,-1)
                state_action_agg_values = torch.gather( state_action_agg_values, 1, action_agg_batch ).view(batch_size,-1)

                next_state_action_group_values, next_state_action_agg_values = target_net(next_state_batch)

                next_state_action_group_values = next_state_action_group_values.max(1)[0].detach().clone().view(-1)
                next_state_action_agg_values = next_state_action_agg_values.max(1)[0].detach().clone().view(-1)

                ## To Update Agent of choosing the data region
                next_state_action_values = torch.zeros(batch_size).double()
                next_state_action_values[done_batch] = next_state_action_group_values[done_batch]
                expected_state_action_group_values = (next_state_action_values * gamma) + reward_batch

                ## To Update Agent of choosing the aggregation func
                next_state_action_values = torch.zeros(batch_size).double()
                next_state_action_values[done_batch] = next_state_action_agg_values[done_batch]
                expected_state_action_agg_values = (next_state_action_values * gamma) + reward_batch

                q_values1.append(state_action_group_values.detach().mean())
                q_values2.append(state_action_agg_values.detach().mean())
                
                #### To Update Agent of choosing the big actions
                state_action_explo_values = policy_net_explore_exploit(explore_exploit_state_batch)
                state_action_explo_values = torch.gather( state_action_explo_values, 1, action_explore_exploit_batch ).view(batch_size,-1)

                #next_state_action_explo_values = target_net_explore_exploit(next_state_batch)
                next_state_action_explo_values = target_net_explore_exploit(state_batch)
                next_state_action_explo_values = next_state_action_explo_values.max(1)[0].detach().clone().view(-1)

                next_state_action_values = torch.zeros(batch_size).double()
                next_state_action_values[done_batch] = next_state_action_explo_values[done_batch]
                expected_state_action_explo_values = (state_action_explo_values * gamma) + reward_batch

                q_values3.append(state_action_explo_values.detach().mean())

                criterion = nn.SmoothL1Loss()
                criterion2 = nn.SmoothL1Loss()
                criterion3 = nn.SmoothL1Loss()

                loss1 = criterion(state_action_group_values, expected_state_action_group_values)
                loss2 = criterion2(state_action_agg_values, expected_state_action_agg_values)
                loss3 = criterion2(state_action_explo_values, expected_state_action_explo_values)

                losses1.append(loss1.item())
                losses2.append(loss2.item())
                losses3.append(loss3.item())

                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                loss1.backward()
                loss2.backward()
                loss3.backward()

                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)

                for param in policy_net_explore_exploit.parameters():
                    param.grad.data.clamp_(-1, 1)

                optimizer.step()
                optimizer2.step()
                optimizer3.step()
            
            state = new_state.clone().detach()

            if (aa % target_update == 0 and aa != 0) or (step % target_update == 0 ):
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(policy_net.state_dict(), f'policy_net_cov_only.pth')

                target_net_explore_exploit.load_state_dict(policy_net_explore_exploit.state_dict())
                torch.save(policy_net_explore_exploit.state_dict(), f'policy_net_explore_exploit_cov_only.pth')

    ep += 1

target_net.load_state_dict(policy_net.state_dict())
torch.save(policy_net.state_dict(), f'policy_net_cov_only.pth')

target_net_explore_exploit.load_state_dict(policy_net_explore_exploit.state_dict())
torch.save(policy_net_explore_exploit.state_dict(), f'policy_net_explore_exploit_cov_only.pth')


dic={'power':powers,
'fdr':fdrs,
'reward':rewards,
'max_pval':max_pvalues,
'min_pval':min_pvalues,
'sum_pval':sum_pvalues,
'coverage':covs,
'episodes':episodes,
'steps_in_episode':episodes_un,
'input_data_region':data_regions,
'attributes_combination_input_data_region':attribute_data_region,
'cust_ids_input_data_region':users_data_regions,
'hypothesis':hypotheses,
'action':explor_exploit,
'size_output_set':sizes,
'output_data_regions':regions,
'attributes_combination_output_data_regions':attribute_set_output_data_regions,
'cust_ids_output_data_regions':users_set_output_data_regions,
'size_ouptput_data_regions':sizes_data_region}

results=pd.DataFrame(data=dic)
results.to_csv('iteration_results_cov_only.csv',index=False)

dic={'loss1':losses1,
'loss2':losses2,
'loss3':losses3,
'qvalues1':q_values1,
'qvalues2':q_values2,
'qvalues3':q_values3}

results=pd.DataFrame(data=dic)
results.to_csv('losses_cov_only.csv',index=False)