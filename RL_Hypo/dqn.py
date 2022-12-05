import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool
import random
import math
from collections import deque, namedtuple
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
    def __init__(self, state_size, hidden_size, num_groups_per_step, 
    num_agg):

        super(Agent, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.num_groups_per_step = num_groups_per_step
        self.num_agg = num_agg

        self.fc_1 = nn.Linear(state_size, hidden_size).double()
        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, num_groups_per_step).double()
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        
        
        self.fc_agg = nn.Linear(state_size, hidden_size).double()
        torch.nn.init.xavier_uniform_(self.fc_agg.weight)
        self.relu_agg = nn.ReLU()

        self.fc_agg_2 = nn.Linear(hidden_size, num_agg).double()
        torch.nn.init.xavier_uniform_(self.fc_agg_2.weight)

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
        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, 2).double()
        torch.nn.init.xavier_uniform_(self.fc_3.weight)


    def forward(self, input_state):
        out = self.fc_1(input_state.clone())
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.relu2(out_2)

        return self.fc_3(out_2)


def select_action(state, policy_agent, epsilon_start, num_groups_per_step, num_agg,
epsilon_end, epsilon_decay, steps_done, steps_add=True):

    sample = random.random()

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
    math.exp(- steps_done * epsilon_decay)

    if steps_add:
        steps_done += 1

    if sample > epsilon:
        with torch.no_grad():

            action_group, action_agg = policy_agent(state)

            return steps_done, epsilon, action_group.argmax(1).view(-1,1),\
            action_agg.argmax(1).view(-1,1)
    else:

        return steps_done, epsilon, torch.tensor([random.randrange(num_groups_per_step)]).long().view(-1,1),\
        torch.tensor([random.randrange(num_agg)]).long().view(-1,1)

def select_explore_exploit(state, policy_agent, epsilon_start, num_groups_per_step, num_agg,
epsilon_end, epsilon_decay, steps_done, steps_add=True):

    sample = random.random()

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
    math.exp(- steps_done * epsilon_decay)

    if sample > epsilon:
        with torch.no_grad():

            action_explore_exploit = policy_agent(state)

            return action_explore_exploit.argmax(1).view(-1,1)
    else:

        return torch.tensor([random.randrange(2)]).long().view(-1,1)


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

idx2method = {0:'TRAD',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',\
5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',8:'Ɛ-Hybrid',9:'Ψ-Support', 10:'Min_VAL', 11:'SMT_cov', 12:'SMT'}

num_groups_per_step = 15
num_agg = 3

batch_size = 64
group_vector_size = 69
state_size = num_groups_per_step*group_vector_size+num_agg #69 is the vector of each groupe vector, 2 is the vector of hypothesis
hidden_size = 128

epiodes = 100

target_update = 10
gamma = 0.99
learning_rate = 0.0003
steps_done = 0
epsilon_start, epsilon_end, epsilon_decay = 0.9, 0.01, 0.01

support = 20

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

losses1 = []
losses2 = []
losses3 = []

q_values1 = []
q_values2 = []
q_values3 = []

replay_memory = ReplayMemory(batch_size*2)

policy_net = Agent(state_size, hidden_size, num_groups_per_step, num_agg)
target_net = Agent(state_size, hidden_size, num_groups_per_step, num_agg)

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

df = pd.read_csv('data_df.csv')
df.index = pd.to_datetime(pd.to_datetime(df.timestamp).dt.date, )

users = set(df.cust_id.unique())

columns = df.columns
columns = [col for col in columns if col not in ('purchase','transaction_date','timestamp')]

#Get here the one hot encoding of each attribute-value
types_columns = dict()
one_hot_columns = dict()
values_to_columns = dict()

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
        for val in all_values:
            values_to_columns[val] = col

    if typ == 'object':
        one = nn.functional.one_hot(torch.tensor(range(len(all_values))), num_classes=len(all_values))

        for i,val in enumerate(all_values):
            one_hot_columns[col,val] = one[i,:].numpy()

first_date = df.index.min() #Timestamp is the index

df = df[columns]
df_2 = period_handler.period(df, 'One-Sample', 'time', [first_date], [0,2])

groups = None

big_time = time.time()
if __name__ == '__main__':
    groups = [ group_handler.groups(d, ['']) for i,d in enumerate(df_2) ] #Create all possible groups

#print(time.time()-big_time)

big_time = time.time()
groups = [ [df.reset_index() for df in grp if len(df)>support  ] for grp in groups ]
#print(time.time()-big_time)

big_time = time.time()
nameGrp_2_index = [ {name_groups(df):df for idx,df in enumerate(grp)} for grp in groups ]
#print(time.time()-big_time)

#all_names = [ [name_groups(df) for df in grp] for grp in groups ]
all_names = list(nameGrp_2_index[0].keys())
all_names = sorted(all_names, key=lambda x: len(x.split('_')))

#name2users = [len(df.cust_id.unique()) for name,df in nameGrp_2_index[0].items()]
#all_names = all_names[0]

#print(len(all_names))

def worker(args):
    name = args[0]
    i = args[1]

    liste = []
    for name_2 in all_names[:i]+all_names[i+1:]:
        if len(set(name.split('_')) - set(name_2.split('_'))) == 0:
            liste.append(name_2)
    return name,liste

big_time = time.time()
args = [(name,i) for i,name in enumerate(all_names)]

pool = Pool()      
res = pool.map(worker, args)
pool.close()

hierarchy_groups = {a[0]:a[1] for a in res}
#print(time.time()-big_time)
del res

aa = 0

all_names = [i for i in all_names if len(i.split('_'))<3]

for ep in range(epiodes):
    done = True
    #seen = set()
    all_names_2 = all_names.copy()
    #name2users_2 = name2users.copy()
    
    #random_idx = np.random.choice(all_names_2, num_groups_per_step)
    random_idx = all_names_2[:num_groups_per_step]

    #random_idx = np.random.choice(range(len(groups[0])), num_groups_per_step)
    
    #random_idx = [len(df) for df in groups[0]]
    #random_idx = np.argsort(random_idx)[::-1][:num_groups_per_step]

    group_state = [nameGrp_2_index[0][i] for i in random_idx]
    #group_state = np.array(groups[0])[random_idx]
    #group_state = [group_state]

    #nameGrp_2_index = [ {name_groups(df):df for idx,df in enumerate(grp)} for grp in group_state ]

    #group_state = group_state[0]
    state_groups = data2state(group_state, types_columns, one_hot_columns)

    #seen = seen.union(set(random_idx))

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

    state = np.concatenate( (state_groups,state_hypo) )
    state = torch.tensor(state).double().view(1,-1)

    b = 0

    while done :
        print(f'Episode : {ep} - {b}')

        explore_exploit_action = select_explore_exploit(state, policy_net_explore_exploit, epsilon_start, num_groups_per_step,\
        num_agg, epsilon_end, epsilon_decay, steps_done)

        explore_exploit_action = explore_exploit_action[0]

        if explore_exploit_action[0]==0:
            #Explore
            #possible_groups = set(all_names) - seen
            #random_idx = np.random.choice(all_names_2, num_groups_per_step)
            random_idx = all_names_2[:num_groups_per_step]

            group_state = [nameGrp_2_index[0][i] for i in random_idx]
            state_2 = data2state(group_state, types_columns, one_hot_columns)

            state_2 = np.concatenate( (state_2,state_hypo) )
            state_2 = torch.tensor(state_2).double().view(1,-1)

            #seen = seen.union(set(random_idx))
        else:
            #Exploit
            state_2 = state.clone().detach()
            #seen = seeen.union()

        steps_done, epsilon, group_action, agg_action = select_action(state_2, policy_net, epsilon_start, num_groups_per_step,\
        num_agg, epsilon_end, epsilon_decay, steps_done)
        
        group_action = group_action[0]

        if group_action[0] >= len(group_state):
            selected_group = group_state[ group_action[0]%len(group_state) ]
        else:
            selected_group = group_state[group_action[0]]
        
        grp_type = []

        for col in columns:
            if col in ['rating','article_id','cust_id']:
                continue

            if len(selected_group[col].unique()) == 1:
                grp_type.append(col)
        
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
        

        if len(columns)-3 == len(grp_type):
            done = False
        
        top_n = [num_groups_per_step]
        num_hyps = [1]
        approaches = [4,0] #Alpha investing
        alpha = 0.05
        #test_arg = ['One-Sample', 2.5]
        dimension = 'rating'

        theName = name_groups(selected_group)
        #seen = seen.union(theName)

        if theName in all_names_2:
            idx = all_names_2.index(theName)
            all_names_2.pop(idx)
        #name2users_2.pop(idx)

        stats, results, names = test_groups([selected_group], [grp_type], nameGrp_2_index, hierarchy_groups, dimension,\
         top_n, num_hyps, approaches, agg_type, test_arg, users, support, alpha, verbose=False)

        names = names[0]

        if len(results) == 0:
            zeros = np.zeros(num_groups_per_step*group_vector_size)
            new_state = np.concatenate( (zeros,new_state_hypo) )
            new_state = torch.tensor(new_state).double().view(1,-1)

            group_state = []
            groups_results = []

            reward_1 = 0
            reward_2 = 0

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

            group_state = [names[gr] for gr in groups_results]
        
            if len(group_state) == num_groups_per_step:
                new_state = data2state(group_state, types_columns, one_hot_columns)
                new_state = np.concatenate( (new_state,new_state_hypo) )
                new_state = torch.tensor(new_state).double().view(1,-1)

                stats = stats[num_hyps[0]*100][num_groups_per_step][0]

                min_p_val = stats.Min_p_value_500.values[0]
                max_p_val = stats.Max_p_value_500.values[0]
                sum_p_val = stats.Sum_p_value_500.values[0]
                cov_total = stats.Cov_total_500.values[0]

                reward_1 = reward_p_value(sum_p_val)
                reward_2 = 0#len(set(groups_results) - seen)/len(set(groups_results))

                fdr = len( set(groups_results)-set(ground_truth) )/len( set(groups_results) )

            elif len(group_state) != 0:
                remaining = num_groups_per_step - len(group_state)
                new_state = data2state(group_state, types_columns, one_hot_columns)
                zeros = np.zeros(remaining*group_vector_size)
                new_state = np.concatenate( (new_state,zeros) )
                new_state = np.concatenate( (new_state,new_state_hypo) )
                new_state = torch.tensor(new_state).double().view(1,-1)

                stats = stats[num_hyps[0]*100][num_groups_per_step][0]

                min_p_val = stats.Min_p_value_500.values[0]
                max_p_val = stats.Max_p_value_500.values[0]
                sum_p_val = stats.Sum_p_value_500.values[0]
                cov_total = stats.Cov_total_500.values[0]

                reward_1 = reward_p_value(sum_p_val)
                reward_2 = 0#len(set(groups_results) - seen)/len(set(groups_results))

                fdr = len( set(groups_results)-set(ground_truth) )/len( set(groups_results) )

                #done = False

            else:
                zeros = np.zeros(num_groups_per_step*group_vector_size)
                new_state = np.concatenate( (zeros,new_state_hypo) )
                new_state = torch.tensor(new_state).double().view(1,-1)

                group_state = []

                reward_1 = 0
                reward_2 = 0

                min_p_val = -1
                max_p_val = -1
                sum_p_val = -1
                cov_total = 0
                
                fdr = 0

                done = False
        
            #seen = seen.union(set(groups_results))
        
        #rwd = alpha*reward_1 + beta*reward_2
        rwd = reward_1
        reward = torch.Tensor([rwd]).long()

        rewards.append(rwd)
        powers.append(power)
        fdrs.append(fdr)
        max_pvalues.append(max_p_val)
        min_pvalues.append(min_p_val)
        sum_pvalues.append(sum_p_val)
        covs.append(cov_total)
        episodes.append(ep)
        episodes_un.append(b)
        
        data_regions.append(theName)
        attribute_data_region.append( name_to_attributes(theName, values_to_columns) )

        sizes.append( len(groups_results) )
        sizes_data_region.append( [len(i) for i in group_state] )
        regions.append( groups_results )
        attribute_set_output_data_regions.append( [name_to_attributes(i, values_to_columns) for i in groups_results] )

        users_data_regions.append( set(selected_group.cust_id.unique()) )
        users_set_output_data_regions.append( [set(i.cust_id.unique()) for i in group_state] )

        hypotheses.append( test_arg+[agg_type] )

        #epsilons.append(epsilon)

        b += 1

        replay_memory.push(state, state_2, explore_exploit_action, group_action, agg_action[0], reward, new_state, torch.Tensor([done]) )

        if len(replay_memory) >= batch_size:
            aa += 1

            state = new_state.clone().detach()

            sample = replay_memory.sample(batch_size)
            batch = Transition(*zip(*sample))

            state_batch = torch.cat(batch.state).view(batch_size,-1)
            next_state_batch = torch.cat(batch.next_state).view(batch_size,-1)
            
            explore_exploit_state_batch = torch.cat(batch.state_explore_exploit).view(batch_size, -1)

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

            next_state_action_explo_values = target_net_explore_exploit(next_state_batch)
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

            #for param in policy_net.parameters():
            #    param.grad.data.clamp_(-1, 1)

            optimizer.step()
            optimizer2.step()
            optimizer3.step()
        
        state = new_state.clone().detach()

        if aa % target_update == 0 and aa != 0 :
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), f'policy_net.pth')

            target_net_explore_exploit.load_state_dict(policy_net_explore_exploit.state_dict())
            torch.save(policy_net_explore_exploit.state_dict(), f'policy_net_explore_exploit.pth')
            
    ep += 1

target_net.load_state_dict(policy_net.state_dict())
torch.save(policy_net.state_dict(), f'policy_net.pth')

target_net_explore_exploit.load_state_dict(policy_net_explore_exploit.state_dict())
torch.save(policy_net_explore_exploit.state_dict(), f'policy_net_explore_exploit.pth')


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
'size_output_set':sizes,
'output_data_regions':regions,
'attributes_combination_output_data_regions':attribute_set_output_data_regions,
'cust_ids_output_data_regions':users_set_output_data_regions,
'size_ouptput_data_regions':sizes_data_region}

results=pd.DataFrame(data=dic)
results.to_csv('iteration_results.csv',index=False)

dic={'loss1':losses1,
'loss2':losses2,
'loss3':losses3,
'qvalues1':q_values1,
'qvalues2':q_values2,
'qvalues3':q_values3}

results=pd.DataFrame(data=dic)
results.to_csv('losses.csv',index=False)