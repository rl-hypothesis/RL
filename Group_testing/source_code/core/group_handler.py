import pandas as pd
import subprocess
import itertools
from config import ROOT_DIR
from multiprocessing import Pool
import os 

class GroupHandler:
    def __items_based_group(self,df):
        cust2items = df[['cust_id','article_id']].groupby('cust_id')['article_id'].apply(lambda x : " ".join(str(i) for i in x)).to_dict()

        outfile = open(str(os.path.join(ROOT_DIR, "source_code/notebooks/pmr.txt")), "w")

        index2id = dict()
        i = 0

        for cust,items in cust2items.items():
            outfile.write(items+"\n")
            index2id[i]=cust
            i = i+1

        outfile.close()
        
        #subprocess_arguments = ["gcc","-o",str(os.path.join(ROOT_DIR, "source_code/notebooks/lcm53/lcm")),
        #str(os.path.join(ROOT_DIR, "source_code/notebooks/lcm53/lcm.c"))]
        
        #subprocess.call(subprocess_arguments, shell=False)

        subprocess_arguments = [str(os.path.join(ROOT_DIR, "source_code/notebooks/lcm53/lcm")),"CfI","-l","10",
        str(os.path.join(ROOT_DIR, "source_code/notebooks/pmr.txt")),"10","out.txt"]

        subprocess.call(subprocess_arguments, shell=False)

        outfile = open("out.txt", "r")

        return [ [index2id[int(us)] for us in line.strip().split()] for i,line in enumerate(outfile) if i%2==1]

        outfile.close()

    def __year_based_group(self, df,clus_value):
        return df[df.year==clus_value]

    def __occupation_based_group(self, df,clus_value):
        return df[df.occupation==clus_value]

    def __runtime_based_group(self, df,clus_value):
        return df[df.runtimeMinutes==clus_value]

    def __age_based_group(self, df,clus_value):
        return df[df.age==clus_value]

    def __genre_based_group(self, df,clus_value):
        k = df[['article_id','genre']].drop_duplicates().apply(lambda x : [ [list(x)[0],i] for i in list(x)[1].split('|')] ,axis=1)
        l = []
        k = [ l.extend(i) for i in list(k)]

        df2 = pd.DataFrame(l,columns=['article_id','genre'])
        df = df.drop(columns=['genre'])
        df = pd.merge(df,df2,on='article_id')
        return df[df.genre==clus_value]

    def __gender_based_group(self, df,clus_value):
        return df[df.gender==clus_value]
    
    def __location_based_group(self, df,clus_value):
        return df[df.location==clus_value]

    def pre_group(self, df, clus_types, clus_values):

        df_local = df

        for clus_type in clus_types:
            
            if clus_type != '':

                clus_value = clus_values[ clus_types.index(clus_type) ]

                if clus_value != '':
                    if clus_type == 'items':
                        df_local = self.__items_based_group(df_local)
                    elif clus_type == 'genre':
                        df_local = self.__genre_based_group(df_local,clus_value)
                    elif clus_type == 'age':
                        df_local = self.__age_based_group(df_local,clus_value)
                    elif clus_type == 'gender':
                        df_local = self.__gender_based_group(df_local,clus_value)
                    elif clus_type == 'location':
                        df_local = self.__location_based_group(df_local,clus_value)
                    elif clus_type == 'year':
                        df_local = self.__year_based_group(df_local,clus_value)
                    elif clus_type == 'occupation':
                        df_local = self.__occupation_based_group(df_local,clus_value)
                    elif clus_type == 'runtimeMinutes':
                        df_local = self.__runtime_based_group(df_local,clus_value)
                    else:
                        print(f'{clus_type} based group is not implemented yet.')
            
        return df_local
        
    def groups(self, df, clus_type):
        #clus_type = clus_type[0]

        if ('genre' not in clus_type) and ('genre' in df.columns):
            k = df[['article_id','genre']].drop_duplicates().apply(lambda x : [ [list(x)[0],i] for i in list(x)[1].split('|')] ,axis=1)
            l = []
            k = [ l.extend(i) for i in list(k)]

            df2 = pd.DataFrame(l,columns=['article_id','genre'])

            df = df.drop(columns=['genre'])
            df = pd.merge(df,df2,on='article_id')

        if '' in clus_type:
            clus_type = [i for i in clus_type if i != '']
            
        #columns = clus_type #+ ['cust_index', 'article_index']
        #df = df.drop(columns=columns)

        columns = list(df.columns)
        columns = [e for e in columns if e not in ('cust_id', 'article_id','rating','purchase','transaction_date')]
        columns = [e for e in columns if e not in clus_type]

        len_col = len(columns)
        gb = []

        for i in range(1,len_col+1):
            gb.extend([k for k in itertools.combinations(columns,i)])

        gb = [ [df.groupby(list(i)),i] for i in gb ]

        pool = Pool()
        res = pool.map(get_groups, gb)
        pool.close()

        res = list(itertools.chain(*res))
        return res

def get_groups(g):
    return [ g[0].get_group(x).set_index(list(g[1])) for x in g[0].groups]

