import pandas as pd
import sys
from datetime import datetime

from config import ROOT_DIR
from source_code.utils.tools import missing_genres, run

columns_to_keep = ['article_id', "gender", "cust_id", "location", "age"]

occupations = { 0:"other",1:"academic-educator",2:"artist",3:"clerical-admin",4:"college-grad student",5:"customer service",6:"doctor-health care",\
7:"executive-managerial",8:"farmer",9:"homemaker",10:"K-12 student",11:"lawyer",12:"programmer",13:"retired",14:"sales-marketing",15:"scientist",\
16:"self-employed",17:"technician-engineer",18:"tradesman-craftsman",19:"unemployed",20:"writer"}

decades = {"0":"2000's","1":"2010's","2":"2020's","99":"90's", "98":"80's", "97":"70's", "96":"60's","95":"50's", "94":"40's", "93":"30's",\
 "92":"20's", "91":"10's","90":"00's", "89":"IXX","37":"XIV"}

ages_ml = {1:"<18",18:"18-24",25:"25-34",35:"35-44",45:"45-49",50:"50-55",56:">56"}

to_supp = ['somewherein space','���','&#32654;&#22269;','caribbean sea','ama lurra','polk','?�?{','n/a - on the road','ouranos','here and there','fifi',\
'n/a','heaven','lornsenland','5057chadwick ct.','far away...','strongbadia','universe','pender','quit','l','burlington','allen', 'thing',\
'\\"n/a\\','everywhere and anywhere', 'citrus.', 'x','galiza neghra','&#20013;&#22269;','hernando','']

to_change = {'spain':['espa�a','spain"','basque country', 'euskal herria', 'galiza', 'catalunya','catalunya spain','grenada','espa�?�a','catalunya(catalonia)',\
'lleida','catalonia','orense','bergued�','madrid','asturies'], 'argentina':['la argentina'], 'italy' : ['l`italia','italia','sicilia','lombardia','rosello','roma',\
'vicenza'], 'mexico' : ['monterrey','m�?�xico','m�xico'], 'dominican republic' : ['dominica'], 'france' : ['la france'], 'morocco' : ['maroc'], 'uruguay' : ['urugua'],\
'united kingdom' : ['queenspark','england','scotland','wales','alderney','jersey','guernsey','hillsborough','isle of man','channel islands','united kindgdom',\
'u.k.'], 'brazil' : ['brasil','_ brasil'], 'usa' : ['u.s. virgin islands','u.s.a.','missouri','united states','cherokee','san bernardino','america','usa & canada',\
'alachua','ysa','lawrenceville','hungary and usa','austbritania','united state','wonderful usa','aruba'], 'saudi arabia' : ['ksa'], \
'ivory cost' : ['cote d`ivoire','c�te d'], 'united arab emirates' : ['u.a.e'], 'belgium' : ['belgique','la belgique'], 'germany' : ['deutsches reich','deutschland'],\
'sweden' : ['goteborg'],'canada' : ['hamilton'], 'algeria' : ['alg�rie','l`alg�rie'], 'salvador' : ['el salvador'], 'south korea' : ['republic of korea','korea'],\
'switzerland' : ['la suisse','suisse'], 'china' : ['中�?�','p.r.china','china ���','p.r.c','china���'], 'greece' : ['greece (=hellas)'], 'peru' : ['per�'],\
'new zealand' : ['aotearoa','nz'], 'netherlands' : ['the netherlands','holland'], 'philippines' : ['phillipines'], 'swaziland' : ['swazilandia'],\
'trinidad and tobago' : ['tobago','trinidad'], 'malaysia' : ['w. malaysia'], 'madagascar' : ['le madagascar']}

def get_dataset_movie_lens(article_id):
    
    datContent = [str(i).strip().split('::') for i in open(f'{ROOT_DIR}/datasets/MovieLens1/movies.dat','rb').readlines()]
    item = pd.DataFrame(datContent,columns=['movieId','name','genres'])
    
    item.movieId = item.movieId.map(lambda x : x.strip('b'))
    item.movieId = item.movieId.apply(lambda x : int(x.split('\'')[1]) if len(x.split('\''))>1 else int(x.split('\"')[1]))
    item.genres = item.genres.map(lambda x : x.split('\\')[0])
    
    datContent = [str(i).strip().split('::') for i in open(f'{ROOT_DIR}/datasets/MovieLens1/ratings.dat','rb').readlines()]
    item_rating = pd.DataFrame(datContent,columns=['userId','itemId','rating','timestamp'])
    
    item_rating.itemId = item_rating.itemId.astype('int64')
    item_rating.rating = item_rating.rating.astype('int64')
    item_rating.userId = item_rating.userId.map(lambda x : int(x.strip('b\'')))
    item_rating.timestamp = item_rating.timestamp.map(lambda x : int(x.split('\\')[0]))
    
    item_rating.timestamp = pd.to_datetime(item_rating.timestamp,unit='s')
    
    datContent = [str(i).strip().split('::') for i in open(f'{ROOT_DIR}/datasets/MovieLens1/users.dat','rb').readlines()]

    user = pd.DataFrame(datContent,columns=['userId','Gender','Age','Occupation','Zip'])

    user.userId = user.userId.apply(lambda x: int(x.strip('b\'')))
    user.Gender = user.Gender.apply(lambda x: str(x))
    user.Age = user.Age.apply(lambda x: int(x))
    user.Occupation = user.Occupation.apply(lambda x: int(x))

    user = user[['userId','Gender','Age','Occupation']]
    user.Occupation = user.Occupation.map(lambda x : occupations[x])
    user.Age = user.Age.map(lambda x : ages_ml[x])

    item.rename(columns={'movieId':'itemId'},inplace=True)
    item_rating.rename(columns={'movieId':'itemId'},inplace=True)
    
    movies_links = pd.read_csv(f'{ROOT_DIR}/datasets/MovieLens25/links.csv') #Link between MovieLen and IMDB
    movies_links.rename(columns={'movieId':'itemId'},inplace=True)
    movies_links = movies_links[['itemId','imdbId']]
     
    imdb_movies = pd.read_csv(f'{ROOT_DIR}/datasets/IMDB/title.basics.tsv',sep='\t') #SHOWS 

    imdb_movies.tconst = imdb_movies.tconst.map(lambda x: x.strip('t'))
    imdb_movies.tconst = pd.to_numeric(imdb_movies.tconst)

    imdb_movies.rename(columns={'tconst':'imdbId'},inplace=True)
    imdb_movies = imdb_movies[['imdbId','runtimeMinutes','genres','startYear']]

    movies_merged = pd.merge(movies_links,imdb_movies,on='imdbId')

    item.rename(columns={'genres':'genres_first'},inplace=True)
    item = pd.merge(item,movies_merged,on='itemId') #Merge between MoviesLen and IMDB attr

    list_genres = list(item.genres_first.unique())
    li = item.apply(missing_genres, genres=list_genres, axis=1) #Filling the missing genres of MoviesLen by IMDB ones
    item.genres_first = li

    item = item[['itemId','genres_first','runtimeMinutes','startYear']]
    #item = item[['itemId','genres_first','startYear']]
    item.rename(columns={'genres_first':'genre','startYear':'year'},inplace=True)

    item = item[item.runtimeMinutes!='\\N']
    item.runtimeMinutes = pd.to_numeric(item.runtimeMinutes)
    item.runtimeMinutes = item.runtimeMinutes.map(lambda x : run(x/60))

    item = item[item.year!='\\N']
    item.year = pd.to_numeric(item.year)
    item.year = item.year.map(lambda x : decades[str(int(x/10)%100)] )

    item_rating = pd.merge(item_rating,user,on='userId')
    item_rating = pd.merge(item_rating,item,on='itemId')

    df = item_rating

    #df["purchase"] = 1
    df.index = pd.to_datetime(pd.to_datetime(df.timestamp, unit="s").dt.date, )

    columns = {
        "userId": "cust_id",
        "itemId": "article_id",
        "Gender": "gender",
        "Age": "age",
        "Occupation":"occupation",
        "timestamp":"transaction_date"
    }

    df.rename(columns=columns, inplace=True)

    features2values = dict()

    for feature in ['age','gender','genre','year','occupation','runtimeMinutes']:
    #for feature in ['age','gender','genre','year','occupation']:
        features2values[feature] = list(df[feature].unique())

    return df,features2values

def get_dataset_tafeng(article_id):
    df = pd.read_csv(f"{ROOT_DIR}/datasets/TAFENG_Kaggle.csv")
    df.columns = [i.lower() for i in df.columns]
    columns = {
        "age_group": "age",
        "transaction_dt": "transaction_date",
        "product_id": "article_id",
        'customer_id': "cust_id",
        'amount':'rating',
        'product_subclass' : 'genre',
        'sales_price' : 'price',
        'pin_code':'location'
    }
    df.rename(columns=columns, inplace=True)
    df.index = pd.to_datetime(pd.to_datetime(df.transaction_date).dt.date)
    df["purchase"] = 1
    df = df.drop(columns=['asset','price'])
    df = df[~df.age.isna()]
    df = df.astype({"genre": str})

    features2values = dict()
    for feature in ['age','location','genre']:
        features2values[feature] = list(df[feature].unique())

    return df,features2values

def get_dataset_yelp(article_id):
    df = pd.read_csv(f"{ROOT_DIR}/datasets/Yelp/yelp.csv")
    df.columns = [i.lower() for i in df.columns]
    columns = {
        "date": "transaction_date",
        "business_id": "article_id",
        'user_id': "cust_id",
        'stars':'rating',
        'categories' : 'genre',
        'fans' : 'popularity',
        'city':'location',
        'Gender':'gender'
    }
    df.rename(columns=columns, inplace=True)
    df.index = pd.to_datetime(pd.to_datetime(df.transaction_date).dt.date)
    df["purchase"] = 1
    df = df.drop(columns=['name','genre'])

    features2values = dict()
    for feature in ['gender','location','popularity']:
        features2values[feature] = list(df[feature].unique())

    return df,features2values

def get_dataset_books(article_id):
    users = pd.read_csv(f'{ROOT_DIR}/datasets/Book/BX-Users.csv',delimiter=';', encoding='latin-1')
    users = users[~users.Age.isna()]

    users.Age = users.Age.apply(lambda x : "<18" if x < 18 else "18-24" if x < 24 else "25-34" if x < 34 else "35-44" if x < 44 \
    else "45-49" if x < 49 else "50-55" if x < 55 else ">56")

    users.Location = users.Location.apply(lambda x: x.split(',')[-1].strip().strip('"'))
    users = users[~users.Location.isin(to_supp)]

    for k,v in to_change.items():
        for value in v:
            users = users.replace(v,k)

    books = pd.read_csv(f'{ROOT_DIR}/datasets/Book/BX-Books.csv',delimiter=';', encoding='latin-1', error_bad_lines=False)
    books = books.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'])
    books = books[~books['Year-Of-Publication'].isin(['Gallimard','DK Publishing Inc'])]
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'] )
    books = books[(books['Year-Of-Publication']<2022) & (books['Year-Of-Publication'] != 0)]

    ratings = pd.read_csv(f'{ROOT_DIR}/datasets/Book/BX-Book-Ratings.csv',delimiter=';', encoding='latin-1')
    ratings = ratings.replace(0,5)

    ratings = pd.merge(ratings,users)
    ratings = pd.merge(ratings,books)

    ratings = ratings.drop(columns=['Book-Title','Book-Author','Publisher'])
    ratings['transaction_date'] = datetime(2000, 1, 1, 0, 0)
    ratings["purchase"] = 1

    ratings.index = ratings.transaction_date

    columns={ "Age": "age",
        "Book-Rating": "rating",
        "Location": "location",
        "Year-Of-Publication": "year",
        "User-ID": "cust_id",
        "ISBN": "article_id"
    }

    ratings = ratings.rename(columns=columns)
    ratings.year = ratings.year.map(lambda x : decades[str(int(x/10)%100)] )

    features2values = dict()
    for feature in ['age','location','year']:
        features2values[feature] = list(ratings[feature].unique())

    sys.exit('LOL')
    return ratings,features2values

def get_data(article_id=None, dataset="MovieLens"):
    if dataset == "Tafeng":
        return get_dataset_tafeng(article_id)
    elif dataset == "MovieLens":
        return get_dataset_movie_lens(article_id)
    elif dataset == "Book":
        return get_dataset_books(article_id)
    elif dataset == "Yelp":
        return get_dataset_yelp(article_id)
    else:
        raise NotImplementedError("Dataset not Available ... !")
