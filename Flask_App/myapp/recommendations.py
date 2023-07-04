import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from time import time
import datetime
import warnings
warnings.simplefilter("ignore", UserWarning)

###########################################################
#         Part-1. Rank Based Recommendations
###########################################################

def rank_based_recommendations(m,df_article_recap):
    '''
    Return the m top articles ordered with most interactions as the top

    INPUT:
        m (int): the number of top articles to return
        df_article_recap: Dataframe with article_id, title, descreption and nb_interactions        
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions.   
    '''

    df_top_articles = df_article_recap.head(m)

    recs_ids = list(df_top_articles['article_id'].values)
    recs_titles = list(df_top_articles['title'].values)
    recs_descriptions = list(df_top_articles['doc_description'].values)
    recs_num_interactions = list(df_top_articles['num_interactions'].values)

    return recs_ids,recs_titles,recs_descriptions,recs_num_interactions

###########################################################
#          Part-2 Knowledge Based Recommendations 
#                   top tanked + filter
###########################################################

def Knowledge_based_recommendations(m,df_article_recap,filter_txt):
    '''
    Return the m top articles filtered and ordered with most interactions as the top.
    The text (filter) is first tokenised and cleaned before being applied as a filter.
    Note: At least two characters are required for the filter to be valid.

    INPUT:
        m (int): the number of top articles to return
        df_article_recap: Dataframe with article_id, title, descreption and nb_interactions
        filter_txt (string): text to tokenize and preprocess.       
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions.
    '''
    article_ids = []
    article_titles = []
    article_descriptions = []
    article_num_interactions = []

    # if filter is not empty
    if len(filter_txt)>1:
        filter_tokens = get_clean_tokens(filter_txt)
    
        # 1. Select from df_article_recap where title or dào_descreption contains `all` filter tokens

        if_tokens_in_title = df_article_recap.title.apply(\
            lambda title_: True if all(token in title_.lower() for token in filter_tokens) else False)
        
        if_tokens_in_desc = df_article_recap.doc_description.apply(\
            lambda desc_: True if all(token in desc_.lower() for token in filter_tokens) else False)

        results = df_article_recap.loc[if_tokens_in_title | if_tokens_in_desc]

        # 2. Sort by interactions
        results = results.sort_values(by="num_interactions",ascending=False)
        results = results.head(m)

        article_ids = list(results['article_id'].values)
        article_titles = list(results['title'].values)
        article_descriptions = list(results['doc_description'].values)
        article_num_interactions = list(results['num_interactions'].values)

    return article_ids,article_titles,article_descriptions,article_num_interactions
    

def get_clean_tokens(text):
    """
    Tokenize and preprocess text:
        1. find urls and replace them with 'urlplaceholder'
        2. Normalization of the text : Convert to lowercase
        3. Normalization of the text : Remove punctuation characters
        4. Split text into words using NLTK
        5. remove stop words 
        
    INPUT:
        text (string): string to tokezine.
            
    OUTPUT:
        clean_tokens (list): list of lemmatized and cleaned words.
    """
    
    # 1. find urls and replace them with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    
    # 2. Convert to lowercase
    text = text.lower().strip() 
    
    # 3. Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # 4. Split text into words using NLTK
    words = word_tokenize(text)
    
    # 5. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    return words


###########################################################
#         Part-3. User-User Based Collaborative Filtering
###########################################################

# 1. create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
        df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
        user_item - user item matrix (number of interactions per user/article)
    
    Description:
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
        an article and a 0 otherwise
    '''
    # Fill in the function here
    
    # 1. unstack the user-item dataframe
    user_item = df.drop_duplicates().groupby(['user_id', 'article_id'])['title'].count().unstack()
    
    # 2. fill missing values with 0
    user_item = user_item.fillna(0)
    
    # 3. convert to int
    user_item = user_item.astype('int')
    
    return user_item # return the user_item matrix 



# 2. Create function below which should take a user_id and provide an ordered 
# list of the most similar users to that user (from most similar to least similar).

def find_similar_users(user_id, user_item):
    '''
    INPUT:
        user_id - (int) a user_id
        user_item - user item matrix (number of interactions per user/article)
    
    OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
        Computes the similarity of every pair of users based on the dot product
        Returns an ordered
    
    '''
    # 1. compute similarity of each user to the provided user (user_id)
    similarity = user_item.dot(user_item.loc[user_id])

    # 2. sort by similarity
    similarity = similarity.sort_values(ascending=False)

    # 3. create list of just the ids
    most_similar_users = similarity.index.tolist()
    
    # 4. remove the own user's id
    most_similar_users.remove(user_id)
       
    # return a list of the users in order from most to least similar
    return most_similar_users 


# 3. Now that we have a function that provides the most similar users to each user, 
# we will use these users to find articles we can recommend.

def get_article_details(article_ids, df_article_recap):
    '''
    INPUT:
        article_ids - (list) a list of article ids
        df_article_recap: Dataframe with article_id, title, descreption and nb_interactions
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions.
    '''

    article_ids = list(map(float, article_ids)) # convert article_ids to float
    details = df_article_recap[df_article_recap.article_id.isin(article_ids)]
    details = details.sort_values(by="num_interactions",ascending=False)

    recs_ids = list(details['article_id'].values)
    recs_titles = list(details['title'].values)
    recs_descriptions = list(details['doc_description'].values)
    recs_num_interactions = list(details['num_interactions'].values)

    return recs_ids,recs_titles,recs_descriptions,recs_num_interactions 


def get_user_articles(user_id, user_item):
    '''
    INPUT:
        user_id - (int) a user id
        user_item - user item matrix (number of interactions per user/article)
    
    OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
    
    Description:
        Provides a list of the article_ids that have been seen by a user
    '''

    # 1. Get past articles read by user_id
    user_articles = user_item.loc[user_id,:]
    
    # 2. Get only articles that user_id has interacted with
    article_ids = user_articles[user_articles == 1].index.values.tolist()
    
    return article_ids


def user_user_recs(user_id, m,user_item,df):
    '''
    INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations we want for the user
    
    OUTPUT:
        recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # 1. Get most similar users 
    closest_neighbors = find_similar_users(user_id,user_item)
    
    # 2. Get past article ids read by user_id
    user_article_ids = get_user_articles(user_id,user_item) 
    
    # 3. Create recommendations for this user
    recs = np.array([])
    
    for neighbor in closest_neighbors:
        # Get neighbor article ids
        neighbor_article_ids = get_user_articles(neighbor,user_item)
        
        # Find new_recs: the list of articles that are not seen by user_id
        new_recs = np.setdiff1d(neighbor_article_ids,user_article_ids, assume_unique=True)
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))
        
        # Exit the loop if we have enough recommendations (at least m recs) 
        if len(recs) > m:
            break 
            
    recs = recs[:m].tolist()
    
    return recs # return our recommendations for this user_id    

# 4. Now we are going to improve the consistency of the user_user_recs function from above.
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given
#   user - choose the users that have the most total article interactions before choosing those 
#   with fewer article interactions.
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles 
#   starts below m and ends exceeding m, choose articles with the articles with the most total interactions
#   before choosing those with fewer total interactions. 
#   This ranking should be what would be obtained from the top_articles function we wrote earlier.

def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
        user_id - (int)
        df - (dataframe) contains user-article interactions 
        user_item - user item matrix (number of interactions per user/article)
    
            
    OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''  
    
    # 1. Compute similarity of each user to the provided user
    similarity = user_item.dot(user_item.loc[user_id]).sort_values(ascending=False)
    
    # 2. Similarity dataframe
    df_similarity = pd.DataFrame(similarity.drop(user_id),columns=['similarity']).reset_index()
    df_similarity.columns = ['neighbor_id','similarity']
    
    # 3. Nbre of interactions DataFrame
    df_interactions = pd.DataFrame(df.user_id.value_counts().reset_index())
    df_interactions.columns=['neighbor_id','num_interactions']
    
    # 4. merge similarity and interactions dataframe
    neighbors_df = pd.merge(df_similarity,df_interactions,on="neighbor_id")    
    
    # 5. sort by similarity and interactions
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)
    
    return neighbors_df # Return the dataframe specified in the doc_string


def get_top_articles_per_User(user_id,n,df):
    '''
    INPUT:
        n - (int) the number of top articles to return. if n=None then all article ids are returned.
        df - (pandas dataframe) df as defined at the top of the notebook 
        user_id (int): the user id
    
    OUTPUT:
        article_ids - (list) A list of the top 'n' article ids for user_id    
    '''    
    
    top_articles_ids = df[df.user_id==user_id]['article_id'].value_counts()
    
    if n!=None:
        top_articles_ids = top_articles_ids.head(n) # keep only the top 'n' 
        
    # return a list of top_articles indexes
    top_articles_ids = list(top_articles_ids.index)
 
    return top_articles_ids


def user_user_recs_part2(user_id, m, user_item, df,df_article_recap,content_recs_ids,user_article_ids):
    '''
    INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user
        user_item - user item matrix (number of interactions per user/article)
        df - (pandas dataframe) df as defined at the top of the notebook
        df_article_recap: dataframe containing article details (title, desc, nbre intercations...),

        content_recs_ids : list of recommendations returned by content-based recommender.
                           This list must be excluded from `user-user` recommendations.
        user_article_ids : articles already read by user_id (use `get_user_articles` method).
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions.
    
    Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found
    
    Notes:
        * Choose the users that have the most total article interactions 
        before choosing those with fewer article interactions.

        * Choose articles with the articles with the most total interactions 
        before choosing those with fewer total interactions. 
   
    '''
    # 1. Get most similar users 
    closest_neighbors_df = get_top_sorted_users(user_id, df, user_item)
    closest_neighbors = closest_neighbors_df.neighbor_id.tolist()
    
    # 2. Concatenate user_article_ids with content_recs_ids
    exclude_article_ids = np.unique(np.concatenate([user_article_ids, content_recs_ids], axis=0)).astype(float)
    
    # 3. Create recommendations for user_id
    recs = np.array([])
    
    for neighbor in closest_neighbors:
        # Get neighbor article ids
        # We choose articles with the most total interactions before choosing those with fewer total interactions.
        neighbor_article_ids = get_top_articles_per_User(neighbor,None,df)
        
        # Find new_recs: the list of articles that are not seen by the user and not in content based recs
        new_recs = np.setdiff1d(neighbor_article_ids, exclude_article_ids, assume_unique=True)
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0)).astype(float)
        
        # Exit the loop if we have enough recommendations (at least m recs) 
        if len(recs) >= m:
            break
    
    recs = recs[:m].tolist()

    recs_ids,recs_titles,recs_descriptions,recs_num_interactions = \
        get_article_details(recs, df_article_recap)
    
    return recs_ids,recs_titles,recs_descriptions,recs_num_interactions



###########################################################
#         Part-4. Content Based Recommendations (NLP)
###########################################################

# Another method we might use to make recommendations is to perform a ranking of the 
# highest ranked articles associated with some term. 
# We might consider content to be the doc_body, doc_description, or doc_full_name.

def tokenize(text):
    """
    Tokenize and preprocess text:
        1. find urls and replace them with 'urlplaceholder'
        2. Normalization of the text : Convert to lowercase
        3. Normalization of the text : Remove punctuation characters
        4. Split text into words using NLTK
        5. remove stop words
        6. Lemmatization    
        
    INPUT:
        text (string): string to tokezine.
            
    OUTPUT:
        clean_tokens (list): list of lemmatized and cleaned words.
    """
    
    # 1. find urls and replace them with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    
    # 2. Convert to lowercase
    text = text.lower().strip() 
    
    # 3. Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # 4. Split text into words using NLTK
    words = word_tokenize(text)
    
    # 5. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    # 6.1 Reduce words to their root form
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    # 6.2 Lemmatize verbs by specifying pos
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return clean_tokens


def compute_dot_prod_articles(df_article_recap):
    """
    Compute the dot product to get an article-article matrix of similarities.
        
    INPUT:
        df_article_recap: dataframe containing article details (title, descreption...).
            
    OUTPUT:
        dot_prod_articles : article-article matrix of similarities.
    """
    
    # 1. Convert doc_description to a matrix of TF-IDF features using `TfidfVectorizer`
    # We will be using `doc_description` from df_article_recap dataframe.

    # articles_description = df_article_recap['doc_description'].fillna(df_article_recap['doc_full_name'])
    articles_description = df_article_recap['doc_description']

    tfidf = TfidfVectorizer(tokenizer=tokenize)
    articles_vect = tfidf.fit_transform(articles_description)
    
    # 2. Convert the sparse matrix to a DataFrame
    articles_vect_df = pd.DataFrame.sparse.from_spmatrix(articles_vect)

    # 3. Take the dot product to obtain an artcle article matrix of similarities
    dot_prod_articles = articles_vect_df.dot(np.transpose(articles_vect_df))

    return dot_prod_articles

def make_content_recs(_id, _id_type, m,df,df_article_recap,dot_prod_articles):
    '''
    INPUT:
        _id (list of int): either a user or article id(s)
        _id_type (str): "article" or "user" 
        m (int): number of recommendations to return 
        df_article_recap (DataFrame): dataframe containing article details
        df (dataframe): dataframe containing article_id and user_id and intercations
        dot_prod_articles : article-article matrix of similarities        
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions.   
    '''
    if(_id_type=="user"):
        user_id = _id[0] # for _id is a list
        try:
            # get past articles read by the user
            article_ids = get_user_articles(user_id,user_item) 
        except KeyError: # user does not exist
            recs_ids,recs_titles,recs_descriptions,recs_num_interactions = \
                user_user_recs_part2(user_id, m, user_item, df,df_article_recap,content_recs_ids,user_article_ids)

            return recs_ids,recs_titles,recs_descriptions,recs_num_interactions
    
    else:
        all_article_ids = _id
        
        # Initialize df_similar_articles   
        df_similar_articles = pd.DataFrame(data={'article_id': [0], 
                                                'similarity': [0], 
                                                'num_interactions': [0],
                                                })
        for article_id in all_article_ids:
            try :
                # 1. Find the row of each article
                article_idx = np.where(df_article_recap['article_id']== article_id)[0][0]

                # 2. Search article_idx in dot_prod_articles matrix
                dot_idx = dot_prod_articles[article_idx]    
                
                # 3. convert dot_idx to dataframe
                dot_idx_df = pd.DataFrame(dot_idx.reset_index(name='similarity'))
                # dot_idx_df = dot_idx_df[~(dot_idx_df['index']==article_idx)] # drop article_idx

                # 4. Merge dot_idx_df with article_interactions:
                dot_idx_df = dot_idx_df.merge(df_article_recap,right_index=True,left_index=True)

                # 5. Round similarity to 2 decimal places.
                dot_idx_df['similarity'] = dot_idx_df['similarity'].round(2) 

                # 6. Order by similarty then occurences
                similar_articles = dot_idx_df.sort_values(by=['similarity', 'num_interactions'], ascending=False) 
                similar_articles = similar_articles[['article_id','similarity', 'num_interactions']]
                
                # 7. concat similar_articles with df_similar_articles
                df_similar_articles = pd.concat([df_similar_articles,similar_articles])
                
            except:
                pass

        # 8. Sort by similarity and occurences
        df_similar_articles = df_similar_articles.sort_values(by=['similarity', 'num_interactions'], ascending=False)  

        # 9. Drop duplicates (keep the highest similarity)
        df_similar_articles = df_similar_articles.drop_duplicates(subset='article_id', keep='first')

        # 10. Drop all_article_ids (Articles already read)
        df_similar_articles = df_similar_articles[~(df_similar_articles.article_id.isin(all_article_ids))]

        # 10. Get recommonation ids 
        recs = df_similar_articles.article_id[:m].values
    
    recs_ids,recs_titles,recs_descriptions,recs_num_interactions = \
        get_article_details(recs, df_article_recap) 

    return recs_ids,recs_titles,recs_descriptions,recs_num_interactions


###########################################################
#               Part-5. Matrix Factorization
###########################################################

def compute_svd(user_item,num_latent_features):
    '''
    compute svd and Restructure with num_latent_features

    INPUT:
        user_item: user item matrix (number of interactions per user/article)
        num_latent_features (int): number of latent features    
    
    OUTPUT:
        u_new, s_new, vt_new: Restructured three SVD matrices.   
    '''    

    # Perform SVD on the User-Item Matrix 
    u, s, vt =  np.linalg.svd(user_item)

    # Restructure with latent features
    s_new = np.diag(s[:num_latent_features])
    u_new = u[:, :num_latent_features]
    vt_new = vt[:num_latent_features, :]    
    
    return u_new, s_new, vt_new


def make_svd_recs(user_id, m , vt_new,user_item,user_article_ids,df_article_recap): 
    '''
    Returns m recommendations using SVD. 
    Cosine similarity will be used.

    INPUT:
        user_id (int): user ID
        m (int): number of recommendations to return 
        vt_new: the restructured SVD vt matrix with k latent features
        user_item: user item matrix (number of interactions per user/article)
        user_article_ids : articles already read by user_id (use `get_user_articles` method).
        df_article_recap (DataFrame): dataframe containing article details
    
    OUTPUT:
        recs_ids: list of recommendation ids.
        recs_titles: list of recommendation titles.
        recs_descriptions: list of recommendation descreptions.
        recs_num_interactions: list of number of interactions. 
    '''   

    all_similarity_df = pd.DataFrame(columns=['article_id','similar_article_id','similarity'])
    
    v_new = vt_new.T # transpose vt_new so that articles are in rows
    
    # calculate magnitude. will be used to calculate cosine similartity
    # Reference: https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/
    magnitude = np.sqrt(np.einsum('ij, ij -> i', v_new, v_new))

    for article_id in user_article_ids:
        try :
            # 1. Find article index in user_item matrix
            article_index = np.where(user_item.columns==article_id)[0][0]
            article_row = v_new[article_index, :]

            # 2. calculate cosine_similarity
            similarity = np.dot(article_row, vt_new) / (magnitude[article_index] * magnitude)
            
            # 3. Create dataframe containing: article_id,similar_article_id and similarity
            sim_df = pd.DataFrame(similarity,columns=['similarity']).reset_index()
            sim_df = sim_df.rename(columns={"index": "similar_article_index"})
            sim_df['similar_article_id'] = sim_df['similar_article_index'].apply(lambda index_:user_item.columns[index_])
            sim_df['article_index'] = article_index
            sim_df['article_id'] = user_item.columns[article_index]
            sim_df = sim_df[['article_id','similar_article_id','similarity']]
            sim_df = sim_df[sim_df.similar_article_id != sim_df.article_id]

            # 4. concat with all_similarity_df
            all_similarity_df = pd.concat([all_similarity_df,sim_df])

        except:
            pass

    # 5. sort by cosine similarity
    all_similarity_df = all_similarity_df.sort_values(by="similarity",ascending=False)

    # 6. exclude already read articles
    all_similarity_df = all_similarity_df[~all_similarity_df.similar_article_id.isin(user_article_ids)]

    recs = all_similarity_df.head(m)['similar_article_id'].to_list()

    recs_ids,recs_titles,recs_descriptions,recs_num_interactions = \
        get_article_details(recs, df_article_recap)
    
    return recs_ids,recs_titles,recs_descriptions,recs_num_interactions


#############################################################################
#         Last Part. Put all together and make customizer recs
#############################################################################

def custom_recs(user_id,m,df,df_article_recap,user_item,list_unique_users,dot_prod_articles,u_new, s_new, vt_new):
    '''
    Returns customised recommendations to the user as follows:
    - If new user: returns top ranked articles.
    - Else if the user has searched for key words, then returns top ranked and filtered articles.
    - Else, returns three lists of recommendations:
        1- SVD based recs, using the `make_svd_recs` method 
        2- Conetnt-based recs, using the `make_content_recs` method       
        3- User-user based recs, using the `user_user_recs_part2` method 

    INPUT:
        user_id: user ID
        m (int): number of recommendations to return 
        df (dataframe): dataframe containing article_id and user_id and intercations
        df_article_recap (DataFrame): dataframe containing article details
        user_item: user item matrix (number of interactions per user/article)
        list_unique_users: unique active users (unique user_id in `df`).
        dot_prod_articles : article-article matrix of similarities
        u_new, s_new, vt_new : SVD matrices.        
    
    OUTPUT: 12 lists

        For `SVD_based` or `top_ranked` recommendeation (Top ranked is the user is new):
            recs_ids: list of recommendation ids.
            recs_titles: list of recommendation titles.
            recs_desc: list of recommendation descreptions.
            recs_nbViews: list of number of interactions. 

        For `content_based` recommendeation:
            rec_content_ids: list of recommendation ids.
            rec_content_titles: list of recommendation titles.
            rec_content_desc: list of recommendation descreptions.
            rec_content_nbViews: list of number of interactions.

        For user-user based recommendeation (called `Other user are viewing`):
            recs_users_ids: list of recommendation ids.
            recs_users_titles: list of recommendation titles.
            recs_users_desc: list of recommendation descreptions.
            recs_users_nbViews: list of number of interactions.   
    '''

    # 1. For new users, recommend top m ranked articles

    if (user_id not in list_unique_users):
        rec_ids,rec_titles,rec_desc,rec_nbViews = \
            rank_based_recommendations(m,df_article_recap)

        # Set `Other user are viewing` and `content` lists to None
        zeros = [0 for k in range(m)]
        empty_str = ['' for k in range(m)]
        rec_users_ids,rec_content_ids = (zeros,zeros)
        rec_users_titles,rec_content_titles = (empty_str,empty_str)
        rec_users_desc,rec_content_desc = (empty_str,empty_str)
        rec_users_nbViews,rec_content_nbViews = (zeros,zeros)

        last_read_article_title = "" # initialize the last read article 

    else:
        # Get past article ids read by user_id
        user_article_ids = get_user_articles(user_id,user_item)

        # 2. SVD based recommendation
        rec_ids,rec_titles,rec_desc,rec_nbViews = \
            make_svd_recs(user_id, m , vt_new,user_item,user_article_ids,df_article_recap)
            # make_svd_recs(user_id, m, user_item, u_new, s_new, vt_new,user_article_ids,df_article_recap)

        # 3. content based recommendation

        last_read_article = df[df.user_id==user_id].tail(1)
        last_read_article_id = last_read_article.article_id.values[0]
        last_read_article_title = last_read_article.title.values[0]      

        rec_content_ids,rec_content_titles,rec_content_desc,rec_content_nbViews = \
            make_content_recs([last_read_article_id], "article", m,df,df_article_recap,dot_prod_articles)

        # 4. user-user based recommendation

        rec_users_ids,rec_users_titles,rec_users_desc,rec_users_nbViews = \
            user_user_recs_part2(user_id, m, user_item, df,df_article_recap,rec_content_ids,user_article_ids)


    return rec_ids,rec_titles,rec_desc,rec_nbViews,\
        rec_content_ids,rec_content_titles,rec_content_desc,rec_content_nbViews, \
        rec_users_ids,rec_users_titles,rec_users_desc,rec_users_nbViews,last_read_article_title

    