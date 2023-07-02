
from flask import Blueprint, render_template
from flask_login import login_required, current_user
from . import db
from .recommendations import create_user_item_matrix,compute_dot_prod_articles
from .recommendations import Knowledge_based_recommendations,custom_recs,compute_svd
from .add_interaction import insert_interaction

from flask import Flask
from flask import render_template, request, jsonify

from sqlalchemy import create_engine, text as sql_text
import pandas as pd
import numpy as np

import json
import plotly
from myapp.plotly_figures import return_plots

# main - Flask Blueprint
main = Blueprint('main', __name__)

# Global variables
Nb_recs = 12 # Number of recommendations returned by our recommendation engine.
Threshold_gold = 10 # artciles that are viewed more than `Threshold_gold` are marked as gold.

# 1. Load data from SQLite dB
connection = create_engine('sqlite:///data/Recommendations.db')

# `df` contains interactions. 
# Updated every time a new interaction is made.
def update_matrices():

    # 1. Load interactions data
    query_intercations = "select * from interactions"
    df = pd.read_sql_query(con=connection.connect(),  sql=sql_text(query_intercations))
    # Drop column `index` from df (`index` is a primary key required by SQLAlchemy (see process_data.py)
    df = df.drop(['index'],axis=1)   

    # 2. Get unique user_id
    list_unique_users = list(df.user_id.unique()) 

    # 3. Compute user_item matrix 
    user_item = create_user_item_matrix(df)  

    # 4. Compute SVD
    u_new, s_new, vt_new = compute_svd(user_item,100) 

    # 5. Load article content
    query_content = "select * from content"
    df_content = pd.read_sql_query(con=connection.connect(),  sql=sql_text(query_content))

    # 6. Merge df and df_content to get article details (article_id, title , nb interactions, desc...) 
    df_article_recap = df.groupby(['article_id','title']).count().reset_index()
    df_article_recap = df_article_recap.rename(columns={"user_id": "num_interactions"})
    df_article_recap = df_article_recap.sort_values(by="num_interactions",ascending=False)
    df_article_recap = pd.merge(df_article_recap,df_content,on='article_id',how='left')

    df_article_recap['doc_description'] = df_article_recap['doc_description'].fillna(df_article_recap['title'])
    df_article_recap['doc_full_name'] = df_article_recap['doc_full_name'].fillna(df_article_recap['title'])

    # 7. Create dot_prod_articles
    dot_prod_articles = compute_dot_prod_articles(df_article_recap)


    return df,list_unique_users,user_item,u_new, s_new, vt_new,df_content,df_article_recap,dot_prod_articles

# Update matrices:
df,list_unique_users,user_item,u_new, s_new, vt_new,df_content,\
    df_article_recap,dot_prod_articles = update_matrices()

# 4. Get plotly figure configuration (call return_plots):
graphs_dahboard = return_plots(df,df_content)


# 5. Home page
@main.route('/')
def index():
    return render_template('index.html')

# 6. Recommendation page
@main.route('/recommendations')
@login_required
def recommendations():
    user_id = current_user.id
    user_name = current_user.name
    
    # Update matrices:
    df,list_unique_users,user_item,u_new, s_new, vt_new,df_content,\
        df_article_recap,dot_prod_articles = update_matrices()

    # Check user type (New user is a user without intercations)
    is_new_uesr =  False
    if (user_id not in list_unique_users):
        is_new_uesr = True 

    # Get recommendation list    
    rec_ids,rec_titles,rec_desc,rec_nbViews,\
    rec_content_ids,rec_content_titles,rec_content_desc,rec_content_nbViews, \
    rec_users_ids,rec_users_titles,rec_users_desc,rec_users_nbViews,last_read_article_title = \
        custom_recs(user_id,Nb_recs,df,df_article_recap,user_item,\
                    list_unique_users,dot_prod_articles,u_new, s_new, vt_new)

    # render html page    
    return render_template('app_article_recs.html', name=user_name,id=user_id, 
                            is_new_uesr = is_new_uesr,

                            rec_ids = rec_ids,
                            rec_titles = rec_titles,
                            rec_desc = rec_desc,
                            rec_nbViews = rec_nbViews,

                            rec_users_ids = rec_users_ids,
                            rec_users_titles = rec_users_titles,
                            rec_users_desc = rec_users_desc,
                            rec_users_nbViews = rec_users_nbViews,

                            rec_content_ids = rec_content_ids,
                            rec_content_titles = rec_content_titles,
                            rec_content_desc = rec_content_desc,
                            rec_content_nbViews = rec_content_nbViews,

                            last_read_article_title = last_read_article_title,

                            Threshold_gold = Threshold_gold,
                            Nb_recs = len(rec_ids)
                            )


# 7. `app_article_read` page
@main.route('/read_article')
@login_required
def read_article():
    user_id = current_user.id
    user_name = current_user.name

    # Get article_id from request.args
    article_id = request.args.get('article_id', '')  

    # Select article_title from df, and (doc_description,doc_body) form df_content  
    article_title = df[df.article_id == float(article_id)]['title'].iloc[0]

    # Some article details are missing in df_content:
    doc_description,doc_body = ("","Article content not found.")
    try:
        doc_description,doc_body = df_content[df_content.article_id == float(article_id)]\
                                    [['doc_description','doc_body']].values[0]
    except:
        pass

    # Insert this intercation into `Recommendations.interactions` table:
    try:
        insert_interaction(article_id,article_title,user_id)
    except:
        print("Can not insert new interaction!")       
    
    # render html page.
    return render_template(
        'app_article_read.html',
        article_id=article_id,
        article_title = article_title,
        doc_description = doc_description,
        doc_body = doc_body
    )


# 8. Search article page
@main.route('/search_articles')
@login_required
def search_articles():
    user_id = current_user.id
    user_name = current_user.name

    # Get filter from request.args
    query = request.args.get('filter', '')  
    
    # Return knowledge-base recommendations (top ranked + filter)
    article_ids,article_titles,article_descriptions,article_num_interactions = \
        Knowledge_based_recommendations(Nb_recs,df_article_recap,query) 

    message = ""
    if len(article_ids) == 0:
        message = "no results found!"
    
    # Render html page.
    return render_template(
        'app_article_search.html',
        query = query,
        article_ids = article_ids,
        article_titles = article_titles,
        article_descriptions = article_descriptions,
        article_num_interactions = article_num_interactions,
        Threshold_gold = Threshold_gold,
        message = message
    )


# 9. Dashboard page
@main.route('/dashboard')
def dashboard():
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_dahboard)]
    graphJSON = json.dumps(graphs_dahboard, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('app_dashboard.html', ids=ids, graphJSON=graphJSON)