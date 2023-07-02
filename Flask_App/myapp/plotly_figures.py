import json
import plotly
import pandas as pd
import numpy as np

from plotly.graph_objs import Histogram, Scatter

from .wordcloud_parameters import worldcloud_generator, wordcloud_params

def return_plots(df,df_content):
    """
    Return Plotly figure configuration (including data and layout config).

    Parameters
    ----------- 
        df (DataFrame) : User-artcile Intercations (loaded from SQLite database).
        df_content (DataFrame) : Artcicle details (loaded from SQLite database).
    
    Output
    ----------- 
        graphs_dahboard: Plotly figure configs
        
    """

    #####################################################
    #                   1. Extract data 
    #####################################################

    # 1.1 User-Article Interactions
    nb_articles_by_user = df.groupby('user_id')['article_id'].count().reset_index(name='count').\
                    sort_values(by='count',ascending=False)

    # 1.2 doc_body length (keep only 99% of samples as the rest are outliers)
    doc_length_df = df_content[~df_content.doc_body.isnull()]['doc_body'].apply(lambda s: len(s.split(' ')))
    percentile_99 = np.percentile(doc_length_df, 99)
    doc_length_df = doc_length_df[doc_length_df < percentile_99]

    # 1.3 WordCloud
    wc = worldcloud_generator(df_content[~df_content.doc_body.isnull()]['doc_body'],
                             background_color='white', max_words=200)
    # 1.4 Get wordcloud parametres (positions, word frequency, colors...)
    position_x_list, position_y_list, freq_list, size_list, color_list, word_list = wordcloud_params(wc)

    #####################################################
    #                   2. Create visuals
    #####################################################

    graphs_dahboard = [

        # Graph 1 - User-Article Interactions - Histogram
        {
            'data': [
                Histogram(x=nb_articles_by_user['count'])
            ],
            'layout': {
                'title': "User-Article Interactions",
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Number of user interactions'}
            }
        },

        # Graph 2 - Distribution of message length
        {
            'data': [
                Histogram(x=doc_length_df)
            ],
            'layout': {
                'title': "Distribution of Article Lengths",
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Number of words in message'}
            }
        },

         # Graph 3 - Wordcloud (Most common words)
        {
            'data': [
                Scatter(x=position_x_list,
                        y=position_y_list,
                        textfont=dict(size=size_list,
                                      color=color_list),
                        hoverinfo='text',
                        #hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                        mode='text',
                        text=word_list
                        )
            ],
            'layout': {
                'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'height': 700,
                'title': 'Most Common Words',
            }
        }
    ]

    return graphs_dahboard