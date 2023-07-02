# Recommendations with IBM

Build a recommendation system and a Flask application for users of the `IBM Watson` community.

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#file_descriptions)
4. [Instructions](#instructions)
5. [Flask Application](#flaskappli)
6. [Recommendation System](#recsys)
7. [Acknowledgements](#acknowledgements)

## Project Motivation <a name="motivation"></a>

This project is part of [Udacity](https://www.udacity.com/)'s Data Science Nanodegree Program in collaboration with [IBM](https://eu-de.dataplatform.cloud.ibm.com/login?preselect_region=true).
The aim of this project is to build a **recommendation system** for users of the `IBM Watson` community.

A dataset of **5149** users, **714** articles and about **46K** interactions is used to develop this RecSys.

The project is divided into the following sections:

1. Build an ETL pipeline to extract, clean and load the data to a SQLite database.
2. Build an ensemble of recommendation algorithms (including ranked based, user-user based, contenet-based and SVD-based recs).
3. Run a WEB application to display these recommendations.

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

1. ML libraries: `NumPy`, `Pandas`, `scikit-learn`
2. NLP libraries: `NLTK`
3. SQLlite library: `SQLalchemy`
4. Authentication and security: `werkzeug`, `flask_login`
5. Web app and visualization: `Flask`, `Plotly`
6. Other libraries: `Wordcloud`

`The full list of requirements can be found in requirements.txt file.`

## File Descriptions <a name="file_descriptions"></a>

- **Flask_App** folder: contains our responsive Flask Web App.
- `myapp.py`: main file to run the web application.
- `data` folder: contains our ETL pipeline.

  - `process_data.py`: A script to build an ETL pipeline that loads the `articles` and `interactions` datasets, merge, clean and save data to a SQLite database. This script also creates a database of users.
  - `articles_community.csv` and `user-item-interactions.csv`: datasets provided by [IBM](hhttps://eu-de.dataplatform.cloud.ibm.com/login?preselect_region=true).

- `myapp` folder, contains the WEB app.

  - `templates` folder: contains 8 html pages:

    - `base.html`: this is the basis of our html code. All the other pages extend this page.
    - `app_aricle_recs.html`, `app_article_read.html`, `app_article_search.html`: display the recommendations, article content and search results respectively.
    - `app_dashboard.html`: displays visualizations of the data.
    - `index.html`: the home page.
    - `login.html` and `signup.html`: pages for authentication.

  - `static` folder: contains our customized `CSS` file and `Bootstrap` (compiled and minified `CSS` bundles and `JS` plugins).

  - `routes.py`: to render our html pages.
  - `recommendation.py`: our recommendation system is built here.
  - `auc_auth.py` and `auc_user_model`: authentication and security scripts.
  - `add_interaction.py`: add intercation (user_id and article_id) to `interactions` table.
  - `plotly_figures.py`: returns `Plotly` figure configuration (data and layout).

- **Notebooks** folder: contains the project's notebooks.

## Instructions <a name="instructions"></a>

1. Run the following command in the app's directory to clean and store data in SQLite database:

   `python data/process_data.py data/articles_community.csv data/user-item-interactions.csv data/Recommendations.db`

2. Run the following command in the app's directory to run the web app.

   `python myapp.py`

3. Go to http://0.0.0.0:3001/

## Flask Application <a name="flaskappli"></a>

1. You can create a new user via the `Sign up` page.

   To test any user of the IBM dataset you can login as follows:<br>
   **Login:** user`x`@test.com (where x is the user id. x is between 1 and 5149). <br>
   **Login example:** user25@test.com<br>
   **Password:** user.
   <div align="center">
     <img src="https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/login.png" >
   </div>

2. If you are a new user (ie. you sign up), our `RecSys` will return the top **12** ranked articles. The **Rank based** algorithm is used here.

   ![image top_ranked](https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/top_ranked.png)

3. When you read an article (for example _'Use deep learning for image classification'_), our `RecSys` updates the interactions table and generates three lists.

   The first list is based on **SVD** (_**S**ingular **V**alue **D**ecomposition_). This list is called `Recommended for you`.

   ![image svd](https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/SVD_recs.png)

4. The second list is the output of the **User-User Based Collaborative Filtering** algorithm. This list is called `Users are viewing`.

   ![image user_based](https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/User_are_viewing.png)

5. The third list is the output of the **content based** algorithm. This list is called `Because you have read`.

   ![image content_based](https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/Content_recs.png)

6. You can apply a filter (using the search TextArea). The output is a list of filtered and top ranked articles. This is the **Knowledge based** method.

   ![image content_based](https://github.com/AlaGrine/udacity-recommendations-with-IBM/blob/main/Notebooks/imgs/Knowledge_recs.png)

## Recommendation System: <a name="recsys"></a>

As we have a small data set, we have opted for an automatic update of our RecSys, which means that when you click on the recommendations link in the navbar, the recommendation list is automatically updated.

Our RecSys is an ensemble of algorithms described below.

### Rank-Based Recommendations:

To find the most popular articles, our RecSys simply uses the number of interactions. Since there are no ratings for any of the articles, we can assume that the articles with the most interactions are the most popular. These top ranked articles are then recommended to **new** users.

> The rank based recs is the simplest way to solve the **cold start problem**, i.e. no information about the user's preferences. When you sign up to the app, you will see this.

### Knowloedge-Based Recommendations:

Another way to deal with the **cold start problem** is to add filters to the web app so that users can add their own preferences.

> To build knowledge-based recommendations, the rank-based algorithm is simply applied to the filtered articles.

### User-User Based Collaborative Filtering:

Here we look at users that are **similar** to user `X` in terms of the articles they have read.
The top ranked articles read by these similar users (and not read by user `X`) are then recommended to user `X`.

> Building the user-user based algorithm is a step forward in providing more personalised recommendations for users.

### Content Based Recommendations:

For simlicity, we could consider content to be the doc descriptions.

> We start by Tokenizing the article descriptions using `TfidfVectorizer` from [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).<br>
> We then compute the **dot product** to get an article-article **matrix of similarities**.<br>
> Fianlly, we sort by similarity and number of interactions to get the most similar articles.

### SVD Based Recommendations:

Here we build a **factorization** of the user-item matrix.

**SVD** (_**S**ingular **V**alue **D**ecomposition_) from [numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) can be used because there are no null values in our user-item matrix.

<div align="center">
<p >U, S, V = np.linalg.svd(user-item)</p>`
</div>

where `V` is a matrix that provides how items (articles in this case) relate to each latent feature.<br><br>

> We can use the `V` matrix to calculate the `cosine similarity` of articles.<br>
> The most similar articles to those read by the user are returned as recommendations.

## Acknowledgements <a name="acknowledgements"></a>

Must give credit to [IBM](https://eu-de.dataplatform.cloud.ibm.com/login?preselect_region=true) for the data, and [udacity](https://www.udacity.com/) for this program.
