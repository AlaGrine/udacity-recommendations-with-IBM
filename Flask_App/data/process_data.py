import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import pandas as pd
from sqlalchemy import create_engine

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)# primary keys are required by SQLAlchemy
    email: Mapped[str] = mapped_column(String(100), unique=True)
    password: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(1000))
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, email={self.email!r})"


def load_data(articles_filepath, interactions_filepath):
    """
    Load Dataframe from articles_community and user-item-interactions filepaths.
    INPUT
        articles_filepath (string): Filepath to articles_community.csv file.
        interactions_filepath (string): Filepath to user-item-interactions.csv file.

    OUTPUT
        df (DataFrame) : User-artcile Intercations
        df_content (DataFrame) : Artcicle details        
    """

    # 1. Load articles_community dataset
    df_content = pd.read_csv(articles_filepath)
    # 2. Load user_item_interactions dataset
    df = pd.read_csv(interactions_filepath)
    
    return df,df_content

def email_mapper(df):
    """
    Map the user email to a user_id column.
    INPUT
        df (DataFrame): User-artcile Intercations.

    OUTPUT
        email_encoded (Series) : Series of user_id        
    """
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

def clean_data(df,df_content):
    """
    Clean the DataFrames as follows:
        1.  Clean df_content: Remove dupplicates; only keep the first article_id.
        2.  Clean df: Map the user email to a user_id column and remove the email column.
        3.  Get unqiue user_ids.
    INPUT
        df (DataFrame) : User-artcile Intercations
        df_content (DataFrame) : Artcicle details   

    OUTPUT
        df (DataFrame): cleaned df dataset.
        df_content (DataFrame): cleaned df_content dataset.
    """

    # 1. Drop Column 'Unnamed: 0'
    del df['Unnamed: 0']
    del df_content['Unnamed: 0']

    # 2. Remove dupplicates from df_content - only keep the first article_id
    df_content = df_content.drop_duplicates(subset='article_id', keep='first')

    # 3. Map the user email to a user_id column and remove the email column
    email_encoded = email_mapper(df)
    del df['email']
    df['user_id'] = email_encoded

    # 4. Capitalize title
    df['title'] = df['title'].apply(lambda t:t.capitalize())  

    return df, df_content

def save_data(df, df_content, database_filename):
    """
    Save the three dataframes into an SQLite database using pandas `to_sql` method combined 
    with the SQLAlchemy library.
    Args:
        df, df_content (DataFrame): three dataframes (cleaned by clean_data function).
        database_filename (str): Name of the SQLite database file.
    """

    engine = create_engine("sqlite:///" + database_filename)

    # 1. Create data tables. Replace if exists (default='fail')
    df.to_sql("interactions", engine, if_exists='replace', index=True)  # primary keys are required by SQLAlchemy
    df_content.to_sql("content", engine, if_exists='replace', index=False) 

    # 2. Create users.db

    print("Create users.dB ...\n")

    engine = create_engine("sqlite:///instance/users.db")
    Base.metadata.create_all(engine)
    distinc_users = list(df.user_id.unique())
    
    from sqlalchemy.orm import Session
    with Session(engine) as session:    
        for user_id in distinc_users:
            user_ = User(
                id=int(user_id),
                name = "user"+str(user_id),
                email = "user"+str(user_id)+"@test.com",
                password = generate_password_hash("user", method='sha256'), 
            )
            try:
                session.add_all([user_])
                session.commit()
            except:
                pass





def main():
    if len(sys.argv) == 4:
        articles_filepath, interactions_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    Articles: {}\n    Interactions: {}".format(
                articles_filepath, interactions_filepath
            )
        )
        df,df_content = load_data(articles_filepath, interactions_filepath)

        print("Cleaning data...")
        df,df_content = clean_data(df,df_content)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df,df_content, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the articles and user-article-interactions "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. "
            "\n\nExample: "
            "python data/process_data.py "
            "data/articles_community.csv "
            "data/user-item-interactions.csv "
            "data/Recommendations_dB.db"
        )


if __name__ == "__main__":
    main()
