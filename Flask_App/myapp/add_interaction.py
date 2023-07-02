from sqlalchemy import create_engine

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

class Interactions(Base):
    """
    Create an Intercation object.
    Args:
        index: primary key required by SQLAlchemy
        article_id: the article id
        title: Article title
        user_id: current user
    """
    __tablename__ = "interactions"
    index: Mapped[int] = mapped_column(primary_key=True)# primary keys are required by SQLAlchemy
    article_id: Mapped[float] = mapped_column(unique=False)
    title: Mapped[str] = mapped_column(String())
    user_id: Mapped[int] = mapped_column(String(100))


def insert_interaction(article_id,title,user_id):
    """
    Insert an interaction into `Recommendations.interactions` table.
    sqlalchemy.orm will be used.

    Args:
        article_id: the article id
        title: Article title
        user_id: current user
    """
    engine = create_engine('sqlite:///data/Recommendations.db')
    Base.metadata.create_all(engine)
    
    from sqlalchemy.orm import Session
    with Session(engine) as session:    
        new_interaction = Interactions(
            article_id=float(article_id),
            title = title,
            user_id = int(user_id) 
        )
        try:
            session.add_all([new_interaction])
            session.commit()
            #print(f"Intercation inserted into intercations table!")
        except:
            pass

