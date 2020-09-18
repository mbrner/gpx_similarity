from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, LargeBinary, Float, Boolean

DATABASE_NAME = 'gpx_similarity'


def create_database(DATABASE_NAME):
    pass

URI = f'postgresql+psycopg2://postgres:postgres@localhost:5435/{DATABASE_NAME}'
ENGINE = create_engine(URI)

# create an engine
# # create a configured "Session" class
# Session = sessionmaker(bind=ENGINE)
# # create a Session
# session = Session()
Base = declarative_base()

class OSMImages(Base):
    __tablename__ = 'osm_images'
    id = Column(Integer, primary_key=True)
    origin = Column(String)
    idx = Column(Integer)
    zoom = Column(Integer)
    p_0_lat = Column(Float)
    p_0_long = Column(Float)
    p_1_lat = Column(Float)
    p_1_long = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    route_type = Column(String)
    show_route = Column(Boolean)
    image = Column(LargeBinary)

Base.metadata.create_all(ENGINE, checkfirst=True)