import pathlib
from functools import partial

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, LargeBinary, Float, Boolean, ForeignKey, Binary
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.expression import func
from sqlalchemy import text


def drop_database(database_name, user, password, host, port):
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    con = psycopg2.connect(user=user, password=password, host=host, port=port)
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor  = con.cursor()
    sqlCreateDatabase = f"drop database {database_name};"
    cursor.execute(sqlCreateDatabase)


def create_database(database_name, user, password, host, port):
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    con = psycopg2.connect(user=user, password=password, host=host, port=port)
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor  = con.cursor()
    sqlCreateDatabase = f"create database {database_name};"
    cursor.execute(sqlCreateDatabase)

def get_engine_and_model(database_name, user=None, password=None, host=None, port=None, table_name='osm_images', train=True):
    if train:
        if user is None or password is None or host is None or port is None:
            raise ValueError('For training postgres credentials has to be set. See config.toml for infos!')
        uri = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}'
    else:
        database_path = pathlib.Path(database_name).absolute().with_suffix('.db')
        uri = f'sqlite+pysqlite:///{database_path}'
        print(f'{uri=}')
        
    engine = create_engine(uri)

    base = declarative_base()

    if not train:
        class Routes(base):
            __tablename__ = 'routes'
            id = Column(Integer, primary_key=True)
            path = Column(String)
            dataset = Column(String)
            route_type = Column(String)
            gpx_file = Column(LargeBinary)

    class OSMImages(base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        if train:
            origin = Column(String)
            dataset = Column(String)
            route_type = Column(String)
        else:
            origin = Column(Integer, ForeignKey('routes.id'))
        track_id = Column(Integer)
        segment_id = Column(Integer)
        idx = Column(Integer)
        zoom = Column(Integer)
        p_0_lat = Column(Float)
        p_0_long = Column(Float)
        p_1_lat = Column(Float)
        p_1_long = Column(Float)
        width = Column(Integer)
        height = Column(Integer)
        show_route = Column(Boolean)
        image = Column(LargeBinary)

    create_models = partial(base.metadata.create_all, engine, checkfirst=True)

    if train:
        try:
            create_models()
        except OperationalError:
            create_database(database_name, user, password, host, port)
            create_models()

        return engine, OSMImages
    else:
        create_models()
        return engine, Routes, OSMImages


def set_seed(session, seed, max_val=1177314959, min_val=0):
    seed_f = seed/max_val
    if seed_f > max_val or seed_f < min_val:
        raise ValueError(f'Seed has to be between {min_val} and {max_val}.')
    sql = text('select setseed({0});'.format(seed_f))  # save the SEED_VAL per user/visit
    session.execute(sql)



def postgres_generator(Session, model, offset=0, total_limit=None, entries_per_query=1000, callback=None):
    session = Session()
    query = session.query(model.image)

    n_results = 0
    if not callable(callback):
        callback = None
        
    while True:
        if total_limit:
            if offset >= total_limit:
                break
            elif offset + entries_per_query > total_limit:
                entries_per_query = total_limit - offset
        result = query.limit(entries_per_query).offset(offset).all()
        n_results = len(result)
        for r in result:
            if callback is not None:
                yield callback(r)
            else:
                yield r
        if n_results < entries_per_query:
            break
        offset += entries_per_query     


def _postgres_generator_rnd(query, offset, total_limit, entries_per_query, seed=None, callback=None):
    if seed is not None:
        set_seed(query.session, seed)
    n_results = 0
    if not callable(callback):
        callback = None
        
    while True:
        if total_limit:
            if offset >= total_limit:
                break
            elif offset + entries_per_query > total_limit:
                entries_per_query = total_limit - offset
        result = query.order_by(func.random()).limit(entries_per_query).offset(offset).all()
        n_results = len(result)
        for r in result:
            if callback is not None:
                yield callback(r)
            else:
                yield r
        if n_results < entries_per_query:
            break
        offset += entries_per_query     


def generator_from_query_rnd_order(query, train_test_split=0.3, test=False, entries_per_query=1000, seed=None, return_entries=False, callback=None):
    entries = query.count()
    test_offset = int(entries * train_test_split + 0.5)
    if test:
        total_limit = test_offset
        offset = 0
        length = test_offset
    else:
        total_limit = None
        offset = test_offset
        length = entries - test_offset
        
    generator = _postgres_generator_rnd(query=query, 
                                        offset=offset,
                                        total_limit=total_limit,
                                        entries_per_query=entries_per_query,
                                        seed=seed,
                                        callback=callback)
    if return_entries:
        return generator, length
    else:
        return generator
        
