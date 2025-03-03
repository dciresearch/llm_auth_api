import fire
from src.api_database import Database


db_path = "./database/generic.db"
api_db = Database(db_path)

if __name__ == '__main__':
    fire.Fire(api_db)
