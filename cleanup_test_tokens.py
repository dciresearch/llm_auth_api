import sys
sys.path.insert(0, "/workdir")

from src.api_database import Database

db = Database("./database/generic.db")

with db.Session() as session:
    from src.api_database import UserAuth, Requests
    users = session.query(UserAuth).filter(UserAuth.user_name.like("test_%")).all()
    count = len(users)
    for u in users:
        session.query(Requests).filter(Requests.user_id == u.id).delete()
        session.delete(u)
    session.commit()
    print(f"Deleted {count} test tokens")
