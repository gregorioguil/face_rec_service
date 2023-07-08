from sqlalchemy import create_engine, text

class UserRepository:
  def __init__(self, config):
    print(config)
    connection_string = "mysql+mysqlconnector://root:password@127.0.0.1:3306/face_rec_service"
    self.engine = create_engine(connection_string, echo=True)
    
  def getUserGym(self, id):
    connection = self.engine.connect()
    result = connection.execute(text("select * from face_rec_service.user_gym where id = :id"), dict(id=id))
    user = {}
    
    for row in result.mappings():
      user = {
      "id": id,
      "name": row["name"],
      "expiredAt": row["expired_at"]}
    return user