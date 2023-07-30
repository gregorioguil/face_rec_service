from sqlalchemy import create_engine, text

class UserRepository:
  def __init__(self, config):
    connection_string = "mysql+mysqlconnector://root:password@host.docker.internal:3306/face_rec_service"
    self.engine = create_engine(connection_string, echo=True)
    
  def getUserGym(self, id):
    connection = self.engine.connect()
    result = connection.execute(
      text("select * from face_rec_service.user_gym u inner join face_rec_service.address a on a.id = u.address_id where u.id = :id"), dict(id=id))
    user = {}
    
    for row in result.mappings():
      user = {
      "id": id,
      "name": row["name"],
      "age": row["age"],
      "email": row["email"],
      "phone": row["phone"],
      "document": row["document"],
      "expiredAt": row["expired_at"],
      "address": {
        "number": row["number"],
        "street": row["street"],
        "city": row["city"],
        "cep": row["cep"],
        "district": row["district"]
      }}
    return user
  
  def listUsersGym(self):
    connection = self.engine.connect()
    result = connection.execute(text("select * from face_rec_service.user_gym"))
    users = []
    
    for row in result.mappings():
      users.append({
        "id": row["id"],
        "name": row["name"],
        "expiredAt": row["expired_at"]})
    return users
  
  def createUserGym(self, user):
    connection = self.engine.connect()
    result = connection.execute(
      text("INSERT INTO user_gym (name, age, document, email, phone, expired_at) values (:name, :age, :document, :email, :phone, :expired_at)"), user)
    connection.commit()

