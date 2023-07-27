import json
from datetime import datetime
from repositories.userRepository import UserRepository

class UserBs:
  def __init__(self):
    self.config = open('config/databases.json')
    self.config = json.load(self.config)

  def get(self, id):
    userRepository = UserRepository(self.config)
    user = userRepository.getUserGym(id)
    today = datetime.today().date()

    monthlyPayment = today <= user["expiredAt"]
    return {
      "user_id": user["id"],
      "name": user["name"],
      "age": user["age"],
      "email": user["email"],
      "phone": user["phone"],
      "document": user["document"],
      "monthly_payment": monthlyPayment,
      "expired_at": user["expiredAt"],
      "address": {
        "number": user["address"]["number"],
        "street": user["address"]["street"],
        "city": user["address"]["city"],
        "cep": user["address"]["cep"],
        "district": user["address"]["district"]
      }
    }
    
  def list(self):
    userRepository = UserRepository(self.config)
    users = userRepository.listUsersGym()
    result = []
    today = datetime.today().date()
    
    for user in users:
      monthlyPayment = today <= user["expiredAt"]
      
      result.append({
        "user_id": user["id"],
        "name": user["name"],
        "monthly_payment": monthlyPayment,
        "expired_at": user["expiredAt"]
      })

    return result
    
  def create(self, user):
    userRepository = UserRepository(self.config)
    userRepository.createUserGym(user)
    return {
      "name": user['name'],
      "created_at": datetime.today().date()
    }