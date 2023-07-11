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

    monthlyPayment = today <= user["expiredAt"].date()
    return {
      "user_id": user["id"],
      "name": user["name"],
      "monthly_payment": monthlyPayment,
      "expired_at": user["expiredAt"]
    }
    
  def list(self):
    userRepository = UserRepository(self.config)
    users = userRepository.listUsersGym()
    result = []
    today = datetime.today().date()
    
    for user in users:
      monthlyPayment = today <= user["expiredAt"].date()
      
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