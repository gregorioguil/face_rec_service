import json
from datetime import datetime
from repositories.userRepository import UserRepository

class UserBs:
  def get(self, id):
    config = open('config/databases.json')
    config = json.load(config)
    userRepository = UserRepository(config)
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
    return [{
      "user_id": 1,
      "name": "Angelina jolie",
      "monthly_payment": True,
      "expired_at": "2023-06-28" 
    },{
      "user_id": 2,
      "name": "Brad Pitt",
      "monthly_payment": True,
      "expired_at": "2023-06-22" 
    },{
      "user_id": 3,
      "name": "Mohamed Ali",
      "monthly_payment": True,
      "expired_at": "2023-02-22" 
    }]
    
  def create(self, user):
    return {
      "name": user['name'],
      "id": 4,
      "created_at": datetime.today().date()
    }