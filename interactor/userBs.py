from datetime import date

class UserBs:
  def get(self, id):
    return {
      "user_id": id,
      "name": "Angelina jolie",
      "monthly_payment": True,
      "expired_at": "2023-06-28" 
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
      "created_at": date.today()
    }