from interactor.userBs import UserBs

class UserController:
    
  def get(self, id):
    userBs = UserBs()
    return userBs.get(id)
  
  def list(self):
    userBs = UserBs()
    return userBs.list()
  
  def create(self, user):
    userBs = UserBs()
    return userBs.create(user)