from sqlalchemy import Column, Integer, String, Boolean, DateTime

class UserGym(Base):
  __tablename__ = "user_gym"
  
  id = Column(Integer, primary_key=True)
  name = Column(String)
  age = Column(Integer)
  document = Column(String)
  monthly_payment = Column(Boolean)
  expired_at = Column(DateTime)
  created_at = Column(DateTime)
  updated_at = Column(DateTime)


  
  