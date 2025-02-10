from .utils import JSONField, Column, Random
from random import choice
import string

def generate_name(with_surname: str):
    """
    Generate a random name with given surname.
    """
    names = ["Mert", "Ahmet", "Ali", "Ayşe", "Fatma", "Mehmet", "Zeynep"]
    dummy_name = choice(string.ascii_uppercase)
    return f"{dummy_name}. {choice(names)} {with_surname}"

@JSONField(increment_by=1000, ignore_types=True, auto_fit_columns=True)
# increment_by: increment value for auto increment columns (defaulted to 1)
# ignore_types: ignore the types of the columns and use the given values directly (defaulted to False)
# auto_fit_columns: automatically fit the rows with NULL value if has missing columns
class DataModel(object):
    user_id: int = Column(increment=True) # auto increment
    name: str = Column(depends=generate_name, with_surname="Yıldız") # depends on generate_name function
    ssn: int = Column(increment=True) # auto increment
    age: int = Column(not_null=True, depends=Random.setrange(18, 65)) # not null and depends to generate random number between 18 and 65
    price: float = Column(not_null=True) # not null and required
    date: str = Column(depends=Random.date, date_fmt="%d-%m-%Y %H:%M") # depends on Random.date function
    hash: str = Column(depends=Random.hash) # depends on Random.hash function