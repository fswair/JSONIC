"""
Using examples :pysonfield: with DataModel
"""

from .utils import PYSONIC, FilterItem, RegExp
from .models import DataModel

with PYSONIC(db_path="database", model=DataModel, audit_fix=True, raise_on_validation=False, allow_promotion=True) as conn:
        # audit_fix: if True, it will fix the corrupted data in the database file.
        # raise_on_validation: if True, it will raise an exception when the data is not valid.
        # allow_promotion: if True, it will allow the promotion of the data type.
        # db_path: the path of the database file.
        # model: the model class that will be used to store data.
        cursor = conn.cursor() # get cursor
        cursor.add({
            "price": 100000,
        }, auto_commit=True) # adds data to stack storage and commits data (defaulted)
        
        """
        data = conn.model.where(filter=RegExp(r"64", "age")) # Where object cursor (iterator-like) with age equals to 64
        conn.model.where(filter=RegExp(r"64", "age")).first # First object cursor (container-like) with age equals to 64
        conn.model.where(filter=RegExp(r"64", "age")).next() # Get next iteration of container
        conn.model.where(filter=RegExp(r"64", "age")).prev() # Get previous iteration of container
        
        while data.has_next(): # check if has next data
            print(data.next()) # get next data
        
        while data.has_prev(): # check if has previous data
            print(data.prev()) # get previous data
        
        data.count # get the count of the data
        data.bulk # get data is bulk or not
        data.json # get the data as JSON-string
        
        ...
        ...
        
        conn.fix() # fix the corrupted data in the database file.
        
        file_path = conn.review(save_report=True, no_raise=True) # review the data and save the report to the file. no exception will be raised.
        found_issues = conn.review(no_raise=True) # review the data and return the issues.
        
        conn.query({
            "ssn": 1000
        }) # query the data and get ResultSet object
        
        pyson = conn.db # reach the database handler origin
        
        cursor[lambda age: age == 75] # get the data where age is 75
        cursor[lambda data: data.age == 21 and data.ssn == 3] # get the data where age is 21 and ssn is 3
        cursor[lambda age, ssn, name: age == 21 and ssn == 3 and name == "Ahmet Yıldız"] # get the data where age is 21 or ssn is 3
        cursor[FilterItem(lambda x: x.age == 21)] # get the data where age is 21
        
        """