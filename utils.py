import json
import random
import re
import sqlite3, tqdm
import traceback
from functools import partial
from pathlib import Path
from typing import (
    Any,        Dict,        List,
    Union,      Type,        AnyStr,
    TextIO,     Callable,    Generic, 
    Iterator,   SupportsInt,         
    NoReturn, overload
)


import numpy as np
import pandas as pd
from pysondb import db
import pysondb

from .exceptions import CursorInvalid, DataCorrupted, KeyAlreadyExists, KeyNotExists, KeyNotFound, NotNullViolation, NotUniqueViolation
from .utiltypes import Date, Object, Random, Self, T, Missing, NULL

class RegExp(object):
    """RegExp object for pattern-based filtering in JSON data.

    A utility class that creates a filter function based on regular expression pattern matching
    against a specified field in dictionary-like data structures.

    Args:
        pattern (str): The regex pattern to match against field values
        field (str): Name of the field to match against in data dictionaries  
        flags (re.RegexFlag, optional): Regex flags to use. Defaults to re.I (case-insensitive)

    Attributes:
        pattern (str): The stored regex pattern
        field (str): Field name to match against
        flags (re.RegexFlag): Regex flags used for matching
        filter (callable): Lambda function that performs pattern matching

    Example:
        >>> reg = RegExp("test", "name")
        >>> reg.filter({"name": "test123"})  # Returns True
        >>> reg.filter({"name": "abc"})      # Returns False

    Notes:
        - Returns True if pattern matches field value, False otherwise
        - Non-string values converted to strings before matching 
        - Empty/missing field values return False
        - Pattern must match anywhere in field value (not full match)
        - Uses re.search() to check for matches
    """
    def __init__(self, pattern: str, field: str, flags: re.RegexFlag = re.I) -> Self:
        assert (pattern), "Pattern and Content must be filled."
        self.pattern = pattern
        self.field = field
        self.flags = flags
        self.filter = lambda data: bool(data.get(field) and re.search(pattern, str(data.get(field)) if not isinstance(data.get(field), (str, bytes)) else data.get(field), flags=flags))

class CRUD(object):
    """CRUD operations for managing ResultSets.

    ## Description
    This class provides Create, Read, Update and Delete operations for managing data
    in a PYSONIC database connection.

    ## Attributes
        db: The database connection from PYSONIC instance
        data (List[Dict]): List of dictionaries containing the data to operate on

    ## Methods
        delete(_id: int = None, all: bool = True) -> bool:
            Deletes record(s) by ID or all records
        
        delete_by_query(query: Dict) -> bool:
            Deletes records matching the provided query
        
        delete_by_filter(predicate: Callable[[Dict], bool]) -> bool:
            Deletes records that satisfy the filter predicate
        
        update(id: int, new_data: Dict) -> bool:
            Updates a single record by ID with new data
        
        update_all(new_data: Dict) -> bool:
            Updates all records with new data
        
        update_by_query(query: Dict, new_data: Dict) -> bool:
            Updates records matching query with new data
        
        update_by_filter(predicate: Callable[[Dict], bool], new_data: Dict) -> bool:
            Updates records that satisfy filter predicate with new data

    ## Raises
        CursorInvalid: If no data is provided during initialization
        ValueError: If neither _id nor all=True is provided for delete operation
    """
    
    def __init__(self, conn: "PYSONIC", data: List[Dict] = None) -> None:
        self.db = conn.db
        self.data = data if isinstance(data, list) else [data]
    
    def delete(self, id: int = None, all: bool = True) -> bool:
        """
        Deletes one or all records from the database.

        ## Args:
            id (int, optional): The ID of the record to delete. Defaults to None.
            all (bool, optional): Whether to delete all records. Defaults to True.

        ## Returns:
            bool: True if deletion was successful, False otherwise.
            For single record deletion, returns result of deleteById operation.
            For all records deletion, returns True if no records remain.

        ## Raises:
            ValueError: If neither id is provided nor all=True.

        ## Examples:
            >>> db.delete(id=1)  # Deletes record with ID 1
            >>> db.delete(all=True)  # Deletes all records
        """

        if not all and not id:
            raise ValueError("You have to pass _id value or set all=True.")
        if id:
            return self.db.deleteById(id)
        elif all:
            self.db.deleteAll()
            return not any(self.db.getAll())
    
    def delete_by_query(self, query: Dict) -> bool:
        """
        Deletes all documents matching the specified query from the database.

        ## Args:
            query (Dict): Dictionary containing query parameters to match documents for deletion

        ## Returns:
            bool: True if all matching documents were successfully deleted, False otherwise

        ## Example:
            >>> db = Database()
            >>> query = {"status": "inactive"}
            >>> db.delete_by_query(query)
            True
        """
        for data in self.db.getByQuery(query):
            self.db.deleteById(data["_id"])
        return not any(self.db.getByQuery(query))
    
    def delete_by_filter(self, predicate: Callable[[Dict, Object], bool]) -> bool:
        """
        Deletes records from the database that match the given predicate function.

        ## Args:
            predicate (Callable[[Dict, Object], bool]): A function that takes a dictionary as input and returns
            a boolean indicating whether the record should be deleted.

        ## Returns:
            bool: True if all matching records were successfully deleted, False if some matching records
            remain in the database after deletion attempt.

        ## Example:
            # Delete all records where age > 30
            success = db.delete_by_filter(lambda x: x['age'] > 30) 
        """
        for data in filter(predicate, map(Object, self.db.getAll())):
            self.db.deleteById(data["_id"])
        return not any(filter(predicate, self.db.getAll()))
    
    def update(self, new_data: Dict) -> bool:
        """
        Updates an existing record in the database with the provided data.

        ## Args:
            id (int): The unique identifier of the record to update.
            new_data (Dict): A dictionary containing the new data to update the record with.

        ## Returns:
            bool: True if the update was successful, False otherwise.
        """
        return self.db.updateById(self.data["_id"], new_data)
    
    def update_all(self, new_data: Dict) -> bool:
        """
        Updates all documents in the database with the provided new data.

        Args:
            new_data (Dict): A dictionary containing the fields to be updated and their new values.

        Returns:
            bool: True if the update operation was successful, False otherwise.

        Example:
            >>> crud.update_all({"status": "inactive", "modified_date": "2023-01-01"})
            True

        Notes:
            - This method will apply the update to all documents in the database.
            - Use with caution as it affects all records.
            - The update operation is atomic.
        """
        return self.db.update({}, new_data)
    
    def update_by_query(self, query: Dict, new_data: Dict) -> bool:
        """
        Updates records that match the given query with new data.

        ## Args:
            query (Dict): Dictionary containing query parameters to match records
            new_data (Dict): Dictionary containing the new data to update matched records

        ## Returns:
            bool: Returns the result of update operation

        ## Example:
            >>> crud.update_by_query({"status": "inactive"}, {"active": True})
            True
        """
        return self.db.updateByQuery(query, new_data)
    
    def update_by_filter(self, predicate: Callable[[Dict], bool], new_data: Dict) -> bool:
        """
        Updates records in the database that match a given predicate with new data.

        ## Args:
            predicate (Callable[[Dict], bool]): A function that takes a dictionary and returns a boolean.
                Used to filter which records should be updated.
            new_data (Dict): The new data to update matching records with.

        ## Returns:
            bool: Always returns True after completing updates.

        ## Example:
            >>> db.update_by_filter(lambda x: x['age'] > 21, {'status': 'adult'})
            True
        """
        for data in filter(predicate, self.data):
            self.db.updateById(data["_id"], new_data)
        return True
    
    def insert(self):
        """
        Inserts a new record into the database.

        ## Args:
            data (Dict): The data to insert as a dictionary.

        ## Returns:
            bool: True if the insertion was successful, False otherwise.
        """
        for data in self.data:
            self.db.add(data)


class ResultSet(CRUD, Generic[T]):
    def __init__(self, data: Dict | List[Dict], conn: "PYSONIC" = None) -> None:
        if data and isinstance(data, list) and isinstance(data[0], (Column, Object)):
            self.data = data
        else:
            self.data = list(map(Object, data if isinstance(data, list) else [data]))
        self._loc = 0
        if conn:
            super().__init__(conn, self.data)
    
    def shuffle(self):
        if self.bulk:
            random.shuffle(self.data)
    
    def reverse(self):
        if self.bulk:
            self.data.reverse()
        
    @property    
    def empty(self):
        return not bool(self.data)
    
    @property
    def bulk(self):
        return bool(self.data and isinstance(self.data, list))
    
    def __repr__(self) -> str:
        return repr(self.data)
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def next(self) -> Union[Object[Dict], StopIteration]:
        if self.has_next():
            data = self.data[self._loc]
            self._loc += 1
            return data
        raise CursorInvalid()
    
    def prev(self) -> Union[Object[Dict], StopIteration]:
        if self.has_prev():
            data = self.data[self._loc]
            self._loc -= 1
            return data
        raise CursorInvalid()
    
    def has_next(self) -> bool:
        return self._loc < len(self.data)
    
    def has_prev(self) -> bool:
        return self._loc > 0
    
    def __getitem__(self, item: int):
        return self.data[item]
    
    def __setitem__(self, *args, **kwargs) -> NoReturn:
        raise AttributeError("You can't set value to ResultSet because it's read-only.")


class UniqueConstraintIssues(ResultSet):
    """A class that encapsulates a list of primary key issues."""
    pass
    
class TypeIssues(ResultSet):
    """A class that encapsulates a list of type issues."""
    pass
    
class RequiredFieldIssues(ResultSet):
    """A class that encapsulates a list of required field issues."""
    pass

class BaseIssues(ResultSet):
    """A class that encapsulates a list of issues."""
    pass


class Where(ResultSet):
    def __init__(self, conn:  "PYSONIC", predicate: Callable | RegExp = lambda x: bool(x) == True):
        """
        Creates a filtered view of data from a PYSONIC connection.

        ## Args:
            conn (PYSONIC): The database connection
            predicate (Callable | RegExp, optional): Filter function or RegExp object. 
            Defaults to lambda x: bool(x) == True

        ## Raises:
            AssertionError: If filter is not callable or RegExp

        ## Attributes:
            filter: The filter function/RegExp
            conn: Database connection
            datas: Raw data from DB
            data: Filtered data

        ## Example:
            ```python
            # Filter by callable
            where = Where(conn, lambda x: x['age'] > 21)
            
            # Filter by regex
            pattern = RegExp(r"test", "name") 
            where = Where(conn, pattern)
            ```

        ## Notes:
            - Callable filters used directly with filter()
            - RegExp filters use RegExp.filter() method
        """

        self.filter = predicate
        self.conn = conn
        self.datas = self.conn.fetch()
        assert self.filter == bool or callable(self.filter) or (isinstance(self.filter, RegExp)), \
        "Filter must be callable or RegExp instance."
        if callable(self.filter):
            self.data = list(filter(self.filter, self.datas))
        else:
            self.data = list(filter(self.filter.filter, self.datas))
        super().__init__(self.data)
        
    def __str__(self) -> str:
        """Returns a string representation of the Where object."""
        return str(self.data)
    
    def __repr__(self) -> str:
        """Returns a string representation of the Where object."""
        return f'<{self.__class__.__name__} filter={self.filter.__name__ if callable(self.filter) else self.filter}>'
    
    @property
    def count(self) -> int:
        """Returns the number of items in the filtered data."""
        return len(self.data)
    
    @property
    def result(self) -> ResultSet[np.ndarray]:
        """Returns the filtered data as a ResultSet."""
        return ResultSet(self.data, self.conn)
    
    @property
    def first(self) -> Dict[Union[SupportsInt, AnyStr], Union[SupportsInt, AnyStr, List, Dict]]:
        """Returns the first item in the filtered data."""
        return self.data[0]
    
    @property
    def all(self):
        """Returns all items in the filtered data."""
        return ResultSet(self.data, self.conn)
    
    def sliced(self, __slice: slice) -> ResultSet[Object]:
        """Returns a sliced view of the filtered data."""
        return ResultSet(self.data[__slice], self.conn)
    
    def to_model(self) -> ResultSet[Object]:
        """Converts the filtered data to a list of model instances."""
        return ResultSet(list(map(Object, self.data)), self.conn)
    
    def to_dataframe(self, print_to_stdout: bool = False) -> pd.DataFrame:
        """Converts the filtered data to a pandas DataFrame."""
        data = self.data.tolist()
        df = pd.DataFrame(data)
        if print_to_stdout:
            print(df)
        return df
    
    @property
    def dataframe(self):
        """Returns the filtered data as a pandas DataFrame."""
        return self.to_dataframe()
    
    @property
    def json(self):
        """Returns the filtered data as a JSON string."""
        return self.dataframe.to_json()
    
    @property
    def csv(self):
        """Returns the filtered data as a CSV string."""
        return self.dataframe.to_csv()
    
    def to_excel(self, _excel_fpath: Union[str, Path]):
        """Writes the filtered data to an Excel file."""
        assert (_excel_fpath), "This path value does not belong a valid path."
        return self.dataframe.to_excel(excel_writer=_excel_fpath)
    
    def to_sql(self, __name: str, __conn: sqlite3.Connection):
        """Writes the filtered data to an SQLite database."""
        return self.dataframe.to_sql(__name, __conn)
    
    @property
    def markdown(self):
        """Returns the filtered data as a Markdown table."""
        return self.dataframe.to_markdown()
    
    @property
    def html(self):
        """Returns the filtered data as an HTML table."""
        return self.dataframe.to_html()
    
    def to_json(self, path: Union[str, Path] = ..., buffer: TextIO = ...):
        """Writes the filtered data to a JSON file or buffer."""
        return self.dataframe.to_json(path or buffer)
    
    def to_xml(self, path: Union[str, Path] = ..., buffer: TextIO = ...):
        """Writes the filtered data to an XML file or buffer."""
        return self.dataframe.to_xml(path or buffer)
    
    def to_numpy(self, predicate: Callable = None) -> np.ndarray:
        """Converts the filtered data to a NumPy array."""
        array = self.dataframe.to_numpy(np.dtype)
        if predicate:
            array = filter(predicate, array)
        return ResultSet(np.fromiter(array, np.dtype), self.conn)

class JSONField:
    """
    A decorator class that transforms Python classes into JSON-based data models.
    
    This class provides functionality to create database-like structures using JSON files,
    with support for schema validation, type checking, and automatic column fitting.

    ## Attributes
        model_schema (List["Column"]): Schema columns defining the model structure
        __jsonfield_version__ (str): JSONField implementation version
        _ignore_types (bool): Whether to skip type validation
        datafields (List[str]): List of data field names

    ## Args
        increment_by (int): Value to increment auto-incrementing fields by. Defaults to 1
        ignore_types (bool): Skip type validation if True. Defaults to False
        auto_fit_columns (bool): Auto-adjust column properties. Merges NULL value to missing columns of each row if has. Defaults to False 
        **options: Additional model configuration options

    ## Example
        ```python
        @JSONField(increment_by=1000) # Increments from 1000 by one
        class User:
            id = Column(int, primary_key=True) 
            name: str = Column()  # Type from annotation
            age = Column(int)     # Type from Column
            created = Column(depends=Date.now) # Auto timestamp
        ```

    ## Features
        - Schema validation with type checking
        - Primary key and unique constraints
        - Auto-incrementing fields
        - Default values and generators
        - JSON file storage backend
        - Query filtering with where() method
        - Flexible type validation options
        - Column auto-fitting capability

    ## Methods
        __call__: Initializes model with decorator config
        schema: Generates database schema from model
        where: Creates filtered data view

    ## Notes
        - Uses JSON files for persistence
        - Supports type annotations and Column definitions
        - Validates schema on write operations
        - Allows disabling type checks
        - Can auto-fit columns to data
    """
    model_schema: List["Column"]
    __jsonfield_version__ = "1.0.0"
    _ignore_types: bool = False
    datafields: List[str]
    
    def __init__(self, increment_by: int = 1, ignore_types: bool = False, 
                 auto_fit_columns: bool = False, **options) -> None:
        self.ignore_types = ignore_types
        self._increment_by = increment_by
        self._auto_fit_columns = auto_fit_columns
        self._options = options
        self.model = NULL
        
    def __call__(self, instance: Callable) -> object:
        """Initializes the JSONField class with the provided model instance."""
        datafields = sorted(attr for attr in dir(instance) if not attr.startswith('_'))
        self.model = instance
        self.model._auto_fit_columns = self._auto_fit_columns
        
        for key, value in self._options.items():
            setattr(instance, f"_{key}", value)

        core_attributes = {
            'getdb': lambda **options: PYSONIC(model=instance, **options),
            'model_schema': self.schema(instance, datafields),
            'where': lambda filter=bool: self.where(self.model._conn, filter),
            '_table': instance.__name__.lower().removesuffix("model"),
            '_ignore_types': self.ignore_types,
            'datafields': datafields,
            '__jsonfield_version__': self.__jsonfield_version__
        }
        
        for attr_name, attr_value in core_attributes.items():
            setattr(instance, attr_name, attr_value)
            
        return instance
    
    def schema(self, instance: Callable, datafields: List["Column"]) -> List["Column"]:
        """
        Generates a schema for a database model by processing its datafields.

        This method maps model fields to database columns, setting their names and types
        if not already defined. It validates that all fields have type information
        either from Column definition or model class annotations.

        Args:
            instance (Callable): The model class to generate schema for
            datafields (List[str]): List of field names to process

        Returns:
            ResultSet[Column]: List of processed Column objects with names and types

        Raises:
            ValueError: If field types are not specified and ignore_types=False

        Example:
            ```python
            @JSONField(increment_by=1000) # Increments from 1000 by one
            class User:
                id = Column(int, primary_key=True) 
                name: str = Column()  # Type from annotation
                age = Column(int)     # Type from Column
            schema = User.model_schema
            ```
        """
        data: List[Column] = list()
        invalid_fields = []
        for field in datafields:
            column: Column = instance.__dict__[field]
            column.name = field
            if not column.typer:
                column.typer = self.model.__annotations__.get(field)
                if not column.typer:
                    if not self.ignore_types:
                        invalid_fields.append(field)
            data.append(column)
        if invalid_fields:
            raise ValueError(f"Type of fields '{', '.join(invalid_fields)}' is not specified. Please annotate fields or make ignore_types=True in JSONField decorator.")
        return ResultSet(data)
    
    @staticmethod
    def where(conn: "PYSONIC" = ..., filter: Callable = bool) -> Where:
        """Creates a filtered view of data from a PYSONIC connection."""
        return Where(conn, **{"predicate": filter} if filter else {})

class PYSONIC:
    table: str
    def __init__(self, db_path: Union[str, Path] = ".", model: JSONField = None, reinit: bool = False, audit_fix: bool = False, commit_on_exit: bool = False, raise_on_validation: bool = False, allow_promotion: bool = False) -> None:
        """Initialize the database handler.

        ## Args:
            db_path (Union[str, Path]): Path to the database directory. Defaults to current directory.
            model (JSONField): JSONField model class to use. Defaults to None.
            audit_fix (bool): Whether to fix corrupted JSON data with backups. Defaults to False.
            commit_on_exit (bool): Whether to commit changes when exiting context. Defaults to False. 
            raise_on_validation (bool): Whether to raise exceptions during validation. Defaults to False.
            allow_promotion (bool): Whether to allow type promotion during validation. Defaults to False.
        
        ## Attributes:
            fpath (Path): Full path to the JSON database file
            commit_on_exit (bool): Whether to auto-commit on context exit
            _audit_fix (bool): Whether to enable audit fixing
            no_raise (bool): Whether to suppress validation exceptions 
            allow_promotion (bool): Whether to allow type promotion
            db (db): PysonDB database instance
            model (JSONField): Associated model class
            stack (Stack): Data operation stack

        ## Example:
            >>> model = UserModel()
            >>> db = PYSONIC("data", model, audit_fix=True)
            >>> with db:
            ...     db.write({"name": "test"})

        ## Notes:
            - Creates/loads JSON file named after model in db_path
            - audit_fix=True fixes corrupted files with backups
            - commit_on_exit=True auto-commits on context exit
            - raise_on_validation=False suppresses validation errors
            - allow_promotion=True enables type conversion
        """
        self.parent_path = Path(db_path)
        self.fpath = self.parent_path / Path(model.__name__.lower().removesuffix("model") + ".json")
        self.parent_path.mkdir(parents=True, exist_ok=True)
        self.commit_on_exit = commit_on_exit
        self._audit_fix = audit_fix
        self.no_raise = not raise_on_validation
        self.allow_promotion = allow_promotion
        self.db = db.getDb(str(self.fpath))
        self.db._id_fieldname = "_id"
        self.model = model
        self.model._conn = self
        self.stack = Stack(self, dict())
        setattr(self, "_data", self.stack.data)
    
    def cursor(self):
        """
        Returns the current cursor to handle database operations.
        """
        return self.stack

    def fix(self) -> None:
        """
        Attempts to fix corrupted JSON data by creating a backup and resetting the database.

        This method tries to read all data from the database. If a JSONDecodeError occurs,
        indicating corrupted data, it will:
        1. Create a backup of the corrupted file if audit_fix is True
        2. Delete all data in the database
        3. If self._audit_fix is False, raise a DataCorrupted exception

        Raises
        ------
        DataCorrupted
            If the JSON data is corrupted and audit_fix is False

        Notes
        -----
        The backup file is created in a '.backup' directory with the same name as the original file
        """
        try:
            self.db.getAll()
        except json.decoder.JSONDecodeError as exc:
            if self._audit_fix:
                tempdir = Path(".backup")
                tempdir.mkdir(exist_ok=True)
                backup_path = tempdir / f"{self.fpath.stem}.json"
                backup_path.write_bytes(self.fpath.read_bytes())
                
                return self.db.deleteAll()
            raise DataCorrupted(exc)
    
    def review(self, save_report: bool = False, report_path: str | Path = "reviews", no_raise: bool = None) -> Union["Issues", Path]:
        """Validates all data in the database against the model schema and generates a report.

        ## Args:
            save_report (bool, optional): Whether to save the report to a file. Defaults to False.
            report_path (str | Path, optional): Path to save the report file. Defaults to "reviews".
            no_raise (bool | None, optional): Whether to suppress validation exceptions. Defaults to None.

        ## Returns:
            Union["Issues", Path]: Either:
            - Issues object containing validation results if save_report=False
            - Path to saved report file if save_report=True

        ## Raises:
            Various validation errors if no_raise=False:
            - KeyNotFound: Missing required fields
            - TypeError: Invalid field types
            - NotNullViolation: Null values in non-null fields
            - KeyAlreadyExists: Duplicate unique values

        ## Example:
            >>> db = PYSONIC("test.db", MyModel)
            >>> report = db.review()  # Get Issues object
            >>> report_path = db.review(save_report=True)  # Save to file
            >>> report_path = db.review(save_report=True, no_raise=True)  # Save to file without raising exceptions
            >>> issues = db.review(no_raise=True)  # Get Issues object without raising exceptions
        """
        model = self.model
        no_raise = no_raise if no_raise is not None else self.no_raise
        review_data = {
            "unique_constraint_issues": [],
            "type_issues": [], 
            "required_field_issues": [],
            "notnull_issues": []
        }
        
        pkey_fields = [field for field in model.model_schema if field.unique]
        
        for data in tqdm.tqdm(self.db.getAll(), "schema is validating..."):
            temp = Stack(self, data)
            record_id = data.get("_id")
            null_fields = temp._process_null_fields(model.model_schema, no_raise=no_raise)
            review_data["notnull_issues"].extend(null_fields)
            if not self.model._ignore_types:
                type_validation = temp._validate_types(model.model_schema, no_raise=no_raise)
                if type_validation is not True:
                    validate_issues = [{
                        "id": record_id,
                        "info": (issue_info:=tval.message),
                        issue_info[:issue_info.find(":")].lower(): True,
                        "message": f"Field was empty. Type validation did not perform properly." if tval['missing'] else f"Type validation failed for '{tval.field}' field while expected '{tval.expected}' but found '{tval.found}'."
                    } for tval in type_validation]
                    review_data["type_issues"].extend(validate_issues)
            required_fields = temp._validate_required_fields(no_raise=no_raise)
            if isinstance(required_fields, (list, tuple)):
                review_data["required_field_issues"].append({
                    "id": record_id,
                    "message": f"Required fields '{', '.join(required_fields)}' is missing."
                })
            for pkey in pkey_fields:
                pkey_value = data.get(pkey.name)
                if pkey_value is not None and not temp._validate_unique_constraints(pkey, pkey_value, no_raise=no_raise).status:
                    review_data["unique_constraint_issues"].append({
                        "id": record_id,
                        "message": f"Unique constraint for '{pkey.name}' already exists."
                    })
        if save_report:
            current_time = int(Date.timestamp())
            time_s = Date.now("%Y_%m_%d_%H_%M_%S")
            report_path = Path(report_path)
            report_path.mkdir(exist_ok=True, parents=True)
            file = f"review-{time_s}-{current_time}.json"
            with open(report_path / file, "w", encoding="utf-8") as f:
                json.dump(review_data, f, indent=4, ensure_ascii=False)
                return Path(report_path / file)
        return Issues(review_data)
    def write(self, data: Dict) -> Dict:
        """Writes data to the database."""
        self.fix()
        try:
            self.db.add(data)
        except pysondb.errors.db_errors.SchemaError:
            if not self.model._auto_fit_columns:
                data_keys = sorted([k for k in data.keys()])
                model_keys = sorted([k.name for k in self.model.model_schema])
                missing_keys = [key for key in model_keys if key not in data_keys]
                extra_keys = [key for key in data_keys if key not in model_keys]
                if missing_keys:
                    raise KeyNotFound(', '.join(missing_keys), f"Please fill them.")
                elif extra_keys:
                    raise KeyNotExists(extra_keys)
            else:
                rows = self.fetch()
                rows.data.append(data)
                new_rows = []
                cols = [col.name for col in self.model.model_schema]
                for row in rows:
                    new_row = {self.db.id_fieldname: row[self.db.id_fieldname]}
                    new_row.update(dict(sorted({col: row.get(col) for col in cols}.items(), key=lambda x: x[0])))
                    new_rows.append(new_row)
            data = {"data": new_rows}
            with open(self.fpath.with_name(f"{self.model._table}_fixed_{Date.now(date_fmt='%Y_%m_%d_%H_%M')}.json"), "w") as f:
                f.write(json.dumps(data, indent=4))
        self.stack.data.clear()    
    def fetch(self) -> ResultSet[Object]:
        """Fetches all data from the database."""
        self.fix()
        return ResultSet(self.db.getAll(), self)

    def query(self, query: Dict = {}) -> ResultSet[Object]:
        """Queries the database with the specified query."""
        self.fix()
        return ResultSet(self.db.getByQuery(query), self)

    def __enter__(self) -> "PYSONIC":
        """Enters the context manager."""
        self.fix()
        assert (self.model is not None), Exception("You have to pass JSONField model.")
        return self
    def __exit__(self, exc_type: Union[Any, None], exc_val: Union[Any, None], traceback: Union[Any, None]):
        """Exits the context manager."""
        if self.commit_on_exit: self.stack.commit()
        pass
    
    async def __aenter__(self) -> "PYSONIC":
        """Enters the async context manager."""
        self.fix()
        assert (self.model is not None), Exception("You have to pass JSONField model.")
        return self
    async def __aexit__(self, exc_type: Union[Any, None], exc_val: Union[Any, None], traceback: Union[Any, None]):
        """Exits the async context manager."""
        if self.commit_on_exit: self.stack.commit()
        pass
    

class Column(object):
    """
    A data model representing a column field in a database-like structure.

    This class defines properties of a column including its type, default value, dependencies,
    uniqueness constraints, nullability, and whether it's a primary key or auto-incrementing.

    ## Attributes:
        typer (Any | None): The type of the column. Defaults to None. Will use model class type annotation if not specified.
        default (Any | None): Default value for the column. Defaults to None. 
        depends (Any | Callable): Dependencies or callable mock generators. Cannot be used with default. Defaults to None.
        name (str | None): Name of the column. Defaults to None.
        not_null (bool): Whether the column can contain null values. Defaults to False.
        increment (bool): Whether the column auto-increments. Cannot be primary key or unique. Defaults to False.
        primary_key (bool): Whether this is a primary key column. Implies unique and not_null. Defaults to False.
        unique (bool): Whether values must be unique. Cannot be used with not_null. Defaults to False.
        kw (Dict): Additional keyword arguments passed to callable in depends field.

    ## Examples:
        >>> Column(str, default="test")  # Simple column with default
        >>> Column(int, increment=True)  # Auto-incrementing column
        >>> Column(str, primary_key=True)  # Primary key column
        >>> Column(str, unique=True)  # Unique column
        >>> Column(str, not_null=True)  # Required column
        >>> Column(str, depends=Random.hash)  # Random hash generator
        >>> Column(depends=lambda **kw: kw["value"], value=42)  # Custom generator

    ## Notes:
        - Cannot use both default and depends
        - Cannot use both unique and not_null (use primary_key instead)  
        - Cannot use increment with primary_key or unique
        - Primary key implies both unique and not_null constraints
    """
    def __init__(self, typer: Any | None = None, default: Any | None = None, 
                 depends: Any | Callable[[Any | None], Any] = None,
                 name: str | None = None, not_null: bool = False, increment: bool = False,
                 primary_key: bool = False, unique: bool = False, **kw: Dict) -> None:
        self.typer = typer
        self.default = default 
        self.depends = depends
        self.name = name
        self.kw = kw
        self.unique = unique
        self.not_null = not_null
        self.increment = increment
        self.primary_key = primary_key
        
        if default:
            self.depends = self.depends or partial(lambda x: x, default)
            self.default = NULL
        
        if depends and default:
            raise ValueError("You can't use both depends and default at the same time.")
        
        if unique and not_null:
            raise ValueError("Set primary_key=True for making unique and not-null field.")
        
        if increment and (primary_key or unique):
            raise ValueError("Increment field can't be primary key or unique. Because it's auto-incrementing.")
        
        if self.primary_key:
            
            self.unique = True
            self.not_null = True
            
    def __repr__(self) -> str:
        """Returns a string representation of the Column object."""
        attrs = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"Column({', '.join(attrs)})"


class FilterItem(object):
    """A class that wraps a filter function.

    This class is used to encapsulate a filter function that can be applied to data processing
    or validation operations.

    ## Args:
        filter (Callable): A callable function that serves as a filter.

    ## Returns:
        Self: Returns an instance of FilterItem.

    ## Example:
        >>> def my_filter(x):
        ...     return x > 0
        >>> filter_item = FilterItem(my_filter)
    """
    def __init__(self, filter: Callable) -> Self:
        self.filter = filter

class Issues(object):
    """A class that encapsulates a list of issues.

    This class is used to store a list of issues that can be used for logging or reporting.

    ## Args:
        issues (List[str]): A list of issues to store.

    ## Returns:
        Self: Returns an instance of Issues.

    ## Example:
        >>> issues = Issues({pkey: [], type: [], required: []})
    """
    
    def __init__(self, issues: Dict) -> Self:
        self.unique_constraint_issues = UniqueConstraintIssues(issues.get("unique_constraint_issues", []))
        self.type_issues = TypeIssues(issues.get("type_issues", []))
        self.required_field_issues = RequiredFieldIssues(issues.get("required_field_issues", []))
        
        for k, v in {k: v for k,v in issues.items() if not hasattr(self, k)}.items():
            setattr(self, k, BaseIssues(v))

class Stack(object):
    """
    A stack implementation for handling JSON-like data operations with database connections.

    This class provides functionality for managing data operations, including validation,
    commitment, and querying of data according to a specified schema.

    ## Attributes:
        __excluded_slots__ (tuple): Tuple of strings representing excluded attribute names
        conn (PYSONIC): Connection object for database operations 
        data (Dict[str, Any]): Dictionary containing the stack's data

    ## Methods:
        current_size(): Returns current number of records in database
        is_convertible(obj, type, allow_promotion): Checks if object can be converted to specified type
        add(data, auto_commit): Adds new data to stack and optionally commits
        commit(): Validates and commits data according to schema

    ## Parameters: 
        conn (PYSONIC): Database connection object
        data (Dict): Initial data dictionary

    ## Raises:
        KeyAlreadyExists: When attempting to insert duplicate unique value
        KeyNotFound: When required fields are missing 
        TypeError: When data types don't match schema
        NotNullViolation: When null value in not-null field
        NotUniqueViolation: When unique constraint violated

    ## Example:
        >>> stack = Stack(connection, {"field": "value"})
        >>> stack["field"] = "new value" / same
        >>> stack.add({"name": "test"})  \ same
        >>> stack.commit()
    """
    __excluded_slots__ = ("model_schema", "where", "datafields", "db", "getdb")
    def __init__(self, conn: "PYSONIC", data: Dict) -> None:
        self.conn = conn
        self.data: Dict[str, Any] = Object(data)
    
    def current_size(self):
        """Returns the current number of records in the database."""
        return len(self.conn.fetch())
    
    @staticmethod
    def is_convertible(__obj: object, __type: Type, allow_promotion: bool = False) -> bool:
        """Checks if an object can be converted to a specified type. If allow_promotion is True, it will attempt to convert the object to the specified type and ignres type equality."""
        if not allow_promotion:
            return isinstance(__obj, __type)
        try:
            __type(__obj)
            return True
        except:
            return False
        
    def add(self, data: Dict | Object, auto_commit: bool = True) -> Dict:
        """Adds data to the stack. New data is merged with existing data."""
        cols = [col for col in self.conn.model.model_schema if col.name not in self.data]
        
        self.data.clear()
        self._process_field_defaults(cols)
        for key, value in data.items():
            self.data[key] = value
        if auto_commit:
            return self.commit()
        return self.data
    
    def commit(self) -> ResultSet[Object]:
        """
        Validates and commits data according to schema definitions.
        
        This method performs several validation steps:
        1. Processes field defaults and dependencies
        2. Validates unique constraints
        3. Validates data types (if not ignored)
        4. Validates required fields
        5. Processes null fields
        6. Writes valid data to database
        
        Returns
        -------
        ResultSet[Object]
            ResultSet containing committed data if successful
            Empty ResultSet if validation fails
            
        Raises
        ------
        KeyAlreadyExists
            If unique constraint is violated during commit
        KeyNotFound  
            If required fields are missing
        TypeError
            If type validation fails and no_raise=False
        NotNullViolation
            If null value found in not-null field
        NotUniqueViolation
            If unique constraint violated by depends process
        """
        schema: List[Column] = self.conn.model.model_schema
        
        # Process default values and dependencies
        self._process_field_defaults(schema)
        
        
        
        for field in filter(lambda col: col.unique, schema):
            unique_check = self._validate_unique_constraints(field, no_raise=self.conn.no_raise, on_commit=True)
            if not unique_check.status:
                
                return ResultSet([])
            
        if not self.conn.model._ignore_types:
            self._validate_types(schema, no_raise=self.conn.no_raise)
         
        if not self._validate_required_fields(no_raise=self.conn.no_raise):
            
            return ResultSet([])

        self._process_null_fields(schema)
            
        self.conn.write(self.data)
        self.data.clear()
        
        return self.conn.fetch()
    
    def _process_null_fields(self, schema: List[Column], no_raise: bool = False):
        """Process null fields in the data to ensure required fields are not null"""
        for field in schema:
            field_value = self.data.get(field.name, Missing)
            if not field.not_null and field_value is None:
                self.data[field.name] = NULL
                
            if field.not_null and field_value is None:
                if no_raise:
                    yield {"id": self.data["_id"],"field": field.name, "value": field_value, "message": f"Field {field.name!r} can not be null."}
                else:
                    raise NotNullViolation(field.name)
        return True
    def _process_field_defaults(self, schema: List[Column]):
        """Process default values and dependencies for fields"""
        for field in schema:
            if field.increment:
                increment_by = getattr(self.conn.model, '_increment_by', 1)
                value = self.current_size() + increment_by
                self.data[field.name] = value
            if not self.data.get(field.name) and (field.default or field.depends):
                self._set_field_value(field)
    def _set_field_value(self, field: Column):
        """Set appropriate value for a field based on its configuration"""
        if field.depends in (Random.date, Date.now, Random.hash):
            
            value = field.depends(**field.kw) if field.depends != Random.hash else Random.hash()
            if field.unique:
                unique_check = self._validate_unique_constraints(field, value, no_raise=self.conn.no_raise)
                if unique_check.status:
                    self.data[field.name] = value
                else:
                    raise NotUniqueViolation(field.name, "Depends procces sent not-unique value.")
            else:
                self.data[field.name] = value
        elif isinstance(field.depends, partial) or callable(field.depends):
            value = field.depends(**field.kw)
            if field.unique:
                unique_check = self._validate_unique_constraints(field, value, no_raise=self.conn.no_raise)
                if unique_check.status:
                    self.data[field.name] = value
                else:
                    raise NotUniqueViolation(field.name, "Depends procces sent not-unique value.")
            else:
                self.data[field.name] = value
    def _validate_unique_constraints(self, field: Column, value: Any = ..., no_raise: bool = False, on_commit: bool = False, debug: bool = False):
        """Validate field is unique"""
        for frame in traceback.extract_stack()[:-1]:  # -1 to exclude current function
            if debug and  frame.name in ("_validate_unique_constraints", "_process_field_defaults", "_set_field_value", "_process_null_fields", "_validate_types", "_validate_required_fields", "commit", "review"):
                print(f"File: {frame.filename.split('/')[-1]}, Line: {frame.lineno}, Function: {frame.name}")
        if not field.unique:
            return Object({"status": True, "context": Object({"field": field.name, "value": value}), "message": f"Field {field.name!r} is not unique."})
        if not on_commit:
            fetched = self.conn.stack[lambda data: data[field.name] == value]
            status = len(fetched) > 1
            return Object({"status": not status, "context": Object({"field": field.name, "value": value}), "message": f"Field {field.name!r} is unique."})
        if on_commit:
            fetched = self.conn.fetch()
            for data in fetched:
                matches = self.conn.stack[lambda x: x[field.name] == data[field.name]]
                if condition:=len(matches) > 1 or (len(matches) == 1 and self.data[field.name] == data[field.name]):
                    message = f"Unique constraint for '{field.name}' already exists." \
                              if condition else \
                              f"Commit process stopt by unique constraint for '{field.name}' field." 
                    if no_raise:
                        
                        return Object({"status": len(matches) < 1, "context": Object({"id": data["_id"], "field": field.name, "value": data[field.name]}), "message": message})
                    raise KeyAlreadyExists(field.name)  
        
        return Object({"status": True, "context": Object({"field": field.name, "value": value}), "message": f"Field is unique but nothing found unsafe."})
    def _validate_types(self, schema: List[Column], no_raise: bool = False):
        """Validate type conversions for all fields"""
        for field in schema:
            value = self.data.get(field.name)
            if field.name not in self.data:
                if no_raise:
                    yield Object({"field": field.name, "expected": field.typer.__name__, "found": type(value).__name__, "message": f"Error: Field {field.name!r} is missing.", "missing": True})
            if self.is_convertible(value, field.typer, allow_promotion=self.conn.allow_promotion):
                if not isinstance(value, field.typer):
                    yield Object({"field": field.name, "expected": field.typer.__name__, "found": type(value).__name__, "message": f"Warning: Field {field.name!r} is convertible to type {field.typer.__name__!r} but not same type."})
            else:
                if no_raise:
                    yield Object({"field": field.name, "expected": field.typer.__name__, "found": type(value).__name__, "message": f"Error: Field {field.name!r} is not convertible to type {field.typer.__name__!r}"})  
                else:
                    raise TypeError(f"Field {field.name!r} can not be converted to type {field.typer.__name__!r}") 
        return True
    def _validate_required_fields(self, no_raise: bool = False) -> Union[bool, list[str]]:
        """Validate that all required model fields are present"""
        datafields = [attr for attr in dir(self.conn.model) 
                     if not attr.startswith('_') and attr not in self.__excluded_slots__]
        
        model_keys = sorted(datafields)
        data_keys = sorted(list(self.data.keys()))
        
        if len(self.data) != len(datafields) or model_keys != data_keys:
            model = self.conn.model
            missing_keys = [key for key in model_keys if key not in data_keys and not any(col.increment and col.name == key for col in model.model_schema)]
            if missing_keys:
                
                if no_raise:
                    return missing_keys
                for mkey in missing_keys:
                    col = next((col for col in model.model_schema if col.name == mkey), None)
                    if not col.not_null:
                        self.data[mkey] = NULL
                missing_keys = [col.name for col in model.model_schema if col.not_null and col.name in missing_keys]
                if missing_keys:
                    raise KeyNotFound(", ".join(missing_keys), "You have to fill all required fields.")
        return True

    def __setitem__(self, item: str | int, value: Any):
        """
        Sets the value for a given key in the data dictionary.

        ## Args:
            item (str | int): The key to set in the dictionary.
            value (Any): The value to associate with the key.

        ## Returns:
            Any: The value that was set for the given key.

        ## Example:
            >>> obj['key'] = 'value'  # Sets 'value' for key 'key' in data dictionary
        """
        self.data[item] = value
        return self.data[item]

    def __getitem__(self, item: int | str | FilterItem) -> ResultSet[Object]:
        """
        Retrieves filtered items from the data using various access methods.

        ## Args:
            item instance of:
            - int or str: Direct index access
            - FilterItem: Object containing filter function
            - Callable: Custom filter function 

        ## Returns:
            ResultSet[Object]: Matching dictionary items, empty list if no matches,
            empty ResultSet if invalid item type.

        ## Raises:
            KeyNotFound: If filter parameter names don't match column names

        ## Example:
            Direct access:
            >>> stack[0] # Get first item
            >>> stack["id"] # Get item by key
            
            Filter object:
            >>> stack[FilterItem(lambda x: x.age > 21)]
            
            Callable filter:
            >>> stack[lambda x: x.status == "active"]
            >>> stack[lambda age, gender: age > 21 and gender == "male"]

        ## Notes:
            Callable filters must either:
            - Use parameter names matching column names
            - Use single parameter for direct value filtering
        """
        if not isinstance(item, (FilterItem, Callable, int, str)):
            return ResultSet([])
            
        if isinstance(item, (int, str)):
            return self.data[item]

        if isinstance(item, FilterItem):
            predicate = item.filter
            params = item.filter.__code__.co_varnames
        else:
            predicate = item
            params = item.__code__.co_varnames

        datas = self.conn.fetch()
        if not datas:
            return ResultSet([])

        responses = []
        sample_data = datas[0]
        
        if len(params) == 1 and params[0] not in sample_data:
            responses = [data for data in datas if predicate(data)]
        else:
            invalid_params = [p for p in params if p not in sample_data]
            if invalid_params:
                raise KeyNotFound(", ".join(invalid_params), 
                "Parameter names must match column names when filtering by columns")
            
            for data in datas:
                kwargs = {k: data[k] for k in params if k in data}
                if predicate(**kwargs):
                    responses.append(data)
        return responses