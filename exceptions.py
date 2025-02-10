""" 
Exceptions for JSONIC tracebacks.
"""

import json

class SystemError(Exception):
    """
    Base class for library exceptions.
    """
    pass

class KeyAlreadyExists(SystemError):
    """
    Raised when a key already exists in the database.
    """
    def __init__(self, key: str) -> None:
        super().__init__(f"Key {key!r} already exists. Process stopt by PrimaryKey exception.")


class NotUniqueViolation(SystemError):
    """
    Raised when a unique-key already exists in the database.
    """
    def __init__(self, key: str, extra: str) -> None:
        super().__init__(f"Key {key!r} already exists. {extra.strip().strip('.')}. Process stopt by UniqueKey exception.")
        
class KeyNotFound(SystemError):
    """
    Raised when a required-key not found in the cursor.
    """
    def __init__(self, key: str, message: str) -> None:
        super().__init__(f"Key {key!r} not found. {message}")

class KeyNotExists(SystemError):
    """
    Raised when a key not exists in the schema.
    """
    def __init__(self, key: str | list[str]) -> None:
        keys = ", ".join(key) if isinstance(key, list) else key
        super().__init__(f"Key '{keys}' not exists in the schema. Please pass only schema keys.")

class NotNullViolation(SystemError):
    """
    Raised when a notnull-key is null.
    """
    def __init__(self, key: str) -> None:
        super().__init__(f"Key {key!r} cannot be null. Process stopt by NotNull exception.")

class CursorInvalid(SystemError):
    """
    Raised when cursor is empty.
    """
    def __init__(self) -> None:
        super().__init__(f"No data found in cursor.")

class DataCorrupted(SystemError):
    """
    Raised when data is corrupted.
    """
    def __init__(self, exc: json.decoder.JSONDecodeError) -> None:
        super().__init__(f"JSON data belongs line {exc.lineno!r} and column {exc.colno!r} is corrupted. Please fix your JSON data or set audit_fix=True in connection to reset JSON database.")