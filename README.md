# JSONIC Database Library

JSONIC is a flexible database abstraction library that provides a simple interface for working with JSON-like data structures and MongoDB. It offers schema validation, type checking, and automatic data management features.

## Features

- Schema-based data validation
- Support for MongoDB and JSON file storage
- Type checking and data validation
- Auto-incrementing fields
- Unique constraints
- Primary key management
- Default value handlers
- Custom field dependencies
- Data review and auditing

## Installation

```bash
pip install <libname>  # Replace with actual package name
```

## Quick Start

Here's a basic example of how to use JSONIC:

```python
from jsonic import JSONField, JSONIC, Column
from jsonic.utils import Random

@JSONField(increment_by=1)
class UserModel:
    id: str = Column(depends=Random.hash, primary_key=True)
    name: str = Column(not_null=True)
    age: int = Column(increment=True)
    email: str = Column(unique=True)

# Create database connection
with JSONIC(engine="json://database", model=UserModel) as db:
    cursor = db.cursor()

    # Add new record
    cursor.add({
        "name": "John Doe",
        "email": "john@example.com"
    })

    # Commit changes
    cursor.commit()
```

## Schema Definition

### Using the JSONField Decorator

The `@JSONField` decorator configures model-level settings:

```python
@JSONField(
    increment_by=1,        # Increment step for auto-increment fields
    ignore_types=False,    # Enable/disable type checking
    auto_fit_columns=True  # Automatically adjust columns
)
class MyModel:
    # Model fields here
```

### Column Types

Columns can be defined with various attributes:

```python
class UserModel:
    id: str = Column(primary_key=True)                    # Primary key field
    name: str = Column(not_null=True)                     # Required field
    age: int = Column(increment=True)                     # Auto-incrementing
    email: str = Column(unique=True)                      # Unique field
    created: str = Column(depends=Random.date)            # Generated field
    hash: str = Column(depends=Random.hash, unique=True)  # Unique hash
```

### Column Properties

- `primary_key`: Makes field unique and not null
- `not_null`: Field cannot be null
- `unique`: Field must have unique values
- `increment`: Auto-incrementing field
- `depends`: Function for generating field values
- `default`: Default value for field

## Database Operations

### Connecting to Database

```python
# JSON File Storage
with JSONIC(engine="json://database", model=MyModel) as db:
    # Operations here

# MongoDB Connection
mongo_uri = "mongodb+srv://user:pass@host"
with JSONIC(
    engine="mongo://dbname@collection",
    model=MyModel,
    database_config=dict(mongo_uri=mongo_uri)
) as db:
    # Operations here
```

### Basic Operations

```python
# Get cursor
cursor = db.cursor()

# Add record
cursor.add({
    "name": "John",
    "email": "john@example.com"
})

# Commit changes
cursor.commit()

# Fetch all records
all_records = db.fetch()

# Query records
results = db.query({"name": "John"})
```

### Data Filtering

```python
# Filter by lambda
adults = db.stack[lambda x: x["age"] >= 18]
adults = db.stack[lambda x: x.age >= 18]
adults = db.stack[lambda age, year: age > 18 and year < 2006]

# Filter by regular expression
name_filter = RegExp(r"^John", "name")
johns = db.stack[name_filter]
```

## Data Validation

### Review Data

```python
# Generate validation report
report = db.review(save_report=False)

# Access issues
print(report.type_issues)
print(report.unique_constraint_issues)
print(report.required_field_issues)
```

## Error Handling

Common exceptions:

- `KeyAlreadyExists`: Duplicate primary key
- `KeyNotFound`: Missing required fields
- `NotNullViolation`: Null value in not-null field
- `NotUniqueViolation`: Duplicate value in unique field
- `TypeError`: Invalid data type
- `DataCorrupted`: Database file corruption

## Best Practices

1. Always use type hints in model definitions
2. Use context managers (`with` statement) for database operations
3. Commit changes explicitly when auto_commit is disabled
4. Regular data validation using review()
5. Handle exceptions appropriately

## Configuration Options

```python
JSONIC(
    engine="json://database",        # Storage engine
    model=MyModel,                   # Data model
    audit_fix=True,                  # Auto-fix corrupted data
    commit_on_exit=False,            # Auto-commit on context exit
    raise_on_validation=True,        # Raise exceptions on validation
    allow_promotion=False            # Allow type promotion
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
