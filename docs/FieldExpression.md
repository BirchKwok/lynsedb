# FieldExpression

**FieldExpression** is a string parser used to parse field expressions.

It can reassemble the parsing result into a Filter object of LynseDB, which can filter the query result. Its main implementation idea is to efficiently create Filters without requiring users to learn the use of complex filtering components such as MatchRange and FieldCondition.

It is an important component of LynseDB and can effectively improve the user-friendliness of LynseDB.

- **Supported syntax**:
    - :field: represents a field, e.g., :order:, :name:, etc.
    - :id: special reserved word for matching IDs

- **Supported comparison operators**:
    - ==
    - !=
    - \>=
    - <=
    - \>
    - <

- **Supported logical operators**:
    - and
    - or
    - not
    - in

- **Supported parentheses**:
    - ()

- **Examples**:
    - :order: > 1
    - :order: >= 1
    - :order: >= 1 and :name: >= 1.3
    - not (:order: == 2)
    - :order: >= 1 and :name: != 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)
    - :id: in [1, 2]
    - :id: not in [1, 2]
    - :id: in [1, 2] and :order: >= 1 and :name: >= 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)

- **Usage**:
```python linenums="1"
    >>> expression = ":order: > 1 and :name: >= 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)"
    >>> parser = ExpressionParser(expression)
    >>> pfilter = parser.to_filter()
    >>> print(pfilter.to_dict())
    {'must_fields': [
            {'key': 'order', 'matcher': {'value': 1, 'comparator': 'gt'}},
            {'key': 'name', 'matcher': {'value': 1.3, 'comparator': 'ge'}}
        ],
    'any_fields': [
            {'key': 'order', 'matcher': {'value': 2, 'comparator': 'eq'}},
            {'key': 'name', 'matcher': {'value': 3, 'comparator': 'eq'}}
        ],
    'must_not_fields': [
            {'key': 'tt', 'matcher': {'value': 2, 'comparator': 'eq'}}
        ]
    }
```

- **Efficient syntax**:
    - Use () to group expressions for better readability and maintainability.
      Specially when using `not` operator.
    - Use `not in` instead of `not :field: in [value]` when possible.
    - Use `not (:field: in [value])` or `not (:field: in [value])`, it can be more efficient.
