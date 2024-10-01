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
    - not in
    - ||()

- **Supported parentheses**:
    - ()

- **Syntax Detail**
    The FieldExpression syntax supports the following:

    1. Field names: Must be enclosed in colons, e.g., :order:, :name:
    2. Comparison operators: ==, !=, >, >=, <, <=
    3. Logical operators: and, or, not
    4. Set operators: in, not in
    5. Parentheses: () for grouping expressions
    6. Any condition operator: ||() to add any conditions
    7. Values: Numbers, strings (with or without quotes), booleans (True/False), and lists

    Expressions can be combined using logical operators. The 'not' operator can be applied to individual conditions or grouped expressions.

    Range queries are supported using the format: value1 op1 :field: op2 value2
    For example: 1 < :age: <= 18

    The ||() operator allows for efficient "any" condition grouping. It must be used as follows:
    - The '||' symbol must be immediately followed by '(' without any spaces.
    - Inside the parentheses, only 'and' can be used as a logical connector.
    - The 'or' operator is not allowed inside ||().
    - Example of correct usage: ||(:order: == 2 and :name: == 3)
    - Example of incorrect usage: ||(:order: == 2 or :name: == 3)

    This operator is used to create a group of conditions where any of them can be true, but within the group, all conditions must be connected by 'and'.

    The ||() operator is equivalent to using 'or' statements between each condition, but it provides a clearer way to express the logical intent. For example:

    ||(:order: == 2 and :name: == 3 and :age: > 18)

    is equivalent to:

    (:order: == 2 or :name: == 3 or :age: > 18)

    Using ||() makes it clear that these conditions are part of a single logical group where any condition can be true, which can be particularly useful in more complex queries. It also allows for potential query optimization in certain database systems.

- **Examples**:
    - :order: > 1
    - :order: >= 1 and :name: >= 1.3
    - not (:order: == 2)
    - :order: >= 1 and :name: != 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)
    - :order: >= 1 and :name: != 1.3 and ||(:order: == 2 and :name: == 3) and not (:tt: == 2)
    - :id: in [1, 2]
    - :id: not in [1, 2]
    - :id: in [1, 2] and :order: >= 1 and :name: >= 1.3 and ||(:order: == 2 and :name: == 3) and not (:tt: == 2)

- **Usage**:
```python linenums="1"
    >>> from lynse.field_models import ExpressionParser
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
    - Use `||()` to add any conditions, which can improve query efficiency.
            It can be more efficient than using `or` in some cases.
