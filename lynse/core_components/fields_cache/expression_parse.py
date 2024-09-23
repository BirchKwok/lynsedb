import re
import operator
from abc import ABC, abstractmethod

from .filter import MatchField, MatchID, FieldCondition, Filter, MatchRange

operator_map = {
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    'in': operator.eq,
    'not in': operator.eq
}


class ASTNode(ABC):
    @abstractmethod
    def evaluate(self):
        """
        Evaluates the AST node.

        Returns:
            Filter: The evaluated filter.
        """
        pass


class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self):
        """
        Evaluates the binary operation node.

        Returns:
            Filter: A filter based on the operator combination.
        """
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        if self.op == 'and':
            return Filter(must=left_val.must_fields + right_val.must_fields,
                          any=left_val.any_fields + right_val.any_fields,
                          must_not=left_val.must_not_fields + right_val.must_not_fields)
        elif self.op == 'or':
            return Filter(any=left_val.must_fields + right_val.must_fields + left_val.any_fields + right_val.any_fields)
        else:
            raise ValueError(f"Unknown operator: {self.op}")


class UnaryOpNode(ASTNode):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def evaluate(self):
        """
        Evaluates the unary operation node.

        Returns:
            Filter: For the 'not' operation, returns the inverted filter.
        """
        result = self.expr.evaluate()
        if self.op == 'not':
            return Filter(
                must=result.must_not_fields,
                must_not=result.must_fields,
                any=result.any_fields
            )
        else:
            raise ValueError(f"Unknown unary operator: {self.op}")


class ConditionNode(ASTNode):
    def __init__(self, field, op, value):
        self.field = field
        self.op = op
        self.value = value

    def evaluate(self):
        """
        Evaluates the condition node.

        Returns:
            Filter: A filter based on the condition.
        """
        field = self.field.replace(':', '')
        field = ':id:' if field == 'id' else field

        if field == ':id:':
            value = [int(v.strip()) for v in self.value.strip('[]').split(',')]
            if self.op == 'not in':
                return Filter(must_not=[FieldCondition(field, MatchID(value))])
            elif self.op == 'in':
                return Filter(must=[FieldCondition(field, MatchID(value))])

        if self.op in ['in', 'not in']:
            value = [v.strip() for v in self.value.strip('[]').split(',')]
            value = [self.parse_number(v) for v in value]
        else:
            value = self.parse_number(self.value)
        comparator = operator_map[self.op]
        condition = FieldCondition(field,
                                   MatchField(value, comparator, all_comparators=False, not_in=self.op == 'not in'))
        if self.op.startswith('not') and self.op != 'not in':
            return Filter(must_not=[condition])
        else:
            return Filter(must=[condition])

    @staticmethod
    def parse_number(value):
        """
        Parses a string into a number (int or float) if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, or str: The parsed value.
        """
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("'\"")


class RangeNode(ASTNode):
    def __init__(self, field, start, end, start_op, end_op):
        self.field = field
        self.start = start
        self.end = end
        self.start_op = start_op
        self.end_op = end_op

    def evaluate(self):
        """
        Evaluates the range node.

        Returns:
            Filter: A filter based on the range condition.
        """
        field = self.field.replace(':', '')
        field = ':id:' if field == 'id' else field

        start = self.parse_number(self.start)
        end = self.parse_number(self.end)

        # Determine actual start and end values
        if self.start_op in ['>', '>='] and self.end_op in ['>', '>=']:
            start, end = end, start

        if start > end:
            raise ValueError(f"Invalid range expression: Start value {start} is greater than end value {end}.")

        start_inclusive = self.end_op in ['>=', '<=']
        end_inclusive = self.start_op in ['<=', '>=']

        if start == end:
            if not (start_inclusive and end_inclusive):
                raise ValueError(f"Invalid range expression: When the start value equals the end value, "
                                 f"both must use one of ['==', '!=', '>=', '<='].")
            inclusive = True
        else:
            if start_inclusive and end_inclusive:
                inclusive = True
            elif not start_inclusive and not end_inclusive:
                inclusive = False
            elif start_inclusive:
                inclusive = 'left'
            else:
                inclusive = 'right'

        condition = FieldCondition(field, MatchRange(start, end, inclusive=inclusive))
        return Filter(must=[condition])

    @staticmethod
    def parse_number(value):
        """
        Parses a string into a number (int or float) if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, or str: The parsed value.
        """
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.replace(':', '')


class ExpressionParser:
    r"""
    FieldExpression is a string parser used to parse field expressions.

    It can reassemble the parsing result into a Filter object of LynseDB, which can filter the query result.
    Its main implementation idea is to efficiently create Filters
        without requiring users to learn the use of complex filtering components such as MatchRange and FieldCondition.

    It is an important component of LynseDB and can effectively improve the user-friendliness of LynseDB.

    Supported syntax:
        - :field: represents a field, e.g., :order:, :name:, etc.
        - :id: special reserved word for matching IDs

    Supported comparison operators:
        - ==
        - !=
        - \>=
        - <=
        - \>
        - <

    Supported logical operators:
        - and
        - or
        - not
        - in

    Supported parentheses:
        - ()

    Examples:
        - :order: > 1
        - :order: >= 1
        - :order: >= 1 and :name: >= 1.3
        - not (:order: == 2)
        - :order: >= 1 and :name: != 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)
        - :id: in [1, 2]
        - :id: not in [1, 2]
        - :id: in [1, 2] and :order: >= 1 and :name: >= 1.3 and (:order: == 2 or :name: == 3) and not (:tt: == 2)

    Usage:
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

    Efficient syntax:
        - Use () to group expressions for better readability and maintainability.
        Specially when using `not` operator.
        - Use `not in` instead of `not :field: in [value]` when possible.
        - Use `not (:field: in [value])` or `not (:field: in [value])`, it can be more efficient.


    """

    def __init__(self, expression):
        """
        Initialize ExpressionParser.

        Parameters:
            expression (str): The expression to be parsed.
        """
        if not expression or expression.strip() == '':
            raise ValueError("Expression cannot be empty.")
        self.expression = expression
        self.tokens = self.tokenize(expression)
        self.current = 0

        if not self.tokens:
            raise ValueError("Unable to parse expression, please check the syntax.")

    @staticmethod
    def tokenize(expression):
        """
        Tokenize the input expression.

        Parameters:
            expression (str): The expression to be tokenized.

        Returns:
            list: A list of tokens.
        """
        token_pattern = re.compile(
            r"(:\w+:|==|!=|>=|<=|>|<|in|not in|and|or|not|\(|\)|\[.*?\]|\d+\.\d+|\d+|'[^']*'|\"[^\"]*\")")
        return token_pattern.findall(expression)

    def parse(self):
        """
        Parse the expression.

        Returns:
            ASTNode: The root node of the abstract syntax tree.
        """
        return self.parse_or()

    def parse_or(self):
        """
        Parse 'or' expression.

        Returns:
            ASTNode: The parsed node.
        """
        node = self.parse_and()
        while self.current < len(self.tokens) and self.tokens[self.current] == 'or':
            self.current += 1
            right = self.parse_and()
            node = BinaryOpNode(node, 'or', right)
        return node

    def parse_and(self):
        """
        Parse 'and' expression.

        Returns:
            ASTNode: The parsed node.
        """
        node = self.parse_not()
        while self.current < len(self.tokens) and self.tokens[self.current] == 'and':
            self.current += 1
            right = self.parse_not()
            node = BinaryOpNode(node, 'and', right)
        return node

    def parse_not(self):
        """
        Parse 'not' expression.

        Returns:
            ASTNode: The parsed node.
        """
        if self.current < len(self.tokens) and self.tokens[self.current] == 'not':
            self.current += 1
            expr = self.parse_not()
            return UnaryOpNode('not', expr)
        else:
            return self.parse_primary()

    def parse_primary(self):
        """
        Parse the basic expression (parentheses and conditions).

        Returns:
            ASTNode: The parsed node.
        """
        if self.tokens[self.current] == '(':
            self.current += 1
            node = self.parse_or()
            if self.current >= len(self.tokens) or self.tokens[self.current] != ')':
                raise ValueError("Missing right parenthesis")
            self.current += 1
            return node
        else:
            return self.parse_condition()

    @staticmethod
    def parse_number(value):
        """
        Parse a string into a number (int or float) if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, or str: The parsed value.
        """
        try:
            return int(value) if value.isdigit() else float(value)
        except ValueError:
            return value

    def compare_values(self, op1, val1, op2, val2):
        """
        Compare two values and return the stricter limit.

        Parameters:
            op1 (str): The operator of the first value.
            val1 (str): The first value.
            op2 (str): The operator of the second value.
            val2 (str): The second value.
        """
        num1 = self.parse_number(val1)
        num2 = self.parse_number(val2)

        if op1 in ['>', '>='] and op2 in ['<', '<=']:
            if num1 > num2:
                return op2, val2  # The stricter upper limit
            else:
                return op1, val1  # The stricter lower limit
        elif op1 in ['<', '<='] and op2 in ['>', '>=']:
            if num1 < num2:
                return op2, val2  # The stricter lower limit
            else:
                return op1, val1  # The stricter upper limit
        else:
            # For the same direction inequality, return the stricter limit
            if op1 in ['>', '>=']:
                return (op1, val1) if num1 > num2 else (op2, val2)
            else:
                return (op1, val1) if num1 < num2 else (op2, val2)

    def parse_condition(self):
        """
        Parse conditions and ranges.

        Returns:
            ASTNode: The parsed node (ConditionNode or RangeNode).
        """
        if self.current < len(self.tokens) - 2 and self.tokens[self.current + 1] in ['>', '<', '>=', '<=']:
            # Detect range expressions, e.g., "1 < :field: < 10"
            left = self.tokens[self.current]
            op1 = self.tokens[self.current + 1]
            middle = self.tokens[self.current + 2]
            if self.current + 3 < len(self.tokens) and self.tokens[self.current + 3] in ['>', '<', '>=', '<=']:
                op2 = self.tokens[self.current + 3]
                right = self.tokens[self.current + 4]
                self.current += 5
                return RangeNode(middle, left, right, op1, op2)
        # Standard condition parsing
        left = self.tokens[self.current]
        self.current += 1
        if self.current >= len(self.tokens):
            raise ValueError("Expression is incomplete, missing operator and value.")
        op = self.tokens[self.current]
        if op == 'not' and self.current + 1 < len(self.tokens):
            op = 'not ' + self.tokens[self.current + 1]
            self.current += 1
        self.current += 1
        if self.current >= len(self.tokens):
            raise ValueError("Expression is incomplete, missing value.")
        right = self.tokens[self.current]
        self.current += 1

        # Check if the field is on the left side
        if ':' in left:
            return ConditionNode(left, op, right)
        else:
            # Field on the right side, reverse the operator
            op = self.reverse_op(op)
            return ConditionNode(right, op, left)

    @staticmethod
    def reverse_op(op):
        """
        Reverse the comparison operator.

        Parameters:
            op (str): The operator to be reversed.

        Returns:
            str: The reversed operator.
        """
        reverse_map = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
        return reverse_map.get(op, op)

    def to_filter(self):
        """
        Get the final filter by parsing and evaluating the expression.

        Returns:
            Filter: The generated filter.
        """
        ast = self.parse()
        return ast.evaluate()
