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
        Evaluate the AST node.
        
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
        Evaluate the binary operation node.
        
        Returns:
            Filter: The combined filter based on the operation.
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
        Evaluate the unary operation node.
        
        Returns:
            Filter: The negated filter for 'not' operation.
        """
        result = self.expr.evaluate()
        if self.op == 'not':
            return Filter(must_not=result.must_fields)
        else:
            raise ValueError(f"Unknown unary operator: {self.op}")


class ConditionNode(ASTNode):
    def __init__(self, field, op, value):
        self.field = field
        self.op = op
        self.value = value

    def evaluate(self):
        """
        Evaluate the condition node.
        
        Returns:
            Filter: The filter based on the condition.
        """
        field = self.field.replace(':', '')
        if field == 'id':
            value = [int(v.strip()) for v in self.value.strip('[]').split(',')]
            if self.op == 'not in':
                return Filter(must_not=[FieldCondition(':id:', MatchID(value))])
            else:
                return Filter(must=[FieldCondition(':id:', MatchID(value))])
        else:
            if self.op in ['in', 'not in']:
                value = [v.strip() for v in self.value.strip('[]').split(',')]
                value = [self.parse_number(v) for v in value]
            else:
                value = self.parse_number(self.value)
            comparator = operator_map[self.op]
            condition = FieldCondition(field,
                                       MatchField(value, comparator, all_comparators=False, not_in=self.op == 'not in'))
            if self.op.startswith('not'):
                return Filter(must_not=[condition])
            else:
                return Filter(must=[condition])

    @staticmethod
    def parse_number(value):
        """
        Parse a string into a number (int or float) if possible.
        
        Parameters:
            value (str): The string to parse.
        
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
        Evaluate the range node.

        Returns:
            Filter: The filter based on the range condition.
        """
        field = self.field.replace(':', '')
        start = self.parse_number(self.start)
        end = self.parse_number(self.end)

        # determine the actual start and end values
        if self.start_op in ['>', '>='] and self.end_op in ['>', '>=']:
            start, end = end, start
            
        if start > end:
            raise ValueError(f"Invalid range expression: start value {start} is greater than end value {end}.")
        
        start_inclusive = self.end_op in ['>=', '<=']
        end_inclusive = self.start_op in ['<=', '>=']
            
        if start == end:
            if not (start_inclusive and end_inclusive):
                raise ValueError(f"Invalid range expression: when start value equals end value, "
                                 f"both sides must use one of ['==', '!=', '>=', '<='].")
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
        Parse a string into a number (int or float) if possible.
        
        Parameters:
            value (str): The string to parse.
        
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
    """
    Utilizing Abstract Syntax Trees (ASTs) and the Visitor pattern,
        "FieldExpression" is employed to parse and evaluate string expressions into Filter objects.

    Supported syntax:
        :field: represents a field, e.g., :order:, :name:, etc.
        :id: special reserved word for matching IDs

    Supported comparison operators:
        ==, !=, >=, <=, >, <

    Supported logical operators:
        and, or, not, in

    Supported parentheses:
        ()
        
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
          Specially when using 'not' operator.
        - Use 'not in' instead of 'not :field: in [value]' when possible.
        - Use 'not (:field: in [value])' or 'not (:field: not in [value])'.

    """

    def __init__(self, expression):
        """
        Initialize the ExpressionParser.

        Parameters:
            expression (str): The expression to parse.
        """
        if not expression or expression.strip() == '':
            raise ValueError("Expression cannot be empty.")
        self.expression = expression
        self.tokens = self.tokenize(expression)
        self.current = 0

        if not self.tokens:
            raise ValueError("Unable to parse the expression, please check the syntax.")

    @staticmethod
    def tokenize(expression):
        """
        Tokenize the input expression.

        Parameters:
            expression (str): The expression to tokenize.

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
            ASTNode: The root node of the Abstract Syntax Tree.
        """
        return self.parse_or()

    def parse_or(self):
        """
        Parse 'or' expressions.

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
        Parse 'and' expressions.

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
        Parse 'not' expressions.

        Returns:
            ASTNode: The parsed node.
        """
        if self.current < len(self.tokens) and self.tokens[self.current] == 'not':
            self.current += 1
            expr = self.parse_condition()
            return UnaryOpNode('not', expr)
        return self.parse_condition()

    @staticmethod
    def parse_number(value):
        """
        Parse a string into a number (int or float) if possible.
        
        Parameters:
            value (str): The string to parse.
        
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
            op1 (str): The operator for the first value.
            val1 (str): The first value.
            op2 (str): The operator for the second value.
            val2 (str): The second value.
        """
        num1 = self.parse_number(val1)
        num2 = self.parse_number(val2)
        
        if op1 in ['>', '>='] and op2 in ['<', '<=']:
            if num1 > num2:
                return op2, val2  # more strict upper limit
            else:
                return op1, val1  # more strict lower limit
        elif op1 in ['<', '<='] and op2 in ['>', '>=']:
            if num1 < num2:
                return op2, val2  # more strict lower limit
            else:
                return op1, val1  # more strict upper limit
        else:
            # return the stricter limit for the same direction inequality
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
        if self.tokens[self.current] == '(':
            self.current += 1
            node = self.parse_or()
            self.current += 1  # skip the right parenthesis
            return node

        left = self.tokens[self.current]
        self.current += 1
        op = self.tokens[self.current]
        self.current += 1
        middle = self.tokens[self.current]
        self.current += 1

        if (op in ['>', '<', '>=', '<='] and self.current < len(self.tokens) and
                self.tokens[self.current] in ['>', '<', '>=', '<=']):
            # detect double-sided range expression
            second_op = self.tokens[self.current]
            self.current += 1
            right = self.tokens[self.current]
            self.current += 1

            if ':' in middle:
                # if the field is in the middle
                if (op in ['<', '<='] and second_op in ['<', '<=']) or (op in ['>', '>='] and second_op in ['>', '>=']):
                    # same direction inequality, create RangeNode
                    return RangeNode(middle, left, right, op, second_op)
                else:
                    # reverse inequality, choose the stricter limit
                    strict_op, strict_val = self.compare_values(op, left, second_op, right)
                    # check if the field is in the middle
                    if ':' in middle:
                        # if the strict value is on the left side of the field, reverse the operator
                        if strict_op == op:
                            strict_op = self.reverse_op(strict_op)

                        return ConditionNode(middle, strict_op, strict_val)
                    else:
                        return ConditionNode(left, strict_op, strict_val)
            else:
                # if the field is not in the middle, keep the original logic
                return ConditionNode(middle, op, left)

        if ':' in left:
            return ConditionNode(left, op, middle)
        else:
            return ConditionNode(middle, op, left)

    @staticmethod
    def reverse_op(op):
        """
        Reverse the comparison operator.

        Parameters:
            op (str): The operator to reverse.

        Returns:
            str: The reversed operator.
        """
        reverse_map = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
        return reverse_map.get(op, op)

    def to_filter(self):
        """
        Get the final filter by parsing and evaluating the expression.

        Returns:
            Filter: The resulting filter.
        """
        ast = self.parse()
        return ast.evaluate()
