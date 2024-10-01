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
            return Filter(
                must=left_val.must_fields + right_val.must_fields,
                any=left_val.any_fields + right_val.any_fields,
                must_not=left_val.must_not_fields + right_val.must_not_fields
            )
        elif self.op == 'or':
            return Filter(
                any=left_val.must_fields + right_val.must_fields + left_val.any_fields + right_val.any_fields
            )
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


class AnyNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

    def evaluate(self):
        """
        Evaluates the AnyNode.

        Returns:
            Filter: A filter with conditions added to the 'any_fields' list.
        """
        expr_filter = self.expr.evaluate()
        # Merge 'must_fields' and 'any_fields' from the expression into 'any_fields' of the current filter
        return Filter(any=expr_filter.must_fields + expr_filter.any_fields)


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
        self.validate_field_format(self.field)
        field = self.field

        if field == ':id:':
            value = [int(v.strip()) for v in self.value.strip('[]').split(',')]
            if self.op == 'not in':
                return Filter(must_not=[FieldCondition(field, MatchID(value))])
            elif self.op == 'in':
                return Filter(must=[FieldCondition(field, MatchID(value))])

        if self.op in ['in', 'not in']:
            value = [v.strip() for v in self.value.strip('[]').split(',')]
            value = [self.parse_value(v) for v in value]
            all_comparators = False  # For 'in' and 'not in' operators, all_comparators does not need to be set
        else:
            if self.value.startswith('[') and self.value.endswith(']'):
                # if the value is a list, set all_comparators=True
                value = [v.strip() for v in self.value.strip('[]').split(',')]
                value = [self.parse_value(v) for v in value]
                all_comparators = True
            else:
                value = self.parse_value(self.value)
                all_comparators = False

        comparator = operator_map[self.op]
        condition = FieldCondition(
            field,
            MatchField(value, comparator, all_comparators=all_comparators, not_in=self.op == 'not in')
        )
        if self.op.startswith('not') and self.op != 'not in':
            return Filter(must_not=[condition])
        else:
            return Filter(must=[condition])

    @staticmethod
    def validate_field_format(field_name):
        """
        Validates that the field name is surrounded by ':'.

        Parameters:
            field_name (str): The field name to validate.

        Raises:
            ValueError: If the field name does not start and end with ':'.
        """
        if not (field_name.startswith(':') and field_name.endswith(':')):
            raise ValueError(
                "Field name is not correctly formatted. The field name must start and end with ':', for example, ':age:'.")

    @staticmethod
    def parse_value(value):
        """
        Parses a string into a number (int or float), boolean, or string if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, bool, or str: The parsed value.
        """
        if value == 'True':
            return True
        elif value == 'False':
            return False
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("'\"")


class RangeNode(ASTNode):
    def __init__(self, field, left_value, right_value, left_op, right_op):
        self.field = field
        self.left_value = left_value
        self.right_value = right_value
        self.left_op = left_op
        self.right_op = right_op

    def evaluate(self):
        """
        Evaluates the range node.

        Returns:
            Filter: A filter based on the range condition.
        """
        self.validate_field_format(self.field)
        field = self.field

        # Parse values
        left_value = self.parse_value(self.left_value)
        right_value = self.parse_value(self.right_value)

        # Determine start and end values and their inclusiveness
        # Adjust operators to standard form
        ops = {'>': ('>', False), '>=': ('>=', True), '<': ('<', False), '<=': ('<=', True)}
        left_op_sym, left_inclusive = ops[self.left_op]
        right_op_sym, right_inclusive = ops[self.right_op]

        # Determine start, end, and inclusiveness based on operator direction
        if ':' in self.field:
            # Variable in the middle, e.g., "16 > :age: <= 18"
            if left_op_sym in ['<', '<=']:
                start = left_value
                start_inclusive = left_inclusive
                end = right_value
                end_inclusive = right_inclusive
            else:
                start = right_value
                start_inclusive = right_inclusive
                end = left_value
                end_inclusive = left_inclusive
        else:
            # Should not reach here in this context
            raise ValueError("Invalid range expression")

        # Determine inclusive parameter for MatchRange
        if start_inclusive and end_inclusive:
            inclusive = True
        elif start_inclusive:
            inclusive = 'left'
        elif end_inclusive:
            inclusive = 'right'
        else:
            inclusive = False

        condition = FieldCondition(field, MatchRange(start, end, inclusive=inclusive))
        return Filter(must=[condition])

    @staticmethod
    def validate_field_format(field_name):
        """
        Validates that the field name is surrounded by ':'.

        Parameters:
            field_name (str): The field name to validate.

        Raises:
            ValueError: If the field name does not start and end with ':'.
        """
        if not (field_name.startswith(':') and field_name.endswith(':')):
            raise ValueError(
                "Field name is not correctly formatted. The field name must start and end with ':', for example, ':age:'.")

    @staticmethod
    def parse_value(value):
        """
        Parses a string into a number (int or float), boolean, or string if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, bool, or str: The parsed value.
        """
        if value == 'True':
            return True
        elif value == 'False':
            return False
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.replace(':', '')


class ExpressionParser:
    """
    Parse the expression and generate the corresponding filter.
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
        # use lookahead to ensure '||(' has no spaces
        token_pattern = re.compile(
            r"(:\w+:|==|!=|>=|<=|>|<|in|not in|and|or|not|\|\|\((?=\S)|\|\||\(|\)|\[.*?\]|\d+\.\d+|\d+|'[^']*'|\"[^\"]*\"|\w+)"
        )
        tokens = token_pattern.findall(expression)

        # check illegal usage of '||' (followed by '(' with a space)
        for idx in range(len(tokens) - 1):
            if tokens[idx] == '||' and tokens[idx + 1] == '(':
                raise ValueError("When using '||', it must be immediately followed by '(', without any spaces.")

        # check illegal usage of '||' (not followed by '(')
        for idx in range(len(tokens)):
            if tokens[idx] == '||':
                raise ValueError("When using '||', it must be immediately followed by '(', without any spaces.")

        return tokens

    def parse(self):
        """
        Parse the expression.

        Returns:
            ASTNode: The root node of the abstract syntax tree.
        """
        self.current = 0
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
        Parse the basic expression (prefix operators, parentheses, and conditions).

        Returns:
            ASTNode: The parsed node.
        """
        if self.current < len(self.tokens) and self.tokens[self.current] == '||(':
            self.current += 1
            expr = self.parse_and_only()
            if self.current >= len(self.tokens) or self.tokens[self.current] != ')':
                raise ValueError("Missing right parenthesis")
            self.current += 1
            return AnyNode(expr)
        elif self.current < len(self.tokens) and self.tokens[self.current] == '(':
            self.current += 1
            node = self.parse_or()
            if self.current >= len(self.tokens) or self.tokens[self.current] != ')':
                raise ValueError("Missing right parenthesis")
            self.current += 1
            return node
        else:
            return self.parse_condition()

    def parse_and_only(self):
        """
        Parse expressions that only allow 'and' as a logical connector.

        Returns:
            ASTNode: The parsed node.
        """
        node = self.parse_not()
        while self.current < len(self.tokens):
            if self.tokens[self.current] == 'and':
                self.current += 1
                right = self.parse_not()
                node = BinaryOpNode(node, 'and', right)
            elif self.tokens[self.current] == 'or':
                raise ValueError("After '||(', only 'and' is allowed as a logical connector, 'or' is not allowed.")
            elif self.tokens[self.current] == ')':
                break
            else:
                break
        return node

    @staticmethod
    def parse_value(value):
        """
        Parses a string into a number (int or float), boolean, or string if possible.

        Parameters:
            value (str): The string to be parsed.

        Returns:
            int, float, bool, or str: The parsed value.
        """
        if value == 'True':
            return True
        elif value == 'False':
            return False
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("'\"")

    def compare_values(self, op1, val1, op2, val2):
        """
        Compare two values and return the stricter limit.

        Parameters:
            op1 (str): The operator of the first value.
            val1 (str): The first value.
            op2 (str): The operator of the second value.
            val2 (str): The second value.
        """
        num1 = self.parse_value(val1)
        num2 = self.parse_value(val2)

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
        if (self.current + 4) < len(self.tokens):
            token_sequence = self.tokens[self.current:self.current + 5]
            if (token_sequence[1] in ['>', '<', '>=', '<='] and
                    ':' in token_sequence[2] and
                    token_sequence[3] in ['>', '<', '>=', '<=']):
                left_value = token_sequence[0]
                left_op = token_sequence[1]
                field = token_sequence[2]
                right_op = token_sequence[3]
                right_value = token_sequence[4]
                self.current += 5
                return RangeNode(field, left_value, right_value, left_op, right_op)
        # Standard condition parsing
        left = self.tokens[self.current]
        self.current += 1
        if self.current >= len(self.tokens):
            raise ValueError("Expression is incomplete, lacks operator and value.")
        op = self.tokens[self.current]
        if op == 'not' and self.current + 1 < len(self.tokens):
            op = 'not ' + self.tokens[self.current + 1]
            self.current += 1
        self.current += 1
        if self.current >= len(self.tokens):
            raise ValueError("Expression is incomplete, lacks value.")
        right = self.tokens[self.current]
        self.current += 1

        # Check if the field is on the left side
        if ':' in left:
            self.validate_field_format(left)
            field = left
            return ConditionNode(field, op, right)
        else:
            # Field on the right side, reverse the operator
            op = self.reverse_op(op)
            self.validate_field_format(right)
            field = right
            return ConditionNode(field, op, left)

    @staticmethod
    def validate_field_format(field_name):
        """
        Validates that the field name is surrounded by ':'.

        Parameters:
            field_name (str): The field name to validate.

        Raises:
            ValueError: If the field name does not start and end with ':'.
        """
        if not (field_name.startswith(':') and field_name.endswith(':')):
            raise ValueError(
                "Field name is not correctly formatted. The field name must start and end with ':', for example, ':age:'.")

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

    def to_dict(self):
        """
        Convert the parsed expression into a dictionary.

        Returns:
            dict: The dictionary representation of the parsed expression.
        """
        return self.to_filter().to_dict()
