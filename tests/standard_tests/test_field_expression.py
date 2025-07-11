import pytest
from lynse.core_components.fields_cache.expression_parse import ExpressionParser


def test_simple_expression():
    expression = ":id: in [1, 2, 3] and :test: == 'test_1'"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {'ids': [1, 2, 3]},
            {
                'key': 'test',
                'matcher': {
                    'value': 'test_1',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            }
        ],
        'any_fields': [],
        'must_not_fields': []}


def test_expression_with_any():
    expression = ":id: in [1, 2, 3] and ||(:test: == 'test_0')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]}],
        'any_fields': [{'key': 'test',
        'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}],
        'must_not_fields': []
    }


def test_expression_with_multiple_any():
    expression = ":id: in [1, 2, 3] and ||(:test: == 'test_0') and ||(:status: == 'active')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {'ids': [1, 2, 3]}
        ],
        'any_fields': [
            {
                'key': 'test',
                'matcher': {
                    'value': 'test_0',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            },
            {
                'key': 'status',
                'matcher': {
                    'value': 'active',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            }
        ],
        'must_not_fields': []
    }


def test_expression_with_all_comparators():
    expression = ":id: in [1, 2, 3] and ||(:test: > ['test_0', 'test_1'])"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {'ids': [1, 2, 3]}
        ],
        'any_fields': [
            {
                'key': 'test',
                'matcher': {
                    'value': ['test_0', 'test_1'],
                    'comparator': 'gt',
                    'all_comparators': True,
                    'not_in': False
                }
            }
        ],
        'must_not_fields': []
    }


def test_expression_with_not_in():
    expression = ":id: not in [4, 5, 6] and ||(:test: == 'test_2')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [],
        'any_fields': [
            {
                'key': 'test',
                'matcher': {
                    'value': 'test_2',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            }
        ],
        'must_not_fields': [{'ids': [4, 5, 6]}]
    }


def test_invalid_expression_missing_parenthesis():
    expression = ":id: in [1, 2, 3] and ||:test: == 'test_0')"
    with pytest.raises(ValueError):
        parser = ExpressionParser(expression)
        parser.to_dict()


def test_invalid_expression_unknown_operator():
    expression = ":id: >>> [1, 2, 3] and ||(:test: == 'test_0')"
    with pytest.raises(ValueError):
        parser = ExpressionParser(expression)
        parser.to_dict()


def test_empty_expression():
    expression = ""
    with pytest.raises(ValueError, match="Expression cannot be empty."):
        parser = ExpressionParser(expression)
        parser.to_dict()


def test_expression_with_not_operator():
    expression = ":id: in [1, 2, 3] and not (:test: == 'test_0')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]}],
        'any_fields': [],
        'must_not_fields': [
            {
                "key": "test",
                'matcher': {
                    'value': 'test_0',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            }
        ]
    }


def test_logical_connector():
    expression = ":id: in [1, 2, 3] and ||(:test: == 'test_0' or :status: == 'active') and not (:type: == 'premium')"
    with pytest.raises(ValueError):
        parser = ExpressionParser(expression)
        parser.to_dict()


def test_complex_expression():
    expression = ":id: in [1, 2, 3] and ||(:test: == 'test_0' and :status: == 'active') and not (:type: == 'premium')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]}],
        'any_fields': [{'key': 'test',
        'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}},
        {'key': 'status',
        'matcher': {'value': 'active',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}],
        'must_not_fields': [{'key': 'type',
        'matcher': {'value': 'premium',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}]
    }


def test_expression_with_multiple_and_or():
    expression = ":id: in [1, 2, 3] and :age: > 25 and ||(:test: == 'test_0')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]},
        {'key': 'age',
        'matcher': {'value': 25,
            'comparator': 'gt',
            'all_comparators': False,
            'not_in': False}}],
        'any_fields': [{'key': 'test',
        'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}],
        'must_not_fields': []
    }


def test_expression_with_nested_any():
    expression = ":id: in [1, 2, 3] and ||(:test: == 'test_0' and ||(:status: == 'active'))"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]}],
        'any_fields': [
            {
                'key': 'test',
                'matcher': {
                    'value': 'test_0',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            },
            {
                'key': 'status',
                'matcher': {
                    'value': 'active',
                    'comparator': 'eq',
                    'all_comparators': False,
                    'not_in': False
                }
            }
        ],
        'must_not_fields': []
    }


def test_expression_with_range():
    expression = "30 > :age: >= 18"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {
                'key': 'age',
                'matcher': {'start': 18, 'end': 30, 'inclusive': 'left'}}],
        'any_fields': [],
        'must_not_fields': []
    }


def test_expression_with_multiple_conditions():
    expression = ":id: in [1, 2, 3] and :age: > 25 and :status: == 'active' and ||(:test: == 'test_0')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {'ids': [1, 2, 3]},
            {'key': 'age',
                'matcher': {'value': 25,
            'comparator': 'gt',
            'all_comparators': False,
            'not_in': False}},
            {'key': 'status',
            'matcher': {'value': 'active',
                'comparator': 'eq',
                'all_comparators': False,
                'not_in': False}}
        ],
        'any_fields': [
            {'key': 'test',
            'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}
        ],
        'must_not_fields': []}


def test_expression_with_not_and_any():
    expression = ":id: in [1, 2, 3] and not (:test: == 'test_0') and ||(:status: == 'active')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [{'ids': [1, 2, 3]}],
        'any_fields': [{'key': 'status',
        'matcher': {'value': 'active',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}],
        'must_not_fields': [{'key': 'test',
        'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}]
    }


def test_expression_with_complex_conditions():
    expression = ":id: in [1, 2, 3] and :age: > 25 and not (:status: == 'inactive') and ||(:test: == 'test_0' and :type: == 'premium')"
    parser = ExpressionParser(expression)
    filter_obj = parser.to_dict()

    assert filter_obj == {
        'must_fields': [
            {'ids': [1, 2, 3]},
            {'key': 'age',
            'matcher': {'value': 25,
                'comparator': 'gt',
                'all_comparators': False,
                'not_in': False}}
        ],
        'any_fields': [
            {'key': 'test',
            'matcher': {'value': 'test_0',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}},
            {'key': 'type',
            'matcher': {'value': 'premium',
                'comparator': 'eq',
                'all_comparators': False,
                'not_in': False}}
        ],
        'must_not_fields': [
            {'key': 'status',
            'matcher': {'value': 'inactive',
            'comparator': 'eq',
            'all_comparators': False,
            'not_in': False}}
        ]
    }
