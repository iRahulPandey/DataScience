from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json


class InvalidInputError(Exception):
    """Invalid model input."""


SYNTAX_ERROR_FIELD_MAP = {"X1": "X1", "X2": "X2", "X3": "X3"}


class DataRequestSchema(Schema):
    X1 = fields.Float()
    X2 = fields.Float()
    X3 = fields.Float()


def _filter_error_rows(errors: dict, validated_input: t.List[dict]) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    #schema = DataRequestSchema()

    # convert syntax error field names (beginning with numbers)
    #for dict in input_data:
    #    for key, value in SYNTAX_ERROR_FIELD_MAP.items():
    #        dict[value] = dict[key]
    #        del dict[key]

    #errors = None
    #try:
    #    schema.load(input_data)
    #except ValidationError as exc:
    #    errors = exc.messages

    # convert syntax error field names back
    # this is a hack - never name your data
    # fields with numbers as the first letter.
    #for dict in input_data:
    #    for key, value in SYNTAX_ERROR_FIELD_MAP.items():
    #        dict[key] = dict[value]
    #        del dict[value]

    #if errors:
    #    validated_input = _filter_error_rows(errors=errors, validated_input=input_data)
    #else:
    #    validated_input = input_data

    #return validated_input, errors
