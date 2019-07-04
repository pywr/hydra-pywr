import marshmallow

# This is a parameter instance.
class DataFrameField(marshmallow.fields.Field):
    """ Marshmallow field representing a Parameter. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        return value.to_json()

    def _deserialize(self, value, attr, data, **kwargs):
        df = pandas.DataFrame.from_dict(value)
        # Row ordering is not preserved by from_dict.
        # We return the dataframe with the same ordering in the rows and columns as given in value.
        # TODO do something better than this.
        row_order = value[list(value.keys())[0]]
        df = df.loc[row_order, list(value.keys())]
        return df


from .hydropower import *
from .ukwrp import *