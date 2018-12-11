from flask_wtf import FlaskForm
from wtforms.fields import (StringField,BooleanField,IntegerField, DecimalField)
from wtforms.validators import  NumberRange
class TipForm(FlaskForm):
    num_of_clients = IntegerField('Number of clients', validators=[NumberRange(min=1, max=7)])
    distance =  DecimalField('Distance', places=2)
    tolls = DecimalField('Tolls', places=2)
    duration = DecimalField('Duration', places=2)
    is_weekend = BooleanField('Is weekend')
