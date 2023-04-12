from django.db import models
from django.forms import fields
from .models import stock
from django import forms


class stock_form(forms.ModelForm):

    class Meta:
        model = stock
        fields = ["name_of_stock", "to_date", "from_date", ]
