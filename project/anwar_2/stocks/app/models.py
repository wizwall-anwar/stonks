from django.db import models
from . import module


class stock(models.Model):
    name_of_stock = models.CharField(max_length=20)
    image = models.ImageField(upload_to="images")
    from_date = models.DateTimeField()
    to_date = models.DateTimeField()

    def save(self):
        self.image = module.fig()
        self.save()

    def __str__(self):
        return self.Name_of_stock
        # Create your models here.
