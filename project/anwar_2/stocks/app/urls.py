from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_naame = 'app'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.stock_view, name='stock'),
]
