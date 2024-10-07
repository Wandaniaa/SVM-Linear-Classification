
from django.urls import path
from django.conf import settings
from . import views
from .views import index2_view
from .views import base_view
from .views import general_view
from .views import advanced_view
from .views import add_person
from .views import edit_person
from .views import delete_person
from .views import klasifikasi_view
from .views import evaluasi_view
from .views import normalisasi_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.BASE, name='BASE'),
     path('index2.html', index2_view, name='index2'),
     path('base.html', base_view, name='base'),
     path('general.html', general_view, name='general'),
     path('advanced.html', advanced_view, name='advanced'),
     path('add_person/', views.add_person, name='add_person'),
     path('edit_person/<int:pk>/', views.edit_person, name='edit_person'),
     path('delete_person/<int:pk>/', views.delete_person, name='delete_person'),
     path('klasifikasi.html', klasifikasi_view, name='klasifikasi'),
     path('evaluasi.html', evaluasi_view, name='evaluasi'),
     path('normalisasi.html', normalisasi_view, name='normalisasi'),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
