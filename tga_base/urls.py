from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('webklas.urls')),
    path('index2.html', include('webklas.urls')),
    path('base.html', include('webklas.urls')),
    path('general.html', include('webklas.urls')),
    path('advanced.html', include('webklas.urls')),
    path('add_person.html', include('webklas.urls')),
    path('delete_person.html', include('webklas.urls')),
    path('edite_person.html', include('webklas.urls')),
    path('klasifikasi.html', include('webklas.urls')),
    path('evaluasi.html', include('webklas.urls')),


]

urlpatterns += static(settings.MEDIA_URL, document_ROOT=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_ROOT=settings.STATIC_ROOT)