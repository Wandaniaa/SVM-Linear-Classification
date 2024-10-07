from django.contrib import admin
from .models import Person


# Register your models here.
class PersonAdmin(admin.ModelAdmin):
    actions = ['delete_selected']

    def delete_selected(self, request, queryset):
        # Menghapus data yang terpilih
        queryset.delete()

admin.site.register(Person, PersonAdmin)