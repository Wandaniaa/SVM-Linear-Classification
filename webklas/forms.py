from django import forms
from .models import Person 

# class ExcelUploadForm(forms.Form):
#     excel_file = forms.FileField()

class MyForm(forms.Form):
    myfile = forms.FileField()

class UploadFileForm(forms.Form):
    uji_coba = forms.FileField()

class EditPersonForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ['nama', 'nik', 'profesi_utama', 'bobot_pendapatan', 'bobot_pt', 'bobot_pengalaman', 'bobot_status', 'label']
        # fields = '__all__'

class AddPersonForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ['nama', 'nik', 'profesi_utama', 'bobot_pendapatan', 'bobot_pt', 'bobot_pengalaman', 'bobot_status', 'label', 'is_newly_added']
        