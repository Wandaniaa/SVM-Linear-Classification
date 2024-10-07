from import_export import resources, fields
from import_export.widgets import ForeignKeyWidget
from .models import Person

class PersonResource(resources.ModelResource):
    class Meta:
        model = Person
        fields = ('no', 'nama', 'nik', 'kecamatan', 'desa', 'pendapatan', 'bobot_pendapatan', 'profesi_utama', 'profesi_tambahan', 'bobot_pt', 'pengalaman', 'bobot_pengalaman', 'status', 'bobot_status', 'label')
        import_id_fields = ['no']  # tentukan kolom sebagai identifikasi unik selama impor