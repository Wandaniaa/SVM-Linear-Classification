from django.db import models
from django.db.models.signals import pre_save
from django.dispatch import receiver

# Create your models here.
def get_next_person_number():
    # Mendapatkan nomor urutan berikutnya berdasarkan data yang sudah ada
    last_person = Person.objects.order_by('-no').first()
    if last_person:
        return last_person.no + 1
    else:
        return 1

class Person(models.Model):
    no = models.IntegerField(default=get_next_person_number, null=True)
    nama = models.CharField(max_length=100,null=True) 
    nik = models.CharField(max_length=100, default='', null=True)
    kecamatan = models.CharField(max_length=100, default='', null=True)
    desa = models.CharField(max_length=100, default='', null=True)
    pendapatan = models.DecimalField(max_digits=15, decimal_places=2, default=0, null=True)
    bobot_pendapatan = models.IntegerField(default=0, null=True)
    profesi_utama = models.CharField(max_length=100, default='', null=True)
    profesi_tambahan = models.CharField(max_length=100, default='', null=True)
    bobot_pt = models.IntegerField(default=0, null=True)
    pengalaman = models.IntegerField(default=0, null=True)
    bobot_pengalaman = models.IntegerField(default=0, null=True)
    status = models.CharField(max_length=100, default='', null=True)
    bobot_status = models.IntegerField(default=0, null=True)
    label = models.CharField(max_length=100, default='', null=True)
    bobot_pendapatan_input = models.IntegerField(blank=True, null=True)
    bobot_pt_input = models.IntegerField(blank=True, null=True)
    bobot_pengalaman_input = models.IntegerField(blank=True, null=True)
    bobot_status_input = models.IntegerField(blank=True, null=True)
    klasifikasi = models.CharField(max_length=20, blank=True, null=True)
    is_newly_added = models.BooleanField(default=False)
    hasil_klasifikasi = models.CharField(max_length=50, blank=True, null=True)  # Asumsikan hasil klasifikasi adalah CharField


    def __str__(self):
            return self.nama
    bobot_pendapatan_normalized = models.FloatField(null=True, blank=True)
    bobot_pt_normalized = models.FloatField(null=True, blank=True)
    bobot_pengalaman_normalized = models.FloatField(null=True, blank=True)
    bobot_status_normalized = models.FloatField(null=True, blank=True)

@receiver(pre_save, sender=Person)
def normalisasi_bobot(sender, instance, **kwargs):
    # Lakukan normalisasi di sini
    instance.bobot_pendapatan_normalized = normalize(instance.bobot_pendapatan, 1, 3, 0, 1)
    instance.bobot_pt_normalized = normalize(instance.bobot_pt, 1, 2, 0, 1)
    instance.bobot_pengalaman_normalized = normalize(instance.bobot_pengalaman, 1, 2, 0, 1)
    instance.bobot_status_normalized = normalize(instance.bobot_status, 1, 4, 0, 1)

def normalize(value, old_min, old_max, new_min, new_max):
    # Fungsi untuk melakukan normalisasi
    return (value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
                
    objects = models.Manager()
    # name = models.CharField(max_length=100)
    # age = models.IntegerField()
    # address = models.CharField(max_length=255)

  