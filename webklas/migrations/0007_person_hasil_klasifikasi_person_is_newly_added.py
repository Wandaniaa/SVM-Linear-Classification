# Generated by Django 5.0 on 2024-03-04 16:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webklas', '0006_person_klasifikasi'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='hasil_klasifikasi',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='person',
            name='is_newly_added',
            field=models.BooleanField(default=False),
        ),
    ]
