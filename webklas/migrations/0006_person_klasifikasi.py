# Generated by Django 5.0 on 2024-03-04 15:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webklas', '0005_person_bobot_pendapatan_input_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='klasifikasi',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
