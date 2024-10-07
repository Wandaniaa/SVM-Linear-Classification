# Generated by Django 5.0 on 2024-02-28 11:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webklas', '0003_alter_person_bobot_pendapatan_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='bobot_pendapatan_normalized',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='person',
            name='bobot_pengalaman_normalized',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='person',
            name='bobot_pt_normalized',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='person',
            name='bobot_status_normalized',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
