# Generated by Django 5.0.4 on 2024-04-28 14:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tagsapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='hashtagmetrics',
            name='relative_attraction',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
