# Generated by Django 5.0.4 on 2024-05-01 10:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tagsapp', '0002_hashtagmetrics_relative_attraction'),
    ]

    operations = [
        migrations.AddField(
            model_name='hashtagmetrics',
            name='topic',
            field=models.CharField(default=0, max_length=255),
            preserve_default=False,
        ),
    ]
