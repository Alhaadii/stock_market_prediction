# Generated by Django 5.0.2 on 2024-07-29 11:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('username', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('email', models.EmailField(max_length=100, unique=True)),
                ('password', models.CharField(max_length=128)),
            ],
        ),
    ]
