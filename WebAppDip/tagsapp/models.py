from django.db import models

class HashtagMetrics(models.Model):
    tag = models.CharField(max_length=255)
    mentions = models.IntegerField()
    upvotes = models.IntegerField()
    comments = models.IntegerField()
    first_post_time_hours_ago = models.IntegerField()
    relative_attraction = models.FloatField()
    topic = models.CharField(max_length=255)
    network = models.CharField(max_length=255)

    def __str__(self):
        return self.tag

class HashtagMetricsX(models.Model):
    tag = models.CharField(max_length=255)
    mentions = models.IntegerField()
    retweets = models.IntegerField()
    quotes = models.IntegerField()
    views = models.IntegerField()
    first_post_time_hours_ago = models.IntegerField()
    relative_attraction = models.FloatField()
    topic = models.CharField(max_length=255)
    network = models.CharField(max_length=255)

    def __str__(self):
        return self.tag

