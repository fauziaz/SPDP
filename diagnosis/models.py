from django.db import models

class Rule(models.Model):
    kode_rule = models.CharField(max_length=10, unique=True)
    kondisi   = models.CharField(max_length=255)  # "G1&G5&G8"
    risiko    = models.CharField(max_length=10)   # low/mid/high
    confidence = models.FloatField(default=1.0)
    support   = models.IntegerField(default=1)

    def kondisi_list(self):
        return self.kondisi.split('&')

    def __str__(self):
        return f"{self.kode_rule}: IF {self.kondisi} THEN {self.risiko.upper()}"

    class Meta:
        ordering = ['kode_rule']


class DiagnosisResult(models.Model):
    RISK_CHOICES = [('low','LOW RISK'),('mid','MID RISK'),('high','HIGH RISK')]

    created_at    = models.DateTimeField(auto_now_add=True)
    usia          = models.FloatField()
    bmi           = models.FloatField()
    usia_kehamilan = models.FloatField()
    sistolik      = models.FloatField()
    diastolik     = models.FloatField()
    proteinuria   = models.BooleanField(default=False)
    diabetes      = models.BooleanField(default=False)
    hipertensi    = models.BooleanField(default=False)

    gejala_aktif  = models.CharField(max_length=200, blank=True)  # "G2,G5,G8"
    rule_aktif    = models.CharField(max_length=500, blank=True)  # "R1,R3"
    rule_result   = models.CharField(max_length=10, blank=True)
    ml_result     = models.CharField(max_length=10, blank=True)
    final_result  = models.CharField(max_length=10, choices=RISK_CHOICES)
    confidence    = models.FloatField(default=0.0)

    def __str__(self):
        return f"[{self.created_at:%Y-%m-%d %H:%M}] {self.final_result.upper()}"
