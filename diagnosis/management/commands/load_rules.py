import json
import os
from django.conf import settings
from django.core.management.base import BaseCommand
from diagnosis.models import Rule

class Command(BaseCommand):
    help = 'Load rules from model/rules.json into database'

    def handle(self, *args, **kwargs):
        rules_path = os.path.join(settings.BASE_DIR, 'model', 'rules.json')
        self.stdout.write(f'Looking for rules at: {rules_path}')
        with open(rules_path) as f:
            rules = json.load(f)
        Rule.objects.all().delete()
        for r in rules:
            Rule.objects.create(
                kode_rule=r['kode_rule'],
                kondisi=r['kondisi'],
                risiko=r['risiko'],
                confidence=r.get('confidence', 1.0),
                support=r.get('support', 1),
            )
        self.stdout.write(self.style.SUCCESS(f'✅ Loaded {len(rules)} rules into DB'))
