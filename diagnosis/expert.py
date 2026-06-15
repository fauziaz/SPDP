import joblib
import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'c45_model.pkl')

ALL_GEJALA = [f'G{i}' for i in range(1, 20)]

GEJALA_LABEL = {
    'G1':  'Usia < 20 tahun',
    'G2':  'Usia 20-35 tahun',
    'G3':  'Usia > 35 tahun',
    'G4':  'BMI < 18.5 (Underweight)',
    'G5':  'BMI 18.5-24.9 (Normal)',
    'G6':  'BMI 25-29.9 (Overweight)',
    'G7':  'BMI ≥ 30 (Obesitas)',
    'G8':  'Sistolik < 120 mmHg (Normal)',
    'G9':  'Sistolik 120-129 mmHg (Elevated)',
    'G10': 'Sistolik 130-139 mmHg (HT Stage 1)',
    'G11': 'Sistolik ≥ 140 mmHg (HT Stage 2)',
    'G12': 'Diastolik < 80 mmHg (Normal)',
    'G13': 'Diastolik 80-89 mmHg (HT Stage 1)',
    'G14': 'Diastolik ≥ 90 mmHg (HT Stage 2)',
    'G15': 'Proteinuria (+)',
    'G16': 'Riwayat Diabetes',
    'G17': 'Riwayat Hipertensi',
    'G18': 'Usia Kehamilan < 20 minggu',
    'G19': 'Usia Kehamilan ≥ 20 minggu',
}

_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def convert_to_gejala(age, bmi, systolic, diastolic, gest_age,
                       proteinuria=False, diabetes=False, hipertensi=False):
    gejala = set()
    if age < 20:      gejala.add('G1')
    elif age <= 35:   gejala.add('G2')
    else:             gejala.add('G3')
    if bmi < 18.5:        gejala.add('G4')
    elif bmi <= 24.9:     gejala.add('G5')
    elif bmi <= 29.9:     gejala.add('G6')
    else:                 gejala.add('G7')
    if systolic < 120:    gejala.add('G8')
    elif systolic <= 129: gejala.add('G9')
    elif systolic <= 139: gejala.add('G10')
    else:                 gejala.add('G11')
    if diastolic < 80:    gejala.add('G12')
    elif diastolic <= 89: gejala.add('G13')
    else:                 gejala.add('G14')
    if proteinuria: gejala.add('G15')
    if diabetes:    gejala.add('G16')
    if hipertensi:  gejala.add('G17')
    if gest_age < 20:  gejala.add('G18')
    else:              gejala.add('G19')
    return gejala


def forward_chaining(gejala_set, rules_qs):
    active_rules = []
    risk_votes = {'low': 0, 'mid': 0, 'high': 0}
    for rule in rules_qs:
        conditions = rule.kondisi_list()
        if all(c in gejala_set for c in conditions):
            active_rules.append(rule)
            risk_votes[rule.risiko] += 1
    if not active_rules:
        return None, []
    priority = {'high': 3, 'mid': 2, 'low': 1}
    final = max(risk_votes, key=lambda r: (risk_votes[r], priority[r]))
    if risk_votes[final] == 0:
        return None, []
    return final, active_rules


def ml_predict(gejala_set):
    pkg = _load_model()
    model   = pkg['model']
    encoder = pkg['encoder']
    features = pkg['features']
    row = {g: (1 if g in gejala_set else 0) for g in features}
    X = pd.DataFrame([row], columns=features)
    pred_enc   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = float(proba.max())
    risk_label = encoder.inverse_transform([pred_enc])[0]
    return risk_label, confidence


def combine_result(rule_result, ml_result, ml_confidence):
    if rule_result is None:
        return ml_result, ml_confidence
    if rule_result == ml_result:
        return rule_result, ml_confidence
    return ml_result, ml_confidence


def run_diagnosis(age, bmi, systolic, diastolic, gest_age,
                  proteinuria, diabetes, hipertensi):
    from .models import Rule
    gejala_set = convert_to_gejala(age, bmi, systolic, diastolic, gest_age,
                                    proteinuria, diabetes, hipertensi)
    rules_all = Rule.objects.all()
    rule_result, active_rules = forward_chaining(gejala_set, rules_all)
    ml_result, ml_confidence  = ml_predict(gejala_set)
    final_result, confidence  = combine_result(rule_result, ml_result, ml_confidence)
    sorted_gejala = sorted(gejala_set, key=lambda g: int(g[1:]))
    return {
        'gejala_set': sorted_gejala,
        'gejala_labels': {g: GEJALA_LABEL.get(g, g) for g in sorted_gejala},
        'rule_result': rule_result,
        'active_rules': active_rules,
        'ml_result': ml_result,
        'ml_confidence': round(ml_confidence * 100, 1),
        'final_result': final_result,
        'confidence': round(confidence * 100, 1),
    }
