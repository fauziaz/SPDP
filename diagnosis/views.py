import json
import joblib
import os
import numpy as np
from django.shortcuts import render, redirect
from django.db.models import Count
from django.http import JsonResponse
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from .forms import DiagnosisForm
from .models import Rule, DiagnosisResult
from .expert import run_diagnosis, GEJALA_LABEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# DIAGNOSIS
# ─────────────────────────────────────────────
def diagnosis_view(request):
    form = DiagnosisForm()
    result = None

    if request.method == 'POST':
        form = DiagnosisForm(request.POST)
        if form.is_valid():
            d = form.cleaned_data
            result = run_diagnosis(
                age=d['usia'],
                bmi=d['bmi'],
                systolic=d['sistolik'],
                diastolic=d['diastolik'],
                gest_age=d['usia_kehamilan'],
                proteinuria=d['proteinuria'],
                diabetes=d['diabetes'],
                hipertensi=d['hipertensi'],
            )
            # Simpan ke database
            DiagnosisResult.objects.create(
                usia=d['usia'],
                bmi=d['bmi'],
                usia_kehamilan=d['usia_kehamilan'],
                sistolik=d['sistolik'],
                diastolik=d['diastolik'],
                proteinuria=d['proteinuria'],
                diabetes=d['diabetes'],
                hipertensi=d['hipertensi'],
                gejala_aktif=','.join(result['gejala_set']),
                rule_aktif=','.join(r.kode_rule for r in result['active_rules']),
                rule_result=result['rule_result'] or '',
                ml_result=result['ml_result'],
                final_result=result['final_result'],
                confidence=result['confidence'],
            )

    return render(request, 'diagnosis.html', {'form': form, 'result': result})


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
def dashboard_view(request):
    total = DiagnosisResult.objects.count()
    dist = DiagnosisResult.objects.values('final_result').annotate(count=Count('final_result'))
    dist_dict = {d['final_result']: d['count'] for d in dist}
    recent = DiagnosisResult.objects.order_by('-created_at')[:10]

    for d in recent:
        d.gejala_list = d.gejala_aktif.split(',')

    context = {
        'total': total,
        'low_count': dist_dict.get('low', 0),
        'mid_count': dist_dict.get('mid', 0),
        'high_count': dist_dict.get('high', 0),
        'recent': recent,
    }
    return render(request, 'dashboard.html', context)


# ─────────────────────────────────────────────
# RULE BASE
# ─────────────────────────────────────────────
def rules_view(request):
    rules = sorted(
        Rule.objects.all(),
        key=lambda r: int(r.kode_rule.replace('R', ''))
    )

    for rule in rules:
        rule.kondisi_natural = [
            GEJALA_LABEL.get(g, g)
            for g in rule.kondisi.split('&')
        ]

    context = {
        'rules': rules,
        'gejala_label': GEJALA_LABEL
    }

    return render(request, 'rules.html', context)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluation_view(request):
    eval_path = os.path.join(BASE_DIR, 'model', 'eval_data.json')
    model_path = os.path.join(BASE_DIR, 'model', 'c45_model.pkl')

    with open(eval_path) as f:
        eval_data = json.load(f)

    pkg = joblib.load(model_path)
    model = pkg['model']
    encoder = pkg['encoder']

    X_test = np.array(eval_data['X_test'])
    y_true_raw = eval_data['y_test']
    y_true = encoder.transform(y_true_raw)
    y_pred = model.predict(X_test)

    # 1. Ambil urutan alfabetis bawaan mesin ('high', 'low', 'mid')
    classes = list(encoder.classes_)
    cm_raw = confusion_matrix(y_true, y_pred)
    
    # 2. Paksa ke urutan logis yang benar (LOW, MID, HIGH)
    desired_order = ['low', 'mid', 'high']
    
    # 3. Cari indeks perubahan posisinya
    idx_map = [classes.index(c) for c in desired_order]
    
    # 4. Susun ulang baris dan kolom matriks menggunakan Numpy
    cm_reordered = np.array(cm_raw)[idx_map, :][:, idx_map].tolist()
    classes_reordered = [c.upper() for c in desired_order]

    # Hitung metrik evaluasi
    acc   = round(accuracy_score(y_true, y_pred) * 100, 2)
    prec  = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    rec   = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    f1    = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)

    context = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cm': json.dumps(cm_reordered),
        'classes': json.dumps(classes_reordered),
        'classes_list': classes_reordered,
        'cm_list': cm_reordered,
        'total_test': len(y_true),
    }
    return render(request, 'evaluation.html', context)