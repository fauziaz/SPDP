"""
Script untuk melatih model C4.5 dan mengekstrak rule dari dataset.
Jalankan sekali sebelum atau saat startup: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATASET
# ─────────────────────────────────────────────
def load_data():
    train = pd.read_excel(os.path.join(BASE_DIR, 'train_dataset.xlsx'))
    test  = pd.read_excel(os.path.join(BASE_DIR, 'test_dataset.xlsx'))
    df = pd.concat([train, test], ignore_index=True)
    df['Risk_level'] = df['Risk_level'].str.strip().str.lower()
    df = df[df['Risk_level'].isin(['low','mid','high'])].copy()
    df = df.dropna(subset=['Age (yrs)','BMI  [kg/m²]','Systolic BP','Diastolic BP',
                            'gestational age (weeks)','diabetes',
                            'History of hypertension (y/n)','Protien Uria'])
    for col in ['diabetes','History of hypertension (y/n)','Protien Uria']:
        df[col] = df[col].fillna(0).astype(int)
    return df

# ─────────────────────────────────────────────
# 2. GEJALA CONVERTER
# ─────────────────────────────────────────────
def convert_to_gejala(age, bmi, systolic, diastolic, gest_age,
                       proteinuria=0, diabetes=0, hipertensi=0):
    gejala = set()
    # Usia
    if age < 20:     gejala.add('G1')
    elif age <= 35:  gejala.add('G2')
    else:            gejala.add('G3')
    
    # BMI
    if bmi < 18.5:        gejala.add('G4')
    elif bmi <= 24.9:     gejala.add('G5')
    elif bmi <= 29.9:     gejala.add('G6')
    else:                 gejala.add('G7')
    
    # Sistolik
    if systolic < 120:    gejala.add('G8')
    elif systolic <= 129: gejala.add('G9')
    elif systolic <= 139: gejala.add('G10')
    else:                 gejala.add('G11')
    
    # Diastolik
    if diastolic < 80:    gejala.add('G12')
    elif diastolic <= 89: gejala.add('G13')
    else:                 gejala.add('G14')
    
    # Faktor risiko (MODIFIKASI NEGASI EKSPLISIT)
    if proteinuria: gejala.add('G15')
    else:           gejala.add('G20')

    if diabetes:    gejala.add('G16')
    else:           gejala.add('G21')

    if hipertensi:  gejala.add('G17')
    else:           gejala.add('G22')
    
    # Usia kehamilan
    if gest_age < 20:  gejala.add('G18')
    else:              gejala.add('G19')
    
    return gejala

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING → GEJALA BINARY
# ─────────────────────────────────────────────
ALL_GEJALA = [f'G{i}' for i in range(1, 23)] # DIUBAH DARI 20 MENJADI 23

# # ─────────────────────────────────────────────
# # 3. FEATURE ENGINEERING → GEJALA BINARY
# # ─────────────────────────────────────────────
# ALL_GEJALA = [f'G{i}' for i in range(1, 20)]

def df_to_gejala_matrix(df):
    rows = []
    for _, row in df.iterrows():
        g = convert_to_gejala(
            age=row['Age (yrs)'],
            bmi=row['BMI  [kg/m²]'],
            systolic=row['Systolic BP'],
            diastolic=row['Diastolic BP'],
            gest_age=row['gestational age (weeks)'],
            proteinuria=int(row['Protien Uria']),
            diabetes=int(row['diabetes']),
            hipertensi=int(row['History of hypertension (y/n)'])
        )
        rows.append({gi: (1 if gi in g else 0) for gi in ALL_GEJALA})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# 4. EXTRACT RULES FROM DECISION TREE
# ─────────────────────────────────────────────
def extract_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left: feature <= threshold (0, gejala tidak ada)
            recurse(tree_.children_left[node],  conditions + [(name, '==', 0)])
            # Right: feature > threshold (1, gejala ada)
            recurse(tree_.children_right[node], conditions + [(name, '==', 1)])
        else:
            value = tree_.value[node][0]
            class_idx = np.argmax(value)
            predicted_class = class_names[class_idx]
            confidence = value[class_idx] / value.sum()
            # Hanya ambil kondisi positif (gejala yang ADA)
            pos_conditions = [c[0] for c in conditions if c[2] == 1]
            if pos_conditions:
                rules.append({
                    'conditions': pos_conditions,
                    'risiko': predicted_class,
                    'confidence': round(confidence, 4),
                    'support': int(value.sum())
                })

    recurse(0, [])
    return rules

# ─────────────────────────────────────────────
# 5. DEDUPLICATE & FORMAT RULES
# ─────────────────────────────────────────────
def deduplicate_rules(rules):
    seen = {}
    for r in rules:
        key = ('&'.join(sorted(r['conditions'])), r['risiko'])
        if key not in seen or r['support'] > seen[key]['support']:
            seen[key] = r
    result = []
    for i, (_, r) in enumerate(seen.items(), start=1):
        result.append({
            'kode_rule': f'R{i}',
            'kondisi': '&'.join(sorted(r['conditions'])),
            'risiko': r['risiko'],
            'confidence': r['confidence'],
            'support': r['support']
        })
    return result

# ─────────────────────────────────────────────
# 6. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_and_save():
    print("Loading data...")
    df = load_data()
    print(f"Total records: {len(df)}")
    print(f"Risk distribution:\n{df['Risk_level'].value_counts()}")

    X = df_to_gejala_matrix(df)
    y = df['Risk_level']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    print(f"\nTraining Decision Tree C4.5 (entropy)...")
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42
    )
    clf.fit(X, y_enc)

    # Save model + encoder
    model_path = os.path.join(BASE_DIR, 'model', 'c45_model.pkl')
    joblib.dump({'model': clf, 'encoder': le, 'features': ALL_GEJALA}, model_path)
    print(f"Model saved → {model_path}")

    # Extract rules
    rules_raw = extract_rules(clf, ALL_GEJALA, class_names)
    rules = deduplicate_rules(rules_raw)
    print(f"Extracted {len(rules)} unique rules")

    rules_path = os.path.join(BASE_DIR, 'model', 'rules.json')
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)
    print(f"Rules saved → {rules_path}")

    # Save test data for evaluation
    df_test = pd.read_excel(os.path.join(BASE_DIR, 'test_dataset.xlsx'))
    df_test['Risk_level'] = df_test['Risk_level'].str.strip().str.lower()
    df_test = df_test[df_test['Risk_level'].isin(['low','mid','high'])].copy()
    X_test = df_to_gejala_matrix(df_test)
    y_test = df_test['Risk_level']
    eval_data = {
        'X_test': X_test.values.tolist(),
        'y_test': y_test.tolist(),
        'features': ALL_GEJALA
    }
    eval_path = os.path.join(BASE_DIR, 'model', 'eval_data.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f)
    print(f"Eval data saved → {eval_path}")

    return rules

if __name__ == '__main__':
    train_and_save()
    print("\n✅ Training complete!")
