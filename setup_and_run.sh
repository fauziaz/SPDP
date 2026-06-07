#!/bin/bash
# ======================================================
# SPDP — Setup & Run Script
# Sistem Pakar Deteksi Risiko Preeklampsia
# ======================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  SPDP — Sistem Pakar Deteksi Preeklampsia   ║"
echo "║           Setup & Run Script                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "📦 Installing Python dependencies..."
pip install django scikit-learn pandas openpyxl joblib numpy -q

# 2. Train model (generates model/c45_model.pkl and model/rules.json)
echo "🧠 Training C4.5 model & extracting rules..."
python train_model.py

# 3. Django migrations
echo "🗄️  Running database migrations..."
python manage.py makemigrations diagnosis --no-input
python manage.py migrate --no-input

# 4. Load rules into DB
echo "📚 Loading rules into database..."
python manage.py load_rules

# 5. Create superuser (optional)
echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting development server..."
echo "   Open: http://127.0.0.1:8000"
echo ""
python manage.py runserver
