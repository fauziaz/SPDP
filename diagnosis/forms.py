from django import forms

class DiagnosisForm(forms.Form):
    usia = forms.FloatField(
        label='Usia Ibu (tahun)',
        min_value=10, max_value=60,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: 28',
            'step': '1'
        })
    )
    bmi = forms.FloatField(
        label='BMI (kg/m²)',
        min_value=10, max_value=60,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: 22.5',
            'step': '0.1'
        })
    )
    usia_kehamilan = forms.FloatField(
        label='Usia Kehamilan (minggu)',
        min_value=1, max_value=45,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: 24',
            'step': '0.1'
        })
    )
    sistolik = forms.FloatField(
        label='Tekanan Sistolik (mmHg)',
        min_value=60, max_value=250,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: 120',
            'step': '1'
        })
    )
    diastolik = forms.FloatField(
        label='Tekanan Diastolik (mmHg)',
        min_value=40, max_value=180,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: 80',
            'step': '1'
        })
    )
    proteinuria = forms.BooleanField(
        label='Proteinuria (+)',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    diabetes = forms.BooleanField(
        label='Riwayat Diabetes',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    hipertensi = forms.BooleanField(
        label='Riwayat Hipertensi',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
