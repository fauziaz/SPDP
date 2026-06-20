import joblib
import pprint

# Alamat menuju file pkl kamu
path_file = 'model/c45_model.pkl'

print("Membongkar file:", path_file)
print("-" * 30)

try:
    # Buka brankasnya
    isi_pkl = joblib.load(path_file)
    
    # Karena dari kodemu sebelumnya terlihat ini adalah Dictionary
    if isinstance(isi_pkl, dict):
        print("Isi file ini berupa Dictionary dengan kunci (keys):", isi_pkl.keys())
        print("-" * 30)
        
        # Mari kita intip isi fiturnya (features)
        if 'features' in isi_pkl:
            print(">> FITUR YANG DIGUNAKAN (features):")
            pprint.pprint(isi_pkl['features'])
            print("-" * 30)
            
        # Mari kita intip parameternya jika ada modelnya
        if 'model' in isi_pkl:
            print(">> KONFIGURASI MODEL C4.5:")
            print(isi_pkl['model'])
            print("Parameter model:", isi_pkl['model'].get_params())
            
        # Intip encoder
        if 'encoder' in isi_pkl:
            print("-" * 30)
            print(">> KELAS ENCODER (Target/Label):")
            print(isi_pkl['encoder'].classes_)
            
    else:
        # Jika ternyata bukan dictionary
        print("Isi file (Tipe: {}):".format(type(isi_pkl)))
        print(isi_pkl)

except Exception as e:
    print("Gagal membuka file! Error:", e)