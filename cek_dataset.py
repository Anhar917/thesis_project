import numpy as np
import os

# Definisikan direktori tempat dataset disimpan
output_features_dir = 'anfis_extracted_features'

def load_and_inspect_dataset(dataset_name):
    """
    Memuat dan menampilkan informasi dasar dari dataset yang disimpan.
    """
    features_filepath = os.path.join(output_features_dir, f'anfis_features_{dataset_name}.npy')
    labels_filepath = os.path.join(output_features_dir, f'anfis_labels_{dataset_name}.npy')

    print(f"\n--- Memuat dan Memeriksa Dataset: '{dataset_name}' ---")

    # Periksa apakah file ada
    if not os.path.exists(features_filepath):
        print(f"ERROR: File fitur tidak ditemukan di: {features_filepath}")
        return
    if not os.path.exists(labels_filepath):
        print(f"ERROR: File label tidak ditemukan di: {labels_filepath}")
        return

    # Muat data
    X = np.load(features_filepath)
    y = np.load(labels_filepath)

    print(f"Dataset '{dataset_name}' berhasil dimuat.")
    print(f"Shape Fitur (X): {X.shape}")
    print(f"Shape Label (Y): {y.shape}")

    # Tampilkan beberapa sampel data (misalnya 5 baris pertama)
    print("\n--- Sampel Fitur (X) ---")
    if X.size > 0: # Periksa apakah array tidak kosong
        print(X[:20]) # Tampilkan 10 baris pertama
    else:
        print("Array fitur kosong.")

    print("\n--- Sampel Label (Y) ---")
    if y.size > 0: # Periksa apakah array tidak kosong
        print(y[:20]) # Tampilkan 10 elemen pertama
    else:
        print("Array label kosong.")

    # Tampilkan informasi tambahan untuk label
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\n--- Distribusi Label (Y) ---")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label} ( {'Asli' if label == 1 else 'Palsu' if label == 0 else 'Unknown'} ): {count} sampel")
    
    print(f"--- Selesai Memeriksa Dataset: '{dataset_name}' ---")

# --- Panggil fungsi untuk setiap dataset yang Anda miliki ---
if __name__ == "__main__":
    load_and_inspect_dataset('50k_depan')
    load_and_inspect_dataset('50k_belakang')
    load_and_inspect_dataset('100k_depan')
    load_and_inspect_dataset('100k_belakang')