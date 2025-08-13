# Menggunakan image dasar Python versi 3.9 yang ringan.
FROM python:3.9-slim

# Menentukan direktori kerja di dalam container.
WORKDIR /app

# Menginstal dependensi sistem yang dibutuhkan oleh beberapa library Python.
# build-essential menyediakan compiler, libdbus-1-dev menyediakan library D-Bus.
RUN apt-get update && apt-get install -y \
    build-essential \
    libdbus-1-dev \
    libsystemd-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libsmbclient-dev \
    libcap-dev \
    libgl1

# Menyalin file requirements.txt ke dalam container.
COPY requirements.txt .

# Menginstal semua library Python yang ada di requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file proyek skripsi Anda ke dalam container.
COPY . .

# Menentukan perintah default saat container dimulai.
CMD ["python3", "main.py"]
