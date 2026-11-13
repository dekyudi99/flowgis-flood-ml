# FlowGIS FloodRisk ML Microservice (`flowgis-flood-ml`)

Microservice berbasis **FastAPI (Python)** untuk melayani pemodelan risiko banjir spasiotemporal harian dan Sistem Peringatan Dini (Early Warning System - EWS) berbasis Machine Learning di Thailand (Sukhothai, Suphan Buri, dan Nakhon Pathom), terintegrasi dengan **FlowGIS Frontend** dan publikasi layer ke **GeoServer**.

---

## 🌟 Fitur Utama
1. **Model Adaptor Multi-Provinsi**:
   - **Sukhothai (SKT)**: Champion *LightGBM* berbasis Zero-Station (Satelit GEE & ERA5 Reanalysis).
   - **Suphan Buri (SPB)**: Champion *LightGBM* berbasis Zero-Station (Satelit GEE & ERA5 Reanalysis).
   - **Nakhon Pathom (NPT)**: Champion *CatBoost* diperkaya fitur telemetri stasiun muka air HII/TMD.
2. **Instant Preview Workflow**:
   - Inferensi 1.000 titik pantau per provinsi secara instan (<200 ms).
   - Menghasilkan format standar GeoJSON FeatureCollection lengkap dengan 5 tingkat bahaya warna (`Very Low` s/d `Very High`).
3. **On-Demand GeoServer Publication**:
   - Hasil analisis ditampilkan terlebih dahulu di peta antarmuka pengguna sebagai layer pratinjau.
   - Pengguna memilih apakah ingin menyimpan hasil ke GeoServer.
   - Mengunggah vektor dan mengaplikasikan style SLD 5 kelas warna secara otomatis.

---

## 🚀 Endpoint API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check & info sesi preview aktif |
| `GET` | `/api/v1/provinces` | Metadata provinsi, model, dan horison lead time |
| `POST` | `/api/v1/predict` | Menjalankan inferensi dan mengembalikan GeoJSON titik |
| `GET` | `/api/v1/preview/{id}` | Mengambil data GeoJSON dari sesi preview aktif |
| `POST` | `/api/v1/save-to-geoserver` | Menerbitkan layer aktif ke GeoServer |

---

## 💻 Menjalankan Service

### Secara Lokal
```bash
cd D:/Yudi-Intern/FlowGIS/flowgis-flood-ml
python -m venv .venv
source .venv/bin/activate # atau .venv\Scripts\activate di Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
```

Dokumentasi Swagger OpenAPI dapat diakses di: `http://localhost:8008/docs`

