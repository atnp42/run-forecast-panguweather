import os
import time
import dropbox
import subprocess
import threading
import xarray as xr
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

# === Dropbox Setup ===
ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_panguweather/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/panguweather_results"
LOCAL_RESULTS_PATH = "/workspace/results"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)

# === Bounding Box (CONUS) ===
lat_min, lat_max = 15, 50
lon_min, lon_max = -130, -66

# === Hilfsfunktionen ===

def download_folder(dbx, dropbox_path, local_path):
    entries = dbx.files_list_folder(dropbox_path).entries
    for entry in entries:
        dp = f"{dropbox_path}/{entry.name}"
        lp = f"{local_path}/{entry.name}"

        if isinstance(entry, dropbox.files.FileMetadata):
            print(f"[ASSETS] Downloading asset: {dp}")
            with open(lp, "wb") as f:
                _, res = dbx.files_download(dp)
                f.write(res.content)
        elif isinstance(entry, dropbox.files.FolderMetadata):
            os.makedirs(lp, exist_ok=True)
            download_folder(dbx, dp, lp)

def subset_and_zip(grib_path, zip_output_path):
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        ds = ds.sortby("longitude")

    ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    temp_dir = Path(grib_path).with_suffix('')
    temp_dir.mkdir(parents=True, exist_ok=True)

    nc_files = []
    for var in ds_sub.data_vars:
        var_ds = ds_sub[[var]]
        nc_path = temp_dir / f"{var}.nc"
        var_ds.to_netcdf(nc_path)
        nc_files.append(nc_path)

    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for nc_file in nc_files:
            zipf.write(nc_file, arcname=nc_file.name)

    for nc_file in nc_files:
        nc_file.unlink()
    temp_dir.rmdir()

    print(f"[ZIP] Created zip archive: {zip_output_path}")

def upload_file_to_dropbox(local_file, dropbox_path):
    print(f"[UPLOAD] Uploading {local_file} to {dropbox_path}")
    with open(local_file, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
    print(f"[UPLOAD] Upload complete: {dropbox_path}")
    os.remove(local_file)

# === Forecast-Loop ===

def run_forecasts():
    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 2, 3)
    lead_time = 168
    time_str = "1200"
    model = "panguweather"

    forecast_count = 0

    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        output_filename = f"panguweather_{date_str}_{time_str}_{lead_time}h_gpu.grib"
        output_path = os.path.join(LOCAL_RESULTS_PATH, output_filename)

        command = [
            "ai-models",
            "--assets", LOCAL_ASSETS_PATH,
            "--path", output_path,
            "--input", "cds",
            "--date", date_str,
            "--time", time_str,
            "--lead-time", str(lead_time),
            model
        ]

        print(f"[FORECAST] Running forecast for {date_str}")
        subprocess.run(command)
        print(f"[FORECAST] Finished forecast for {date_str}")

        # Subset & zip from the second forecast onward
        if forecast_count >= 1:
            zip_name = Path(output_filename).with_suffix('.zip')
            zip_output_path = os.path.join(LOCAL_RESULTS_PATH, zip_name)
            subset_and_zip(output_path, zip_output_path)
            upload_file_to_dropbox(zip_output_path, f"{DROPBOX_RESULTS_PATH}/{zip_name}")

        os.remove(output_path)
        print(f"[CLEANUP] Deleted local GRIB file: {output_filename}\n")

        forecast_count += 1
        start_date += timedelta(days=1)

# === Main ===

if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    run_forecasts()

    print("[DONE] All forecasts and uploads completed.")
