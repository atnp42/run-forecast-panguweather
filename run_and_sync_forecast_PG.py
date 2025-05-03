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

def subset_and_zip(grib_path):
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

    zip_name = Path(grib_path).with_suffix('.zip').name
    zip_output_path = Path(LOCAL_RESULTS_PATH) / zip_name

    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for nc_file in nc_files:
            zipf.write(nc_file, arcname=nc_file.name)

    print(f"[ZIP] Created zip archive: {zip_output_path}")

    return zip_output_path, temp_dir

def upload_and_cleanup(zip_path, dropbox_path, grib_path, temp_dir):
    print(f"[UPLOAD] Uploading {zip_path} to {dropbox_path}")
    with open(zip_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
    print(f"[UPLOAD] Upload complete: {dropbox_path}")

    # Clean up local files
    os.remove(zip_path)
    print(f"[CLEANUP] Deleted zip: {zip_path}")

    for nc_file in temp_dir.glob("*.nc"):
        nc_file.unlink()
    temp_dir.rmdir()
    print(f"[CLEANUP] Deleted temp nc files and folder: {temp_dir}")

    if os.path.exists(grib_path):
        os.remove(grib_path)
        print(f"[CLEANUP] Deleted GRIB file: {grib_path}")

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

        if forecast_count >= 1:
            try:
                zip_path, temp_dir = subset_and_zip(output_path)
                dropbox_target = f"{DROPBOX_RESULTS_PATH}/{zip_path.name}"
                upload_and_cleanup(zip_path, dropbox_target, output_path, temp_dir)
            except Exception as e:
                print(f"[ERROR] Error during processing/uploading: {e}")

        forecast_count += 1
        start_date += timedelta(days=1)

# === Main ===

if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    run_forecasts()

    print("[DONE] All forecasts and uploads completed.")
