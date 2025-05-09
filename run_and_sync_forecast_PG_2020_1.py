import os
import dropbox
import subprocess
import threading
import xarray as xr
import zipfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from eccodes import codes_grib_new_from_file, codes_get, codes_release

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def subset_and_upload(grib_path):
    temp_dir = Path(grib_path).with_suffix('')
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n[SUBSET] Starting full subsetting of: {grib_path}")
    with open(grib_path, 'rb') as f:
        seen = set()
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                type_of_level = codes_get(gid, 'typeOfLevel')
                level = codes_get(gid, 'level')
                short_name = codes_get(gid, 'shortName')

                key = (short_name, type_of_level, level)
                if key in seen:
                    continue
                seen.add(key)

                print(f"[SUBSET] Processing variable='{short_name}', typeOfLevel='{type_of_level}', level={level}")
                try:
                    ds = xr.open_dataset(
                        grib_path,
                        engine="cfgrib",
                        backend_kwargs={
                            "filter_by_keys": {
                                "typeOfLevel": type_of_level,
                                "level": int(level),
                                "shortName": short_name
                            },
                            "indexpath": ""
                        },
                        decode_times=True
                    )

                    if ds.longitude.max() > 180:
                        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
                        ds = ds.sortby("longitude")

                    ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    out_name = f"{short_name}_{int(level)}.nc"
                    out_path = temp_dir / out_name
                    ds_sub.to_netcdf(out_path)
                    print(f"[SUBSET] Saved: {out_path}")

                except Exception as e:
                    print(f"[WARNING] Skipped variable='{short_name}', typeOfLevel='{type_of_level}', level={level}: {e}")
            finally:
                codes_release(gid)

    zip_name = Path(grib_path).with_suffix('.zip').name
    zip_output_path = Path(LOCAL_RESULTS_PATH) / zip_name

    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for nc_file in temp_dir.glob("*.nc"):
            zipf.write(nc_file, arcname=nc_file.name)

    print(f"[ZIP] Created zip archive: {zip_output_path}")
    dropbox_target = f"{DROPBOX_RESULTS_PATH}/{zip_output_path.name}"

    print(f"[UPLOAD] Uploading {zip_output_path} to {dropbox_target}")
    with open(zip_output_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_target, mode=dropbox.files.WriteMode("overwrite"))
    print(f"[UPLOAD] Upload complete: {dropbox_target}")

    os.remove(zip_output_path)
    print(f"[CLEANUP] Deleted zip: {zip_output_path}")

    for nc_file in temp_dir.glob("*.nc"):
        nc_file.unlink()
    temp_dir.rmdir()
    print(f"[CLEANUP] Deleted temp nc files and folder: {temp_dir}")

    if os.path.exists(grib_path):
        os.remove(grib_path)
        print(f"[CLEANUP] Deleted GRIB file: {grib_path}")

def run_forecasts():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 6, 30)
    lead_time = 168
    time_str = "1200"
    model = "panguweather"

    forecast_queue = []
    next_date = start_date

    print("[FORECAST] Starting first two forecasts...")

    # Initial two forecasts
    for _ in range(2):
        date_str = next_date.strftime("%Y%m%d")
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

        forecast_queue.append(output_path)
        next_date += timedelta(days=1)

    # Loop through remaining forecasts, controlled by previous processing
    while next_date <= end_date:
        # Start subsetting + upload for the first in queue
        grib_to_process = forecast_queue.pop(0)
        print(f"[PROCESS] Starting processing of {grib_to_process}")
        subset_and_upload(grib_to_process)

        # After processing finishes, start the next forecast
        date_str = next_date.strftime("%Y%m%d")
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

        forecast_queue.append(output_path)
        next_date += timedelta(days=1)

    # Process remaining in queue
    while forecast_queue:
        grib_to_process = forecast_queue.pop(0)
        print(f"[PROCESS] Final processing of {grib_to_process}")
        subset_and_upload(grib_to_process)

if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    run_forecasts()

    print("[DONE] All forecasts and uploads completed.")
