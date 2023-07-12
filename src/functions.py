import os
import shutil
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SU


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            print("Failed to call {}".format(url))
            raise ValueError(r.text)
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def last_sunday(date):
    return date + relativedelta(weekday=SU(-1))


def download_simulation_data(start, end, lake, api, folder, max_iter=52):
    if end < start:
        raise ValueError("Start date must be before end date.")
    path = os.path.join(folder, lake)
    os.makedirs(path, exist_ok=True)
    date = last_sunday(start)
    files = [date]
    i = 0
    while date + timedelta(days=7) < end and i < max_iter:
        i = i + 1
        date = date + timedelta(days=7)
        files.append(date)

    file_paths = []
    for file in files:
        out_file = os.path.join(path, "{}.nc".format(file.strftime('%Y%m%d')))
        file_paths.append(out_file)
        if os.path.exists(out_file):
            print("Existing file: {}".format(out_file))
        else:
            print("Downloading file: {}".format(out_file))
            download_file(api.format(lake, file.strftime('%Y%m%d')), out_file)

    return file_paths

