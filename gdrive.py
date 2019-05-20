import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd


def get_pat_stats():
    # get patient stats, ybocs ...
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('ocdmriml_gdrive_client_secret.json', scope)
    gsclient = gspread.authorize(creds)
    pat_stats_sheet = gsclient.open("AKSHAY_pat_stats").sheet1

    pat_frame_stats = pd.DataFrame(pat_stats_sheet.get_all_records())
    pat_frame_stats.index = pat_frame_stats.loc[:, 'subject']
    pat_frame_stats.drop(columns=['', 'subject'], inplace=True)

    return pat_frame_stats
