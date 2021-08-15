##
from bs4 import BeautifulSoup
import requests
import re
import os
from IPython.display import display

##
def find_files(url, headers):
    # get a soup of the directory url
    soup = BeautifulSoup(requests.get(url, auth=(headers['user'], headers['passwd'])).text,
                         features="html.parser")
    # make a list of all the links in the url
    hrefs_list = []
    for link in soup.find_all('a'):
        hrefs_list.append(link.get('href'))

    return hrefs_list


def download_file(download_file_url, file_path, headers, output=False):
    if output:
        # print it is downloading
        print('Downloading: ' + download_file_url)
    # download the file to the directory
    r = requests.get(download_file_url, auth=(headers['user'], headers['passwd']))
    with open(file_path, 'wb') as f:
        f.write(r.content)


# needs a directory to download it to
def download_TUH(DIR, headers, sub_dir, output=False):
    # directory url
    dir_url = 'https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.0/' + sub_dir

    hrefs_dir_list = find_files(dir_url, headers)

    # for each link in the directory
    for link in hrefs_dir_list:
        # download the files outside of participant folders we want
        if re.findall('.xlsx|\.edf|\.tse(?!_)', str(link)):
            # if the file doesnt already exist in the directory
            if not os.path.exists(os.path.join(DIR, link)):
                download_file(dir_url + '/' + str(link), DIR + '/' + str(link), headers, output)


##
from getpass import getpass
import os
import sys
import os
from bs4 import BeautifulSoup
import requests
import re
import wget
import zipfile

DOWNLOAD_DIR = "/root/kistoff2/dataset/TUH Database"

if not os.path.exists(DOWNLOAD_DIR):
  os.makedirs(DOWNLOAD_DIR)

# user = getpass('TUH Username: ')
# key = getpass('TUH Password: ')
auth_dict = {'user': 'nedc', 'passwd': 'nedc_resources'}

download_TUH(DOWNLOAD_DIR, auth_dict, '_DOCS', output=True)


##
import pandas as pd
seiz_types_path = '/root/kistoff2/dataset/TUH Database/seizures_types_v01.xlsx'
seiz_types = pd.read_excel(seiz_types_path)

seiz_types = seiz_types.set_index('Class Code')
display(seiz_types)