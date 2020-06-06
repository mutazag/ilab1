import requests
import re 
from pathlib import Path
import os

def get_file_name_from_cd(cd):
    """ GET FILE NAME FORM CONTENT-DISPOSITION ATTRIBUTE OF RESPONSE HEADER
    Arguments:
        cd {string} -- content-disposition attribute of a response header, usually: r.headers.get('content-disposition')
    """
    if not cd: 
        return None
    fname = re.findall('filename=(.+)', cd) 
    if (fname) == 0: 
        return None
    return fname[0]

def get_file_name_from_resposne(r): 
    """ Get file name from response object
    
    Arguments:
        r {response} -- response object from a web request usually: r = requests.get(url, allow_redirects=True)
    """
    if not r: 
        return None
    return get_file_name_from_cd(r.headers.get())


def download_files(urls, folder):
    """download files form web
    
    Arguments:
        urls {list of urls} -- list of urls for files to be downloaded [url1, url2, url3]
        folder {string} -- folder spath for storing downlaoded file s
    """ 

    if not urls: 
        return None
    if not folder: 
        return None
    
    folder_path = Path(folder)
    if not folder_path.exists():
        os.makedirs(folder_path)

    