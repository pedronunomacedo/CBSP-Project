import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
import json


def download_file(session, link, path):
    r = session.get(link, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r:
                f.write(chunk)


base_url = "https://spotdownloader.com/"
form_url = "https://spotdownloader.com/submit-form-url"  # Replace this with the actual action URL of the form


# create a new Firefox session
driver = webdriver.Firefox()
driver.implicitly_wait(30)
driver.get(url)

with requests.Session() as session:
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Assuming 'downloader' is an id; replace with the correct identifier
    search_section = soup.find(id="downloader")
    if search_section:
        search_input = search_section.find('input')
        submit_button = search_section.find('button')
        if search_input and 'name' in search_input.attrs:
            input_name = search_input['name']
            # Simulate pasting content into the input field and submitting the form
            payload = {input_name: "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF"}
            response = session.post(form_url, data=payload)
            print("Form submitted. Response status:", response.status_code)
        else:
            print("Input field not found or has no name attribute.")
    else:
        print("Downloader section not found.")

    link = soup.select_one(".onerror a")['href']
    flash_url = urljoin(response.url, link)

    response = session.get(flash_url)
    soup = BeautifulSoup(response.content, "html.parser")
    mp3_link = soup.select_one("param[name=flashvars]")['value'].split("url=", 1)[-1]
    print(mp3_link)

    download_file(session, mp3_link, "download.mp3")