import os
import re
import numpy as np
import pandas as pd
from time import sleep
from numpy import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document


# OPENAI api key
os.environ['OPENAI_API_KEY'] = 'XXXXXXXXXXXXXXXX'
# base url that will be recursively checked
START_URL = 'https://support.optimizely.com/'
# filter for specific URLs - use START_URL is no additional URL is needed
FILTER_URLS = 'https://support.optimizely.com/hc/en-us/'
# HTML class that contains the text in the URLs 
CLASS_HTML = 'lg:pl-5'
# output from linkchecker
OUTPUT_URLS = 'log_urls.csv'
# FAISS vector database name
OUTPUT_INDEX = 'faiss_search_index'
# final text and URL pairs - optional
OUTPUT_SOUP = 'final_soup.csv'
# only 1 thread is used here as Cloudflare was blocking more
WORKERS = 1


def scrape_docs(start_url, output_urls):
    '''
    Use linkchecker to recursiverly get all URL associated with the start_url
    Depth (parameter r) is here set to 2 levels
    '''

    try:
        os.system(f'linkchecker --r=2 --timeout=10 --threads=100 --no-warnings --output csv {start_url} > {output_urls}')
    except:
        print('Cannot scrape docs')


def load_output(output_urls, filter_urls):
    '''
    Load urls from linkchecker to be cleaned
    '''

    # ignore last summary row and first 3 lines of header
    df = pd.read_csv(output_urls, sep=';', skipfooter=1, skiprows=3, engine='python')
    # remove unnecessary urls
    df = df.tail(-2)
    # drop any duplicates
    df = df.drop_duplicates(subset=['urlname'])
    urls = df['urlname']
    # return only specific urlnames
    urls = urls[urls.str.contains(filter_urls)]

    return urls


def get_driver():
    ''''
    Create Chrome Driver with options for automation
    '''

    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--enable-javascript")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--nogpu")
    chrome_options.add_argument("--headless=false")
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent="Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 640 XL LTE) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Mobile Safari/537.36 Edge/12.10166')

    driver = webdriver.Chrome(options=chrome_options)

    return driver


def split_urls(file, workers):
    '''
    Split url list to be split for workers
    '''

    N = workers
    filenames = []
    for i, file in enumerate(np.array_split(file, N)):
        filename = f"urls_{i + 1}.csv"
        filenames.append(filename)
        file.to_csv(filename, index=False)

    return filenames


def get_handles(filename, driver):
    '''
    Get text elements, extract class_html from there, and clean spaces
    '''

    df = pd.read_csv(filename)

    soups = []
    for _, row in df.iterrows():
        try:
            url = row['urlname']
            driver.get(url)
            element_present = EC.presence_of_element_located((By.ID, 'page-container'))
            # wait 5 seconds at most to get element present
            WebDriverWait(driver, 5).until(element_present)
            # small random sleep to avoid large bursts of access
            sleep(random.uniform(1, 3))
            soup = BeautifulSoup(driver.page_source,"lxml")
            body_soup = soup.find("div", {"class": CLASS_HTML})
            clean_soup = re.sub("\s\s+", " ", body_soup.get_text())
            soups.append([clean_soup, url])
        except Exception:
            continue

    df_soup = pd.DataFrame(soups, columns=["page_content", "source"])
    df_soup.to_csv(filename.replace('urls', 'soups'), index = False)


def save_workers(filenames, output_soup_file):
    '''
    Save text elements from each worker
    '''

    dfs = [pd.read_csv(f.replace('urls', 'soups')) for f in filenames]
    # Combine the list of dataframes
    df = pd.concat(dfs, ignore_index=True)
    # save df to csv
    df.to_csv(f'{output_soup_file}', index = False)

    return df


def setup_workers(list_urls):
    '''
    Setup pool of threads
    '''

    files = split_urls(list_urls, WORKERS)
    drivers = [get_driver() for _ in range(WORKERS)]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        executor.map(get_handles, files, drivers)

    [driver.quit() for driver in drivers]

    df = save_workers(files, OUTPUT_SOUP)

    return df
def save_search_index(df, output_index):
    '''
    Create embeddings and save search index
    '''

    source_chunks = []
    # chunks of a maximum size of 1000 characters; no overlap
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=0)

    sources = [Document(page_content=x, metadata={"source": y}) for x, y in zip(df['page_content'], df['source'])]
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    search_index.save_local(output_index)


if __name__ == "__main__":

    if not os.path.exists(OUTPUT_URLS):
        # run the scrape to get all links; skip if it exists
        scrape_docs(START_URL, OUTPUT_URLS)

    # clean output from linkcheck, get only url names
    urls = load_output(OUTPUT_URLS, FILTER_URLS)

    # get CLASS_HTML element from all links
    final_df = setup_workers(urls)

    # save search index
    save_search_index(final_df, OUTPUT_INDEX)
