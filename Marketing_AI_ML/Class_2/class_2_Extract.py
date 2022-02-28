
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


def extract():
    url = 'https://www.worldometers.info/world-population/population-by-country/'

    request = requests.get(url)

    soup = BeautifulSoup(request.text, 'lxml')
    table1 = soup.find('table', id='example2')

    # Obtain every title of columns with tag <th>
    headers = []
    for i in table1.find_all('th'):
        title = i.text
        headers.append(title)

    # Create a dataframe
    mydata = pd.DataFrame(columns = headers)

    # Create a for loop to fill mydata
    for j in table1.find_all('tr')[1:]:
        row_data = j.find_all('td')
        row = [i.text for i in row_data]
        length = len(mydata)
        mydata.loc[length] = row

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    save_loc = mydata.to_csv(dir_path+"\df_csv.csv")
    return "file created to CSV"

def API():
    wsj_token = os.environ.get('WSJ_TOKEN')
    url='https://api.nytimes.com/svc/community/v3/user-content/url.json?api-key={}&offset=0&url=https%3A%2F%2Fwww.nytimes.com%2F2019%2F06%2F21%2Fscience%2Fgiant-squid-cephalopod-video.html'.format(wsj_token)
    wsj_community = requests.get(url)
    return wsj_community.text

if __name__ == '__main__':
    print(extract())
    print(API())