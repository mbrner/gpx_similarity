"""Script to scrape gpx files from bikepacking.com.

The script worked in September 2020 but it depends on the html of the website.
Therefore it is probably going to break sooner or later.
"""
import pathlib

import click
import requests
from bs4 import BeautifulSoup


HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'}


def get_routes(num=1, url_pattern='https://bikepacking.com/routes/page/{num}'):
    routes = []
    while True:
        url = url_pattern.format(num=num)
        page = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            post_list = soup.find_all(lambda tag: tag.name == 'ul' and 
                                                  tag.get('class') == ['postlist'])[0]
        except IndexError:
            break
        else:
            posts = post_list.find_all(lambda tag: tag.name == 'li')
            for post in posts:
                href = post.find(lambda tag: tag.name == 'a').get('href')
                routes.append(href)
            num += 1
    return routes


def get_gpx(url, save_dir):
    page = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(page.content, 'html.parser')
    gpx_file = None
    if gps_tag := soup.find(lambda tag: tag.name == 'div' and tag.get('class') == ['gps']):
        if download_url := gps_tag.find(lambda tag: tag.name == 'a' and tag.get('download') is not None):
            download_url = download_url.get('href')
            if download_url is not None:
                filename = download_url.split('/')[-1]
                r = requests.get(download_url, allow_redirects=True, headers=HEADERS)            
                if r.status_code == 200:
                    gpx_file = save_dir / filename
                    with gpx_file.open('wb') as stream:
                        stream.write(r.content)
    return gpx_file


@click.command()
@click.argument('save_dir',  type=click.Path())
def main(save_dir):
    """Download gpx files to SAVE_DIR."""
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f'Going to download gpx files form bikepacking.com and to {save_dir}')
    click.echo('Fetching routes...')
    routes = get_routes()
    click.echo('Start downloading...')
    def show_item(item):
        if item is not None:
            return f'url: {item}'
        else:
            return '-'

    with click.progressbar(routes, item_show_func=show_item, show_pos=True) as bar:
        for route_base_url in bar:
            get_gpx(route_base_url, save_dir)


if __name__ == '__main__':
    main()