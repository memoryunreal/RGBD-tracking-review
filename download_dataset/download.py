import requests
from lxml import etree


#请求头和目标网址
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36'
}
# url = 'https://www.jianshu.com/u/472a595d244c'
url = 'https://zenodo.org/record/5794115'

#第一种写法的 xpath
# xpath_link = '//ul[@class="note-list"]/li/div/a/@href'

# xpath_title = '//ul[@class="note-list"]/li/div/a/text()'
# xpath_comment_num = '//ul[@class="note-list"]/li/div/div[@class="meta"]/a[2]/text()'
# xpath_heart_num = '//ul[@class="note-list"]/li/div/div[@class="meta"]/span/text()'

#获取和解析网页
r = requests.get(url, headers=headers)
r.encoding = r.apparent_encoding
dom = etree.HTML(r.text)
import subprocess
xpath_link = '//*[@id="collapseTwo"]/table/tbody//*/span/a/@href'
links = dom.xpath(xpath_link)
base_url = 'https://zenodo.org'
print(links[0])
downlinks = [base_url + i for i in links]
import wget 
for link in downlinks:
    #wget.download(link)
    cmd = 'wget -c %s' % (link)
    subprocess.call(cmd, shell=True)
    print('download complete', link)

