import json
import os
import requests
data_file = './description.json'
with open(data_file) as f:
    data = json.load(f)

home_page = data['homepage']
seqs = data['sequences']

for v in seqs:
    link = '%s%s' % (home_page,v['annotations']['url'])
    print('download %s' % link)
    os.system('wget %s -O %s_ann.zip' % (link, v['name']))
    link = '%s%s' % (home_page,v['channels']['color']['url'])
    print('download %s' % link)
    os.system('wget %s -O %s_chn.zip' % (link, v['name']))
