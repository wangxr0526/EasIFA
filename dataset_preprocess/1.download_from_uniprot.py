import re
import requests
from requests.adapters import HTTPAdapter, Retry

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)


# url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Ccc_interaction&format=tsv&query=Insulin%20AND%20%28reviewed%3Atrue%29&size=500'

# url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Corganism_name%2Clength%2Cec%2Cxref_alphafolddb%2Cxref_pdb&format=tsv&query=%28%28ec%3A%2A%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28organism_name%3A%2A%29%29&size=500' # no sequence

# url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Corganism_name%2Clength%2Cec%2Cxref_alphafolddb%2Cxref_pdb%2Csequence&format=tsv&query=%28%28ec%3A%2A%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28organism_name%3A%2A%29%29&size=500'

url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Corganism_name%2Clength%2Cec%2Cxref_alphafolddb%2Cft_act_site%2Cft_binding%2Cft_site%2Cxref_pdb%2Csequence&format=tsv&query=%28%28reviewed%3Atrue%29%29&size=500'

progress = 0
with open('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot-download_sequence_site.tsv', 'w') as f:
    for batch, total in get_batch(url):
        lines = batch.text.splitlines()
        if not progress:
            print(lines[0], file=f)
        for line in lines[1:]:
            print(line, file=f)
        progress += len(lines[1:])
        print(f'{progress} / {total}')