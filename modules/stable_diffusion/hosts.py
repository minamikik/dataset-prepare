import requests
import json
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

def is_live(host):
    payload = {
        "prompt": 'test',
        "steps": 5,
        "width": 64,
        "height": 64,
        "batch_size": 1
    }
    try:
        r = requests.post(
            url=f'http://{host}/sdapi/v1/txt2img',
            data = json.dumps(payload),
            timeout=(3.0, 5.0)
            )
        if r.status_code == 200:
            logging.info(f'{host} {r.status_code} {r.reason}')
            return True
        else:
            logging.error(f'{host} {r.status_code} {r.reason}')
            return False
    except Exception as e:
        logging.error(f'{host} {e}')
        return False

def check_available_hosts(hosts):
    available_hosts = []
    for host in hosts:
        if is_live(host):
            available_hosts.append(host)
    return available_hosts