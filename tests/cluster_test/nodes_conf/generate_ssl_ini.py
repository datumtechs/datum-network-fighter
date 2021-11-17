import os
import argparse
from configparser import ConfigParser
scripts_path = os.path.split(os.path.realpath(__file__))[0]
base_path = os.path.join(scripts_path, '../../..')
import sys
sys.path.insert(0, base_path)
from common.utils import load_cfg

class CaseSenseConfigParser(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

ssl_cnf_file = f"{base_path}/third_party/gmssl/ssl.ini"
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='config.yaml')
parser.add_argument('--result', type=str, default='new_ssl.ini')
args = parser.parse_args()
cfg_file = args.cfg_file
result = args.result

cfg = load_cfg(cfg_file)
conf = CaseSenseConfigParser()
conf.read(ssl_cnf_file, encoding="utf-8")
for i,ip in enumerate(cfg["ip"]):
    conf.set("alt_names", f"IP.{i+1}", ip)
conf.write(open(result, 'w'))
print(f"write to {result} success.")