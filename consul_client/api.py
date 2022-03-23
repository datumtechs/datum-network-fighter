import re
import consul
import base64


class ConsulApi(object):
    def __init__(self,
                 host='127.0.0.1',
                 port=8500,
                 token=None,
                 scheme='http',
                 consistency='default',
                 dc=None,
                 verify=True,
                 cert=None,
                 **kwargs):
        self.c = consul.Consul(
            host, port, token, scheme, consistency, dc, verify, cert, **kwargs
        )
        self.cfg = None
        self.service_id = ''

    def headers(self):
        token = None
        headers = {}
        token = token or self.c.token
        if token:
            headers['X-Consul-Token'] = token
        return headers

    def query_service_info_by_filter(self, filter_str):
        """
        filter_str detail please see https://www.consul.io/api-docs/agent/service#ns
        filter_str example:
                          Service==jobNode
                          via in Tags
                          ID=="jobNode_192.168.21.188_50053
        """
        result = self.c.http.get(consul.base.CB.json(),
                                 path='/v1/agent/services',
                                 params=[('filter', filter_str)],
                                 headers=self.headers())
        if not result:
            return False, None
        if len(result) > 1:
            return False, f'The number of query results is greater than one, and the query condition is {filter_str}'
        # result, *_ = list(result.values())
        # return f'{result["Address"]}:{result["Port"]}'
        return True, result

    def get_kv(self, flag):
        result = self.c.http.get(consul.base.CB.json(),
                                 path=f'/v1/kv/{flag}',
                                 headers=self.headers())
        connection, *_ = result
        return base64.b64decode(connection["Value"]).decode()

    def check_service_config(self):
        try:
            consul_config = self.cfg['consul']
            ip, port = self.cfg['bind_ip'], int(self.cfg['port'])
            name, tag = consul_config['name'], consul_config['tag']
            interval, deregister = consul_config['interval'], consul_config['deregister']
            str_re = '^(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|[1-9])\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)$'
            if not re.compile(str_re).match(ip):
                raise Exception(f'IP address {ip} is illegal')
            if not 1024 < port <= 65535:
                raise Exception(f'PORT {port} illegal,port 1024~65535')
            if name not in ['dataService', 'jobService', 'carrierService']:
                raise Exception(f'Found the wrong service type {name}, which can only be data or compute')
            return ip, port, name, tag, interval, deregister
        except Exception as e:
            raise Exception(f'check_service_config exception is:{e}')

    def register(self, cfg):
        self.cfg = cfg
        ip, port, name, tag, interval, deregister = self.check_service_config()
        print(ip, port, name, tag, interval, deregister)
        check = consul.Check().grpc(
            grpc=f'{ip}:{port}/{name}',
            interval=interval,
            deregister=deregister
        )
        self.service_id = f'{name}_{ip}_{port}'
        params = {
            'name': name,
            'service_id': self.service_id,
            'address': ip,
            'tags': [tag],
            'port': int(port),
            'check': check
        }
        return self.c.agent.service.register(**params)

    def stop(self, service_id=None):
        if not service_id:
            service_id = self.service_id
        result, info = self.query_service_info_by_filter(f'ID=="{service_id}"')
        if result:
            if self.c.agent.service.deregister(service_id):
                return f"stop {self.service_id} fail!"
            else:
                return f"stop {self.service_id} successful!"
        return info


def get_consul_client_obj(cfg_info):
    if cfg_info.get('consul', None) is None:
        raise Exception('consul is None')
    service_ip = cfg_info['consul'].get('service_ip', None)
    if not service_ip:
        raise Exception('service_ip is None')
    service_port = cfg_info['consul'].get('service_port', None)
    if service_port:
        obj = ConsulApi(host=service_ip, port=service_port)
    else:
        obj = ConsulApi(host=service_ip)
    return obj
