# coding:utf-8

import time
import math
import queue
import logging
import psutil
import grpc
from common.consts import EVENT_QUEUE
from protos.lib.api import sys_rpc_api_pb2 as pb2
from protos.lib.api import sys_rpc_api_pb2_grpc as pb2_grpc


log = logging.getLogger(__name__)

class ReportEngine(object):

    def __init__(self, server_addr: str):
        self.conn = grpc.insecure_channel(server_addr)
        self.__client = pb2_grpc.YarnServiceStub(channel=self.conn)

    def report_task_event(self):
        """
        service YarnService {
            rpc ReportTaskEvent (ReportTaskEventRequest) returns (api.protobuf.SimpleResponse) {
                option (google.api.http) = {
                    post: "/carrier/v1/yarn/reportTaskEvent"
                    body: "*"
                };
            }
        }
        message ReportTaskEventRequest {
            types.TaskEvent task_event = 1;
        }
        message TaskEvent {
            string type = 1;
            string task_id = 2;
            string identity_id = 3;
            string party_id = 4;
            string content = 5;
            uint64 create_at = 6;
        }
        """
        try:
            # get event
            event = EVENT_QUEUE.get(block=True, timeout=1)
            req = pb2.ReportTaskEventRequest()
            req.task_event.type = event.type_
            req.task_event.task_id = event.dict_["task_id"]
            req.task_event.identity_id = event.dict_["identity_id"]
            req.task_event.party_id = event.dict_["party_id"]
            req.task_event.content = "{}:{}".format(event.dict_["party_id"], event.dict_["content"])
            req.task_event.create_at = event.dict_["create_at"]
            str_req = '{' + str(req).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
            log.debug(str_req)
            return self.__client.ReportTaskEvent(req)
        except queue.Empty as e:
            pass
        except Exception as e:
            log.exception(str(e))

    def report_upload_file_summary(self, summary):
        """
        service YarnService {
            rpc  ReportUpFileSummary (ReportUpFileSummaryRequest) returns (api.protobuf.SimpleResponse) {
                option (google.api.http) = {
                    post: "/carrier/v1/yarn/reportUpFileSummary"
                    body: "*"
                };
            }
        }
        message ReportUpFileSummaryRequest {
            string origin_id = 1;
            string file_path = 2;
            string ip = 3;
            string port = 4;
        }
        """
        try:
            req = pb2.ReportUpFileSummaryRequest()
            req.origin_id = summary["origin_id"]
            req.file_path = summary["file_path"]
            req.ip = summary["ip"]
            req.port = str(summary["port"])
            str_req = '{' + str(req).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
            log.debug(str_req)
            return self.__client.ReportUpFileSummary(req)
        except Exception as e:
            log.exception(str(e))

    def report_task_result_file_summary(self, summary):
        """
        service YarnService {
            rpc ReportTaskResultFileSummary (ReportTaskResultFileSummaryRequest) returns (api.protobuf.SimpleResponse) {
                option (google.api.http) = {
                post: "/carrier/v1/yarn/reportTaskResultFileSummary"
                body: "*"
                };
            }
        }
        message ReportTaskResultFileSummaryRequest {
            string task_id = 1;
            string origin_id = 2;
            string file_path = 3;
            string ip = 4;
            string port = 5;
        }
        """
        try:
            req = pb2.ReportTaskResultFileSummaryRequest()
            req.task_id = summary["task_id"]
            req.origin_id = summary["origin_id"]
            req.file_path = summary["file_path"]
            req.ip = summary["ip"]
            req.port = str(summary["port"])
            str_req = '{' + str(req).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
            log.debug(str_req)
            return self.__client.ReportTaskResultFileSummary(req)
        except Exception as e:
            log.exception(str(e))

    def report_task_resource_usage(self, task_id, party_id, node_type, ip:str, port:str, resource_usage):
        """
        service YarnService {
            rpc ReportTaskResourceUsage (ReportTaskResourceUsageRequest) returns (api.protobuf.SimpleResponse) {
                option (google.api.http) = {
                    post: "/carrier/v1/yarn/reportTaskResourceUsage"
                    body: "*"
                };
            }
        }
        message ReportTaskResourceUsageRequest {
            string                      task_id = 1;
            string                      party_id = 2;
            NodeType                    node_type = 3; 
            string                      ip = 4;
            string                      port = 5;
            types.ResourceUsageOverview usage = 6;
        }
        enum NodeType {
            NodeType_Unknown = 0;           // Unknown node
            NodeType_SeedNode = 1;          // Seed node
            NodeType_JobNode = 2;           // Compute node
            NodeType_DataNode = 3;          // Data node
            NodeType_YarnNode = 4;          // Schedule node
        }
        message ResourceUsageOverview {
            uint64 total_mem = 2;   // byte
            uint64 used_mem = 3;    // byte
            uint32 total_processor = 4;  // int
            uint32 used_processor = 5;   // int
            uint64 total_bandwidth = 6;  // bps
            uint64 used_bandwidth = 7;   // bps
            uint64 total_disk = 8;   // byte
            uint64 used_disk = 9;    // byte
        }
        """
        if node_type == "compute_svc":
            report_node_type = pb2.NodeType_JobNode
        elif node_type == "data_svc":
            report_node_type = pb2.NodeType_DataNode
        else:
            raise Exception("node_type only support compute_svc/data_svc")

        try:
            req = pb2.ReportTaskResourceUsageRequest()
            req.task_id = task_id
            req.party_id = party_id
            req.node_type = report_node_type
            req.ip = ip 
            req.port = str(port)
            req.usage.total_mem = resource_usage["total_mem"]
            req.usage.used_mem = resource_usage["used_mem"]
            req.usage.total_processor = resource_usage["total_processor"]
            req.usage.used_processor = resource_usage["used_processor"]
            req.usage.total_bandwidth = resource_usage["total_bandwidth"]
            req.usage.used_bandwidth = resource_usage["used_bandwidth"]
            req.usage.total_disk = resource_usage["total_disk"]
            req.usage.used_disk = resource_usage["used_disk"]
            str_req = '{' + str(req).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
            log.debug(str_req)
            return self.__client.ReportTaskResourceUsage(req)
        except Exception as e:
            log.exception(str(e))
        
    def close(self):
        self.conn.close()


def report_task_event(server_addr: str, stop_event):
    report_engine = ReportEngine(server_addr)
    try:
        while True:
            report_engine.report_task_event()
            if stop_event.is_set():
                break
    except Exception as e:
        log.exception(f"report event error. {str(e)}")
    finally:
        report_engine.close()
        log.info('report_engine closed')


def report_upload_file_summary(server_addr: str, summary: dict):
    report_engine = ReportEngine(server_addr)
    ret = report_engine.report_upload_file_summary(summary)
    report_engine.close()
    return ret

def report_task_result_file_summary(server_addr: str, summary: dict):
    report_engine = ReportEngine(server_addr)
    ret = report_engine.report_task_result_file_summary(summary)
    report_engine.close()
    return ret

def _get_resource_usage(task_pid, total_bandwidth):
    p = psutil.Process(task_pid)
    resource_usage = {}
    resource_usage["total_mem"] = psutil.virtual_memory().total
    resource_usage["used_mem"] = p.memory_info().rss
    resource_usage["total_processor"] = psutil.cpu_count()
    resource_usage["used_processor"] = math.ceil(psutil.cpu_count() * p.cpu_percent() / 100)
    resource_usage["total_bandwidth"] = total_bandwidth
    net_1 = psutil.net_io_counters()
    time.sleep(1)
    net_2 = psutil.net_io_counters()
    resource_usage["used_bandwidth"] = (net_2.bytes_sent - net_1.bytes_sent) + (net_2.bytes_recv - net_1.bytes_recv)
    resource_usage["total_disk"] = psutil.disk_usage('/').total
    resource_usage["used_disk"] = psutil.disk_usage('/').used
    return resource_usage

def report_task_resource_usage(task_pid, server_addr:str, task_id, party_id, node_type, ip, port:str, total_bandwidth, interval=10):
    report_engine = ReportEngine(server_addr)
    try:
        while True:
            resource_usage = _get_resource_usage(task_pid, total_bandwidth)
            report_engine.report_task_resource_usage(task_id, party_id, node_type, ip, port, resource_usage)
            time.sleep(interval)
    except Exception as e:
        log.exception(f"report resource usage error. {str(e)}")
    finally:
        report_engine.close()
        log.info('report_engine closed')
