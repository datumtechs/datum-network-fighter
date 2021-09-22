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
        self.conn = grpc.insecure_channel(server_addr)  # 连接调度节点的ip和port
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
            string          party_id = 1;
            types.TaskEvent task_event = 2;
        }
        message TaskEvent {
            string type = 1;
            string task_id = 2;
            string identity_id = 3;
            string content = 4;
            uint64 create_at = 5;
        }
        """
        try:
            # get event
            event = EVENT_QUEUE.get(block=True, timeout=1)
            req = pb2.ReportTaskEventRequest()
            # req.party_id = event.dict_["party_id"]
            req.task_event.type = event.type_
            req.task_event.task_id = event.dict_["task_id"]
            req.task_event.identity_id = event.dict_["identity_id"]
            req.task_event.content = "{}:{}".format(event.dict_["party_id"], event.dict_["content"])
            req.task_event.create_at = event.dict_["create_at"]
            self.__client.ReportTaskEvent(req)
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
            return self.__client.ReportUpFileSummary(req)
        except Exception as e:
            log.exception(str(e))

    def report_task_resource_expense(self, node_type, node_id:str, resource_usage):
        """
        service YarnService {
            rpc ReportTaskResourceExpense (ReportTaskResourceExpenseRequest) returns (api.protobuf.SimpleResponse) {
                option (google.api.http) = {
                    post: "/carrier/v1/yarn/reportTaskResourceExpense"
                    body: "*"
                };
            }
        }
        message ReportTaskResourceExpenseRequest {
            NodeType                    node_type = 1;
            string                      node_id = 2;
            types.ResourceUsageOverview usage = 3;
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
            req = pb2.ReportTaskResourceExpenseRequest()
            req.node_type = report_node_type
            req.node_id = node_id
            req.usage.total_mem = resource_usage["total_mem"]
            req.usage.used_mem = resource_usage["used_mem"]
            req.usage.total_processor = resource_usage["total_processor"]
            req.usage.used_processor = resource_usage["used_processor"]
            req.usage.total_bandwidth = resource_usage["total_bandwidth"]
            req.usage.used_bandwidth = resource_usage["used_bandwidth"]
            req.usage.total_disk = resource_usage["total_disk"]
            req.usage.used_disk = resource_usage["used_disk"]
            self.__client.ReportTaskResourceExpenseRequest(req)
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

def get_resource_usage():
    resource_usage = {}
    resource_usage["total_mem"] = psutil.virtual_memory().total
    resource_usage["used_mem"] = psutil.virtual_memory().used
    resource_usage["total_processor"] = psutil.cpu_count()
    load1, load5, load15 = psutil.getloadavg()
    resource_usage["used_processor"] = math.ceil(load15 * psutil.cpu_count())
    resource_usage["total_bandwidth"] = 0
    net_1 = psutil.net_io_counters()
    time.sleep(1)
    net_2 = psutil.net_io_counters()
    resource_usage["used_bandwidth"] = (net_2.bytes_sent - net_1.bytes_sent) + (net_2.bytes_recv - net_1.bytes_recv)
    resource_usage["total_disk"] = psutil.disk_usage('/').total
    resource_usage["used_disk"] = psutil.disk_usage('/').used
    return resource_usage

def report_task_resource_expense(server_addr:str, node_type, node_id, interval=10):
    report_engine = ReportEngine(server_addr)
    try:
        while True:
            resource_usage = get_resource_usage()
            report_engine.report_task_resource_expense(node_type, node_id, resource_usage)
            time.sleep(interval)
    except Exception as e:
        log.exception(f"report resource usage error. {str(e)}")
    finally:
        report_engine.close()
        log.info('report_engine closed')
