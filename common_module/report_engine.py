# coding:utf-8

import os
import sys
import threading
import time
import math
import queue
import logging
import psutil
import grpc
from common_module.consts import EVENT_QUEUE, COMMON_EVENT
from common_module.utils import process_recv_address
from pb.carrier.api import sys_rpc_api_pb2 as pb2
from pb.carrier.api import sys_rpc_api_pb2_grpc as pb2_grpc

log = logging.getLogger(__name__)
CLIENT_OPTIONS = [('grpc.enable_retries', 0),
                  ('grpc.keepalive_time_ms', 60000),
                  ('grpc.keepalive_timeout_ms', 20000)
                  ]

def request_to_str(req):
    str_req = '{' + str(req).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
    return str_req

class ReportEngine(object):

    def __init__(self, server_addr: str):
        self.conn = grpc.insecure_channel(server_addr)
        self.__client = pb2_grpc.YarnServiceStub(channel=self.conn)

    def report_task_event(self, event):
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
        req = pb2.ReportTaskEventRequest()
        req.task_event.type = event.type_
        req.task_event.task_id = event.dict_["task_id"]
        req.task_event.identity_id = event.dict_["identity_id"]
        req.task_event.party_id = event.dict_["party_id"]
        req.task_event.content = "{}:{}".format(event.dict_["party_id"], event.dict_["content"])
        req.task_event.create_at = event.dict_["create_at"]
        response = self.__client.ReportTaskEvent(req)
        log.info(request_to_str(req))
        return response

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
            string ip = 2;
            string port = 3;
            string data_hash = 4;
            types.OrigindataType data_type = 5;
            string metadata_option = 6;
        }
        """
        req = pb2.ReportUpFileSummaryRequest()
        req.origin_id = summary["origin_id"]
        req.ip = summary["ip"]
        req.port = str(summary["port"])
        req.data_hash = summary["data_hash"]
        req.data_type = summary["data_type"]
        req.metadata_option = summary["metadata_option"]
        response = self.__client.ReportUpFileSummary(req)
        log.info(request_to_str(req))
        return response

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
            string ip = 3;
            string port = 4;
            string extra = 5;
            string data_hash = 6;
            types.OrigindataType data_type = 7;
            string metadata_option = 8;
        }
        """

        req = pb2.ReportTaskResultFileSummaryRequest()
        req.task_id = summary["task_id"]
        req.origin_id = summary["origin_id"]
        req.ip = summary["ip"]
        req.port = str(summary["port"])
        req.extra = str(summary["extra"])
        req.data_hash = summary["data_hash"]
        req.data_type = summary["data_type"]
        req.metadata_option = summary["metadata_option"]
        response = self.__client.ReportTaskResultFileSummary(req, timeout=10)
        log.info(request_to_str(req))
        return response

    def report_task_resource_usage(self, task_id, party_id, node_type, ip: str, port: str, resource_usage):
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
        response = self.__client.ReportTaskResourceUsage(req)
        log.debug(request_to_str(req))
        return response

    def close(self):
        self.conn.close()


def report_task_event(server_addr, create_event, event_type, content):
    log.info('start report task event.')
    report_success = False
    try_max_times = 20
    try_cnt = 0
    while not report_success:
        try:
            report_engine = ReportEngine(server_addr)
            event = create_event(event_type, content)
            report_engine.report_task_event(event)
            report_engine.close()
            report_success = True
        except grpc._channel._InactiveRpcError as e:
            try_cnt = try_cnt + 1
            if (try_cnt >= try_max_times):
                raise
            time.sleep(0.5)
        except:
            raise

def report_task_result(server_addr: str, report_type: str, content: dict, *args):
    report_success = False
    try_max_times = 20
    try_cnt = 0
    while not report_success:
        try:
            report_engine = ReportEngine(server_addr)
            if report_type == 'upload_file':
                ret = report_engine.report_upload_file_summary(content)
            elif report_type == 'result_file':
                ret = report_engine.report_task_result_file_summary(content)
            elif report_type == 'last_event':
                ret = report_engine.report_task_event(content)
            elif report_type == 'resource_usage':
                ret = report_engine.report_task_resource_usage(*args, content)
            else:
                raise ValueError(f'no report_type {report_type}')
            report_engine.close()
            report_success = True
            return ret
        except grpc._channel._InactiveRpcError as e:
            try_cnt = try_cnt + 1
            if (try_cnt >= try_max_times):
                raise
            time.sleep(0.5)
        except:
            raise


def resource_info(avg_used_memory, avg_used_cpu, total_bandwidth, avg_used_bandwidth):
    return {"total_mem": psutil.virtual_memory().total,
            "used_mem": math.ceil(avg_used_memory),
            "total_processor": psutil.cpu_count(),
            "used_processor": math.ceil(avg_used_cpu),
            "total_bandwidth": total_bandwidth,
            "used_bandwidth": math.ceil(avg_used_bandwidth),
            "total_disk": psutil.disk_usage('/').total,
            "used_disk": psutil.disk_usage('/').used}


def monitor_resource_usage(task_pid, limit_time, limit_memory, limit_cpu, limit_bandwidth, server_addr, create_event,
                           event_type, task_id, party_id, node_type, ip, port, total_bandwidth, interval=5):
    log.info(f'monitor_resource_usage start. params is task_pid:{task_pid},limit_time:{limit_time},limit_memory:{limit_memory},limit_cpu:{limit_cpu},limit_bandwidth:{limit_bandwidth},interval:{interval}')
    start_time = time.time()
    p = psutil.Process(task_pid)
    total_cpu_num = psutil.cpu_count()
    memory_list, cpu_list, bandwidth_list = [], [], []

    # prevent warning
    count = 0
    avg_used_memory = 0
    avg_used_cpu = 0
    avg_used_bandwidth = 0

    first = True
    while True:
        try:
            # time limit
            use_time = time.time() - start_time
            assert use_time <= limit_time, f"task used time:{round(use_time, 2)}s, exceeds the limit({limit_time}s)."

            # memory limit
            used_memory = p.memory_info().rss
            # memory_list.insert(0, used_memory)
            memory_list.append(used_memory)

            avg_used_memory = sum(memory_list) / len(memory_list)
            # if limit_memory and (avg_used_memory > limit_memory):
            #     log.error(f"memory_list: {memory_list}")
            #     raise Exception(f"memory used({round(avg_used_memory, 2)}B) exceeds the limit({limit_memory}B).")

            # cpu limit
            used_processor = round(total_cpu_num * p.cpu_percent() / 100)
            # cpu_list.insert(0, used_processor)
            cpu_list.append(used_processor)
            avg_used_cpu = sum(cpu_list) / len(cpu_list)
            # if limit_cpu and (avg_used_cpu > limit_cpu):
            #     log.error(f"cpu_list: {cpu_list}")
            #     raise Exception(f"cpu used({round(avg_used_cpu, 2)}) exceeds the limit({limit_cpu}).")

            # bandwidth statistics
            net_1 = psutil.net_io_counters()
            time.sleep(1)
            net_2 = psutil.net_io_counters()
            bandwidth_list.append(net_2.bytes_sent - net_1.bytes_sent)

            # bandwidth limit
            avg_used_bandwidth = sum(bandwidth_list) / len(bandwidth_list)
            # if limit_bandwidth and (avg_used_bandwidth > limit_bandwidth):
            #     log.error(f"bandwidth_list: {bandwidth_list}")
            #     raise Exception(
            #         f"bandwidth used({round(avg_used_bandwidth, 2)}) bps,exceeds the limit({limit_bandwidth}) bps.")
            count += 1
            if first:
                first = False
                avg_used_cpu = limit_cpu
                resource_usage = resource_info(avg_used_memory, avg_used_cpu, total_bandwidth, avg_used_bandwidth)
                log.info(f'first report resource usage info is: {resource_usage}')
                report_task_result(server_addr, 'resource_usage', resource_usage, task_id, party_id, node_type, ip,
                                   port)
            if count == interval:
                count = 0
                avg_used_cpu = limit_cpu
                resource_usage = resource_info(avg_used_memory, avg_used_cpu, total_bandwidth, avg_used_bandwidth)
                report_task_result(server_addr, 'resource_usage', resource_usage, task_id, party_id, node_type, ip,
                                   port)
        except Exception as e:
            log.exception(str(e))
            avg_used_cpu = limit_cpu
            resource_usage = resource_info(avg_used_memory, avg_used_cpu, total_bandwidth, avg_used_bandwidth)
            log.warning(f"resource_usage come from exception:{resource_usage}")
            report_task_result(server_addr, 'resource_usage', resource_usage, task_id, party_id, node_type, ip, port)

            event = create_event(event_type["RESOURCE_LIMIT_FAILED"], str(e)[:80])
            report_task_result(server_addr, 'last_event', event)
            event = create_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
            report_task_result(server_addr, 'last_event', event)
            # ensure the event report, then kill process
            p.kill()
            p.wait()
