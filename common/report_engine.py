# coding:utf-8

import queue

import grpc

from common.consts import EVENT_QUEUE
from protos.lib.api import sys_rpc_api_pb2 as pb2
from protos.lib.api import sys_rpc_api_pb2_grpc as pb2_grpc


class ReportEngine(object):

    def __init__(self, server_addr: str):
        self.conn = grpc.insecure_channel(server_addr)  # 连接调度节点的ip和port
        self.__client = pb2_grpc.YarnServiceStub(channel=self.conn)

    def report_event(self):
        try:
            # get event
            event = EVENT_QUEUE.get(block=True, timeout=1)
            req = pb2.ReportTaskEventRequest()
            req_task_event = req.task_event
            req_task_event.type = event.type_
            req_task_event.task_id = event.dict_["task_id"]
            req_task_event.identity_id = event.dict_["identity_id"]
            req_task_event.content = event.dict_["content"]
            req_task_event.create_at = event.dict_["create_at"]
            self.__client.ReportTaskEvent(req)
        except queue.Empty as e:
            pass
        except Exception as e:
            print(e)

    def report_file_summary(self, summary):
        try:
            req = pb2.ReportUpFileSummaryRequest()
            req.origin_id = summary["origin_id"]
            req.file_path = summary["file_path"]
            req.ip = summary["ip"]
            req.port = str(summary["port"])
            return self.__client.ReportUpFileSummary(req)
        except Exception as e:
            print(e)

    def close(self):
        self.conn.close()


def report_event(server_addr: str, stop_event):
    report_engine = ReportEngine(server_addr)
    try:
        while True:
            report_engine.report_event()
            if stop_event.is_set():
                break
    except Exception as e:
        print(f"report event error. {str(e)}")
    finally:
        report_engine.close()
        print('report_engine closed')


def report_file_summary(server_addr: str, summary: dict):
    report_engine = ReportEngine(server_addr)
    ret = report_engine.report_file_summary(summary)
    report_engine.close()
    return ret
