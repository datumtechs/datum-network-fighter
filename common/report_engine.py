# coding:utf-8

import grpc
from protos import sys_rpc_api_pb2 as pb2
from protos import sys_rpc_api_pb2_grpc as pb2_grpc
from common.consts import EVENT_QUEUE

class ReportEngine(object):

    def __init__(self, server_addr: str):
        conn = grpc.insecure_channel(server_addr)  # 连接调度节点的ip和port
        self.__client = pb2_grpc.ScheduleProviderStub(channel=conn)
        self.__queue = EVENT_QUEUE

    def report_event(self):
        try:
            event = self.get_event()

            req = pb2.ReportTaskEventRequest()
            req_task_event = req.task_event
            req_task_event.type = event.type_
            req_task_event.task_id = event.dict_["task_id"]
            req_task_event.identity_id = event.dict_["identity_id"]
            req_task_event.content = event.dict_["content"]
            req_task_event.create_at = event.dict_["create_at"]
            self.__client.ReportTaskEvent(req)
        except Exception as e:
            print(e)

    def get_event(self):
        event = self.__queue.get()
        return event
