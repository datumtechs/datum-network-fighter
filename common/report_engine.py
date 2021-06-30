# coding:utf-8

import grpc
# from protos import schedule_svc_pb2 as pb2
# from protos import schedule_svc_pb2_grpc as pb2_grpc
from common.consts import EVENT_QUEUE

class ReportEngine(object):

    def __init__(self, server_ip:str, server_port:str):
        conn = grpc.insecure_channel(f'{server_ip}:{server_port}')  # 连接调度节点的ip和port
        self.__client = pb2_grpc.ScheduleProviderStub(channel=conn)
        self.__queue = EVENT_QUEUE

    def report_event(self):
        try:
            event = self.get_event()

            req = pb2.ComputeNodeReportReq()
            req.type = event.type_
            req.identity = event.dict_["identity"]
            req.task_id = event.dict_["task_id"]
            req.content = event.dict_["content"]
            req.create_time = event.dict_["create_time"]
            self.__client.HandleComputeNodeReportStatus(req)
        except Exception as e:
            print(e)

    def get_event(self):
        event = self.__queue.get()
        return event

if __name__=='__main__':
    from common.event_engine import Event, EventEngine

    # report_engine = ReportEngine(server_ip='127.0.0.1', server_port='50031')
    type_ = "0301001"
    identity = "did:pid:0xeeeeff...efaab"
    task_id = "sfasasfasdfasdfa"
    content = "compute task start."
    event_engine = EventEngine()
    event_engine.fire_event(type_, identity, task_id, content)

    event = EVENT_QUEUE.get()
    print(event.type_, event.dict_)
    # report_engine.report_event()