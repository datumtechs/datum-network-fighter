import logging
from lib.api import sys_rpc_api_pb2
from lib.api import sys_rpc_api_pb2_grpc
from lib.common import base_pb2
from lib.common import base_pb2_grpc


log = logging.getLogger(__name__)

class YarnService(sys_rpc_api_pb2_grpc.YarnServiceServicer):
    '''
    Only use for test, test the report interface.
    '''
    
    def ReportTaskEvent(self, request, context):
        log.info(f'get event start.')
        event = {}
        event["task_id"] = request.task_event.task_id
        event["identity_id"] = request.task_event.identity_id
        event["party_id"] = request.task_event.party_id
        event["content"] = request.task_event.content
        event["create_at"] = request.task_event.create_at
        log.info(f'get event finish: {event}')
        return base_pb2.SimpleResponse(status=0, msg="report event ok.")
        

    def ReportTaskResourceUsage(self, request, context):
        log.info(f'get task resourece usage start.')
        resource_usage = {}
        resource_usage["task_id"] = request.task_id
        resource_usage["party_id"] = request.party_id
        resource_usage["node_type"] = request.node_type
        resource_usage["ip"] = request.ip
        resource_usage["port"] = request.port
        resource_usage["total_mem"] = request.usage.total_mem
        resource_usage["used_mem"] = request.usage.used_mem
        resource_usage["total_processor"] = request.usage.total_processor
        resource_usage["used_processor"] = request.usage.used_processor
        resource_usage["total_bandwidth"] = request.usage.total_bandwidth
        resource_usage["used_bandwidth"] = request.usage.used_bandwidth
        resource_usage["total_disk"] = request.usage.total_disk
        resource_usage["used_disk"] = request.usage.used_disk
        log.info(f'get task resourece usage finish: {resource_usage}')
        return base_pb2.SimpleResponse(status=0, msg="report resource ok.")

    def ReportUpFileSummary(self, request, context):
        log.info(f'get upload file summary start.')
        summary = {}
        summary["origin_id"] = request.origin_id
        summary["file_path"] = request.file_path
        summary["ip"] = request.ip
        summary["port"] = request.port
        log.info(f'get up file summary finish: {summary}')
        return base_pb2.SimpleResponse(status=0, msg="report file summary ok.")

    def ReportTaskResultFileSummary(self, request, context):
        log.info(f'get task result file summary start.')
        summary = {}
        summary["task_id"] = request.task_id
        summary["origin_id"] = request.origin_id
        summary["file_path"] = request.file_path
        summary["ip"] = request.ip
        summary["port"] = request.port
        log.info(f'get task result file summary finish: {summary}')
        return base_pb2.SimpleResponse(status=0, msg="report task result file summary ok.")

