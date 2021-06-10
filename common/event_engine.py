# coding:utf-8

import time
from common.consts import EVENT_QUEUE

class Event(object):
    """事件对象"""

    def __init__(self, event_type=None):
        self.type_: str = event_type  # 事件类型
        self.dict_ = {}               # 事件的内容的其他字段

class EventEngine(object):
    def __init__(self):
        self.__queue = EVENT_QUEUE

    def fire_event(self, event_type, identity, task_id, content, time_format='%Y-%m-%d %H:%M:%S'):
        '''
        生成事件，并推入队列
        args:
            event_type: 事件类型
            identity: 节点标识，格式是did:pid:0xeeeeff...efaab
            task_id: 任务id
            content: 事件内容
            create_time: 事件生成时间
        '''

        event = Event(event_type)
        create_time = time.strftime(time_format)
        info = dict(identity=identity, task_id=task_id, content=content, create_time=create_time)
        event.dict_.update(info)
        self.put_event(event)

    def put_event(self, event: Event) -> None:
        """
        向事件队列中存入事件
        """
        self.__queue.put(event)


