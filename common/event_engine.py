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

    def fire_event(self, event_type, task_id, identity_id, content):
        '''
        生成事件，并推入队列
        args:
            event_type: 事件类型码
            task_id: 事件对应的任务id
            identity_id: 产生事件的节点身份标识，格式是did:pid:0xeeeeff...efaab
            content: 事件内容
            create_at: 事件产生时间
        '''

        event = Event(event_type)
        create_at = int(time.time())
        info = dict(task_id=task_id, identity_id=identity_id, content=content, create_at=create_at)
        event.dict_.update(info)
        self.put_event(event)

    def put_event(self, event: Event) -> None:
        """
        向事件队列中存入事件
        """
        self.__queue.put(event)

event_engine = EventEngine()

if __name__ == '__main__':
    type_ = "0301001"
    task_id = "sfasasfasdfasdfa"
    identity_id = "did:pid:0xeeeeff...efaab"
    content = "compute task start."
    event_engine.fire_event(type_, task_id, identity_id, content)
