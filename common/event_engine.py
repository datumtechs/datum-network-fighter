# coding:utf-8

import time
from common.consts import EVENT_QUEUE


class Event(object):
    """
    Event Object
    """

    def __init__(self, event_type=None):
        self.type_: str = event_type  # Event type, different types can have different event content
        self.dict_ = {}  # Contents of the event


class EventEngine(object):
    def __init__(self):
        self.__queue = EVENT_QUEUE

    def create_event(self, event_type:str, content:str, task_id:str, party_id:str, identity_id=""):
        event = Event(event_type)
        create_at = int(time.time() * 1000)  # in ms
        info = dict(task_id=task_id, party_id=party_id, content=content,
                identity_id=identity_id, create_at=create_at)
        event.dict_.update(info)
        return event

    def fire_event(self, event_type:str, content:str, task_id:str, party_id:str, identity_id=""):
        '''
        Generate events and push them to the queue
        args:
            event_type: Event type code
            party_id: The party id corresponding to current node
            task_id: The task id corresponding to the event
            content: The content of the event
            create_at: Time of event
            identity_id: The identity of the node that generated the event, 
                         the format is 'did:pid:0xeeeeff...efaab'
        '''
        event = self.create_event(event_type, content, task_id, party_id, identity_id)
        self.put_event(event)

    def put_event(self, event: Event) -> None:
        """
        Push events into the queue
        """
        self.__queue.put(event)


event_engine = EventEngine()

if __name__ == '__main__':
    type_ = "0301001"
    party_id = "p0"
    task_id = "sfasasfasdfasdfa"
    content = "compute task start."
    identity_id = "did:pid:0xeeeeff...efaab"
    event_engine.fire_event(type_, content, task_id, party_id, identity_id)
