# coding:utf-8

import unittest

from common.consts import EVENT_QUEUE
from common.event_engine import EventEngine

class TestEventEngine(unittest.TestCase):
    def test_init(self):
        event_engine = EventEngine()
        self.assertTrue(isinstance(event_engine, EventEngine))

    def test_fire_event(self):
        event_engine = EventEngine()
        type_ = "0301001"
        identity = "did:pid:0xeeeeff...efaab"
        task_id = "sfasasfasdfasdfa"
        content = "compute task start."
        event_engine.fire_event(type_, identity, task_id, content)

        event = EVENT_QUEUE.get()
        self.assertEqual(event.type_, type_)
        self.assertEqual(event.dict_.get("identity"), identity)
        self.assertEqual(event.dict_.get("task_id"), task_id)
        self.assertEqual(event.dict_.get("content"), content)

if __name__ == '__main__':
    unittest.main()