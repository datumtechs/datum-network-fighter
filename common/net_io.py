import logging

from io_channel import IOChannel

print('enter net_io.py')
log = logging.getLogger(__name__)


def create_io(task, peers):
    log.info(f'create IOChannel for {task.id}')
    return IOChannel(task, peers)
