from queue import Queue

EVENT_QUEUE = Queue()

DATA_SVC_EVENT_TYPE = {
  "0200000": "EVENT_DATA_TASK_START",         # 数据任务开始
  "0200001": "EVENT_DATA_TASK_SUCCESS",       # 数据任务成功
  "0200002": "EVENT_DATA_TASK_FAILED",        # 数据任务失败
  "0300003": "EVENT_DATA_TASK_CALCEL",        # 数据任务已被撤销
  "0300004": "EVENT_DATA_TASK_REGISTER_VIA",  # 数据服务已注册到via
}

COMPUTE_SVC_EVENT_TYPE = {
  "0300000": "EVENT_COMPUTE_TASK_START",         # 计算任务开始
  "0300001": "EVENT_COMPUTE_TASK_LOAD_ALGO",     # 计算任务加载算法完成
  "0300002": "EVENT_COMPUTE_TASK_BUILD_ENVRION", # 计算任务构建环境完成
  "0300003": "EVENT_COMPUTE_TASK_SET_IO",        # 计算任务设置IO完成
  "0300004": "EVENT_COMPUTE_TASK_SUCCESS",       # 计算任务成功
  "0300005": "EVENT_COMPUTE_TASK_FAILED",        # 计算任务失败
  "0300006": "EVENT_COMPUTE_TASK_CALCEL",        # 计算任务已被撤销
  "0300007": "EVENT_COMPUTE_TASK_REGISTER_VIA",  # 计算任务已注册到via
}
