from multiprocessing import Queue

EVENT_QUEUE = Queue()

DATA_EVENT = {
    "UPLOAD_DATA_SUCCESS": "0207000",  # 源数据上传成功
    "UPLOAD_DATA_FAILED": "0207001",  # 源数据上传失败
    "DOWNLOAD_DATA_SUCCESS": "0207002",  # 源数据下载成功
    "DOWNLOAD_DATA_FAILED": "0207003",  # 源数据下载失败
    "DELETE_DATA_SUCCESS": "0207004",  # 源数据删除成功
    "DELETE_DATA_FAILED": "0207005",  # 源数据删除失败

    "TASK_START": "0209000",  # 开始新任务
    "DOWNLOAD_CONTRACT_SUCCESS": "0209001",  # 下载合约代码成功
    "DOWNLOAD_CONTRACT_FAILED": "0209002",  # 下载合约代码失败
    "BUILD_ENVIRONMENT_SUCCESS": "0209003",  # 构建计算任务环境成功
    "BUILD_ENVIRONMENT_FAILED": "0209004",  # 构建计算任务环境失败
    "CREATE_CHANNEL_SUCCESS": "0209005",  # 创建网络通道成功
    "CREATE_CHANNEL_FAILED": "0209006",  # 创建网络通道失败
    "REGISTER_TO_VIA_SUCCESS": "0209007",  # 注册到via服务成功
    "REGISTER_TO_VIA_FAILED": "0209008",  # 注册到via服务失败
    "SET_CHANNEL_SUCCESS": "0209009",  # 设置网络通道成功
    "SET_CHANNEL_FAILED": "0209010",  # 设置网络通道失败
    "CONTRACT_EXECUTE_START": "0209011",  # 开始执行合约
    "CONTRACT_EXECUTE_SUCCESS": "0209012",  # 合约执行成功
    "CONTRACT_EXECUTE_FAILED": "0209013",  # 合约执行失败
}

COMPUTE_EVENT = {
    "TASK_START": "0309000",  # 开始新任务
    "DOWNLOAD_CONTRACT_SUCCESS": "0309001",  # 下载合约代码成功
    "DOWNLOAD_CONTRACT_FAILED": "0309002",  # 下载合约代码失败
    "BUILD_ENVIRONMENT_SUCCESS": "0309003",  # 构建计算任务环境成功
    "BUILD_ENVIRONMENT_FAILED": "0309004",  # 构建计算任务环境失败
    "CREATE_CHANNEL_SUCCESS": "0309005",  # 创建网络通道成功
    "CREATE_CHANNEL_FAILED": "0309006",  # 创建网络通道失败
    "REGISTER_TO_VIA_SUCCESS": "0309007",  # 注册到via服务成功
    "REGISTER_TO_VIA_FAILED": "0309008",  # 注册到via服务失败
    "SET_CHANNEL_SUCCESS": "0309009",  # 设置网络通道成功
    "SET_CHANNEL_FAILED": "0309010",  # 设置网络通道失败
    "CONTRACT_EXECUTE_START": "0309011",  # 开始执行合约
    "CONTRACT_EXECUTE_SUCCESS": "0309012",  # 合约执行成功
    "CONTRACT_EXECUTE_FAILED": "0309013",  # 合约执行失败
}

COMMON_EVENT = {
    "END_FLAG_SUCCESS": "0008000",  # 成功终止符，成功终止符，数据节点或计算节点只有最终运行成功，才会生成该事件
    "END_FLAG_FAILED": "0008001",  # 失败终止符，数据节点或计算节点任何运行失败，都会生成该事件
}

MAX_MESSAGE_LENGTH = 2 * 1024 ** 3 - 1
GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
]
