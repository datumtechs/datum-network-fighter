# 算法相关参数说明

## 1.训练和预测
训练与预测的相关算法有logistic regression、linear regression、DNN、XGBoost等，下面对每个算法说明：

### 1.1 logistic regression

**1.1.1 LR训练**

算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。\n
**本方的配置参数(self_cfg_params)：**
  暂时仅包含input_data字段，如果后面有新的需求变化，可以增加新的字段。input_data里面的字段含义如下：
  ①input_type：输入数据的类型. (算法用，标识数据使用方式).  0:unknown, 1:origin_data, 2:psi_output, 3:model等等。可以根据数据类型的增加而增加。暂时只有三种类型：源数据，psi输出结果，模型结果
  ②access_type: 访问数据的方式, (fighter用，决定是否预先加载数据). 0:unknown, 1:local, 2:http, 3:https, 4:ftp等等。现阶段仅支持local
  ③data_type：数据的格式, (算法用，标识数据格式). 0:unknown, 1:csv, 2:folder, 3:xls, 4:txt, 5:json, 6:mysql, 7:bin等等。现阶段仅支持csv和folder。
  ④data_path：如果数据在本地(access_type=local)，则这里是数据路径。如果数据在远程(access_type=http/https/ftp)，则这里是超链接
  ⑤key_column：id列，作为样本的唯一标识。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
  ⑥selected_columns：选择的列，指的是自变量(特征列)。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
**算法动态参数(algorithm_dynamic_params)：**
  所含字段随着算法变化而变化，需根据算法来定制

训练存在如下角色：源数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

- data1方(源数据输入方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式. 0:unknown, 1:csv, 2:folder, 3:xls, 4:txt, 5:json, 6:mysql, 7:bin
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 逻辑回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 验证集占输入数据集的比例，值域(0,1)
          "predict_threshold": 0.5     # 验证集预测结果的分类阈值，值域[0,1]
      }
  }
}
```

- compute1方(计算方)的配置
```
{
  "self_cfg_params": {
      "party_id": "compute1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 逻辑回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 验证集占输入数据集的比例，值域(0,1)
          "predict_threshold": 0.5     # 验证集预测结果的分类阈值，值域[0,1]
      }
  }
}
```

- result1方(结果接收方)的配置
```
{
  "self_cfg_params": {
      "party_id": "result1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 逻辑回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 验证集占输入数据集的比例，值域(0,1)
          "predict_threshold": 0.5     # 验证集预测结果的分类阈值，值域[0,1]
      }
  }
}
```


**1.1.2 LR预测**

算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。
预测存在如下角色：源数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：

- data1方(源数据输入方)的配置
源数据输入方的输入数据可以同时有源数据和psi结果，这是因为psi结果只服务于该方的源数据，而不服务其他方，所以可以将psi结果与源数据看成一个整体，作为一个数据提供方。
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```

- model1方(模型提供方)的配置
模型作为输入，需作为单独的一方，这是因为模型是最终目的是作为3个计算方的输入，并不是为了给源数据使用，不能与源数据打包成个整体，他是一个单独的数据提供方。
```
{
  "self_cfg_params": {
      "party_id": "model1",    # 本方party_id
      "input_data": [
        {
            "input_type": 3,    # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 2      # 数据的格式.
            "data_path": "path/to/data"   # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```

- compute1方(计算方)的配置
```
{
  "self_cfg_params": {
      "party_id": "compute1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```

- result1方(结果接收方)的配置
```
{
  "self_cfg_params": {
      "party_id": "result1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```


### 1.2 linear regression

**1.2.1 LinR训练**

算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。
self_cfg_params里的参数的含义请详见逻辑回归训练的部分。

训练存在如下角色：源数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：
- data1方(源数据输入方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_path": "path/to/data",  # 数据所在的本地路径
            "data_format": 1,      # 数据的格式.
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,
            "access_method": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "data_format": 1,
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
   "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 线性回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 验证集占输入数据集的比例，值域(0,1)
      }
  }
}
```

- compute1方(计算方)的配置
```
{
  "self_cfg_params": {
      "party_id": "compute1",    # 本方party_id
      "input_data": []
  },
   "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 线性回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 验证集占输入数据集的比例，值域(0,1)
      }
  }
}
```

- result1方(计算方)的配置
```
{
  "self_cfg_params": {
      "party_id": "result1",    # 本方party_id
      "input_data": []
  },
   "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",    # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 线性回归的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的数
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 验证集占输入数据集的比例，值域(0,1)
      }
  }
}
```


**1.2.2 LinR预测**

算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。
预测存在如下角色：源数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：

- data1方(源数据输入方)的配置
源数据输入方的输入数据可以同时有源数据和psi结果，这是因为psi结果只服务于该方的源数据，而不服务其他方，所以可以将psi结果与源数据看成一个整体，作为一个数据提供方。
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```

- model1方(模型提供方)的配置
模型作为输入，需作为单独的一方，这是因为模型是最终目的是作为3个计算方的输入，并不是为了给源数据使用，不能与源数据打包成个整体，他是一个单独的数据提供方。
```
{
  "self_cfg_params": {
      "party_id": "model1",    # 本方party_id
      "input_data": [
        {
            "input_type": 3,    # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 2      # 数据的格式.
            "data_path": "path/to/data"   # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1"  # 模型所在方
  }
}
```

- compute1方(计算方)的配置
```
{
  "self_cfg_params": {
      "party_id": "compute1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1"  # 模型所在方
  }
}
```

- result1方(结果接收方)的配置
```
{
  "self_cfg_params": {
      "party_id": "result1",    # 本方party_id
      "input_data": []
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1"  # 模型所在方
}
```


### 1.3 DNN

**1.3.1 DNN训练**

cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归训练相同，algorithm_dynamic_params是由本算法定制。

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,        # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # DNN的超参数
          "epochs": 10,            # 训练轮次
          "batch_size": 256,       # 批量大小
          "learning_rate": 0.1,    # 学习率
          "layer_units": [32, 128, 32, 1],  # 隐藏层与输出层的每层单元数，例子中有3个隐藏层，
                                            # 每层的单元数分别是32，128，32。输出层单元数是1。 大于0的整数
          "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],   # 隐藏层与输出层的每层的激活函数
                                                                        # 仅支持"sigmoid"/"relu"/""/null
          "init_method": "random_normal",  # 指定模型参数初始化方法, 
                                          # 仅支持random_normal/random_uniform/zeros/ones
          "use_intercept": true,       # 指定模型结构中是否使用bias, true-用，false-不用
          "optimizer": "sgd",          # 优化器，暂时仅支持sgd
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 验证集占输入数据集的比例，值域(0,1)
          "predict_threshold": 0.5     # 验证集预测结果的分类阈值，值域[0,1]
      }
  }
}
```



**1.3.2 DNN预测**

cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归预测相同，algorithm_dynamic_params是由本算法定制。

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # DNN的超参数
          "layer_units": [32, 128, 32, 1],   # 隐藏层与输出层的每层单元数，此参数配置必须与训练时的一样
          "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],  # 隐藏层与输出层的每层的激活函数
                                                                            # 此参数配置必须与训练时的一样
          "use_intercept": true,    # 指定模型结构中是否使用bias, 此参数配置必须与训练时的一样
          "predict_threshold": 0.5  # 二分类的阈值，值域[0,1]
      }
  }
}
```

- model1方，模型提供方
```
{
  "self_cfg_params": {
      "party_id": "model1",    # 本方party_id
      "input_data": [
        {
            "input_type": 3,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1       # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # DNN的超参数
          "layer_units": [32, 128, 32, 1],   # 隐藏层与输出层的每层单元数，此参数配置必须与训练时的一样
          "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],  # 隐藏层与输出层的每层的激活函数
                                                                            # 此参数配置必须与训练时的一样
          "use_intercept": true,    # 指定模型结构中是否使用bias, 此参数配置必须与训练时的一样
          "predict_threshold": 0.5  # 二分类的阈值，值域[0,1]
      }
  }
}
```



### 1.4 XGBoost

**1.4.1 XGBoost训练**

cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归训练相同，algorithm_dynamic_params是由本算法定制。

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,        # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,           # 是否使用psi
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # XGBoost的超参数
          "epochs": 10,            # 训练轮次
          "batch_size": 256,       # 批量大小
          "learning_rate": 0.01,   # 学习率
          "num_trees": 3,          # 多少棵树，大于0的整数
          "max_depth": 4,          # 树的深度，大于0的整数
          "num_bins": 5,           # 特征的分箱数，大于0的整数
          "num_class": 2,          # 标签的类别数，大于1的整数
          "lambd": 1.0,            # L2正则项系数, [0, +∞)
          "gamma": 0.0,            # 复杂度控制因子，用于防止过拟合。
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 验证集占输入数据集的比例，值域(0,1)
          "predict_threshold": 0.5     # 验证集预测结果的分类阈值，值域[0,1]
      }
  }
}
```


**1.4.2 XGBoost预测**

cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归预测相同，algorithm_dynamic_params是由本算法定制。
- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        },
        {
            "input_type": 2,
            "data_type": 1,
            "data_path": "path/to/data1/psi_result.csv",
            "key_column": "",
            "selected_columns": []
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # XGBoost的超参数
          "num_trees": 3,   # 多少棵树，大于0的整数，此参数配置必须与训练时的一样
          "max_depth": 4,   # 树的深度，大于0的整数，此参数配置必须与训练时的一样
          "num_bins": 5,    # 特征的分箱数，大于0的整数，此参数配置必须与训练时的一样
          "num_class": 2,   # 标签的类别数，大于1的整数，此参数配置必须与训练时的一样
          "lambd": 1.0,     # L2正则项系数, [0, +∞)，此参数配置必须与训练时的一样
          "gamma": 0.0,     # 复杂度控制因子，用于防止过拟合。此参数配置必须与训练时的一样
          "predict_threshold": 0.5  # 二分类时的阈值，值域[0,1]
      }
  }
}
```

- model1方，模型提供方
```
{
  "self_cfg_params": {
      "party_id": "model1",    # 本方party_id
      "input_data": [
        {
            "input_type": 3,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1       # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_psi": true,    # 是否使用psi
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # XGBoost的超参数
          "num_trees": 3,   # 多少棵树，大于0的整数，此参数配置必须与训练时的一样
          "max_depth": 4,   # 树的深度，大于0的整数，此参数配置必须与训练时的一样
          "num_bins": 5,    # 特征的分箱数，大于0的整数，此参数配置必须与训练时的一样
          "num_class": 2,   # 标签的类别数，大于1的整数，此参数配置必须与训练时的一样
          "lambd": 1.0,     # L2正则项系数, [0, +∞)，此参数配置必须与训练时的一样
          "gamma": 0.0,     # 复杂度控制因子，用于防止过拟合。此参数配置必须与训练时的一样
          "predict_threshold": 0.5  # 二分类时的阈值，值域[0,1]
      }
  }
}
```



## 2. Private Set Intersection

cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归训练相同，algorithm_dynamic_params是由本算法定制。

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型. 0: unknown, 1: origin_data, 2: psi_output 3: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": []  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
      "psi_type": "T_V1_Basic_GLS254"  # 支持T_V1_Basic_GLS254, T_V1_Basic_SECP等
  }
}
```


## 3. 模型评估

现阶段模型评估有2种类型，一种是二分类模型评估，一种是回归类型模型评估。

+ **二分类模型评估**
现在支持二分类模型评估的训练算法有logistic regression, DNN, XGBoost.
生成的评估指标的内容如下：
```
{
    "AUC": 0.95,
    "accuracy": 0.91,
    "f1_score": 0.85,
    "precision": 0.93,
    "recall": 0.81
}
```

+ **回归类型模型评估**
现在支持回归类型模型评估的训练算法有linear regression, DNN.
生成的评估指标的内容如下：
```
{
    "R2-score": 0.968016,
    "RMSE": 0.120324,
    "MSE": 0.014478,
    "MAE": 0.087593
}
```
