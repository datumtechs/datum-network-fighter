# 隐私算法(privacy algorithm)相关参数说明
算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。
+ 本方的配置参数(self_cfg_params)：

   暂时仅包含input_data字段，如果后面有新的需求变化，可以增加新的字段。input_data里面的字段含义如下：
  ```
  ①input_type：输入数据的类型. (算法用，标识数据使用方式).  0:unknown, 1:origin_data, 2:model等等。可以根据数据类型的增加而增加。暂时只有两种类型：源数据，模型结果
  ②access_type: 访问数据的方式, (fighter用，决定是否预先加载数据). 0:unknown, 1:local, 2:url等等。现阶段仅支持local
  ③data_type：数据的格式, (算法用，标识数据格式). 0:unknown, 1:csv, 2:dir, 3:binary, 4:xls, 5:xlsx, 6:txt, 7:json等等。现阶段仅支持csv和dir。
  ④data_path：如果数据在本地(access_type=local)，则这里是数据路径。如果数据在远程(access_type=url)，则这里是超链接
  ⑤key_column：id列，作为样本的唯一标识。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
  ⑥selected_columns：选择的列，指的是自变量(特征列)。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
  ```
+ 算法动态参数(algorithm_dynamic_params)：

  所含字段随着算法变化而变化，需根据算法来定制

## 1.训练和预测
训练与预测的相关算法有logistic regression、linear regression、DNN、XGBoost等，下面对每个算法说明：

### 1.1 logistic regression
#### 1.1.1 LR训练

训练存在如下角色：数据提供方、计算方、结果接收方。每种角色都可能存在多个组织。下面分别按参与方角色说明配置：

- data1方(源数据输入方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: model
            "data_type": 1,      # 数据的格式. 
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
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


#### 1.1.2 LR预测

预测存在如下角色：源数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：

- data1方(源数据输入方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
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
            "input_type": 2,    # 输入数据的类型.
            "data_type": 2      # 数据的格式.
            "data_path": "path/to/data"   # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
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
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```


### 1.2 linear regression

#### 1.2.1 LinR训练

训练存在如下角色：数据提供方、计算方、结果接收方。每种角色都可能存在多个组织。下面分别按参与方角色说明配置：
- data1方(源数据输入方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: model
            "data_type": 1,      # 数据的格式. 
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
   "algorithm_dynamic_params": {
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


#### 1.2.2 LinR预测

预测存在如下角色：数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型.
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
      "model_restore_party": "model1",  # 模型所在方
      "predict_threshold": 0.5      # 预测结果的分类阈值，值域[0,1]
  }
}
```

- model1方(模型提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "model1",    # 本方party_id
      "input_data": [
        {
            "input_type": 2,    # 输入数据的类型. 0: unknown, 1: origin_data, 2: model
            "data_type": 2      # 数据的格式.
            "data_path": "path/to/data"   # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
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
      "model_restore_party": "model1"  # 模型所在方
}
```


### 1.3 DNN

#### 1.3.1 DNN训练

训练存在如下角色：数据提供方、计算方、结果接收方。每种角色都可能存在多个组织。下面分别按参与方角色说明配置：

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型.
            "data_type": 1,        # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
      "label_owner": "data1",       # 标签所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # DNN的超参数
          "epochs": 5,            # 训练轮次
          "batch_size": 256,       # 批量大小
          "learning_rate": 0.1,    # 学习率
          "layer_units": [32, 1],  # 隐藏层与输出层的每层单元数，例子中有3个隐藏层，每层的单元数分别是32，128，32。输出层单元数是1。 大于0的整数
          "layer_activation": ["sigmoid", "sigmoid"],   # 隐藏层与输出层的每层的激活函数, 仅支持"sigmoid"/"relu"/""/null
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



#### 1.3.2 DNN预测

预测存在如下角色：数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：
- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型.
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # DNN的超参数
          "layer_units": [32, 1],   # 隐藏层与输出层的每层单元数，此参数配置必须与训练时的一样
          "layer_activation": ["sigmoid", "sigmoid"],  # 隐藏层与输出层的每层的激活函数
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
            "input_type": 2,     # 输入数据的类型.
            "data_type": 2       # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      "model_restore_party": "model1",   # 模型所在方
      "hyperparams": {           # DNN的超参数
          "layer_units": [32, 1],   # 隐藏层与输出层的每层单元数，此参数配置必须与训练时的一样
          "layer_activation": ["sigmoid", "sigmoid"],  # 隐藏层与输出层的每层的激活函数
                                                                            # 此参数配置必须与训练时的一样
          "use_intercept": true,    # 指定模型结构中是否使用bias, 此参数配置必须与训练时的一样
          "predict_threshold": 0.5  # 二分类的阈值，值域[0,1]
      }
  }
}
```



### 1.4 XGBoost

#### 1.4.1 XGBoost训练

存在如下角色：数据提供方、计算方、结果接收方。每种角色都可能存在多个组织。下面分别按参与方角色说明配置：
- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型.
            "data_type": 1,        # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
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


#### 1.4.2 XGBoost预测

预测存在如下角色：数据提供方、计算方、结果接收方、模型提供方。下面分别按参与方角色说明配置：
- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型.
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
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
            "input_type": 2,     # 输入数据的类型.
            "data_type": 2       # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
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

## 3. Private Set Intersection

该模块用于单独使用，后面不接训练或者预测算法。常见应用场景是隐私黑名单查询业务。
cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归训练相同，algorithm_dynamic_params是由本算法定制。
数据提供方有两方[data1, data2],  计算方有两方[compute1, compute2], 结果方有两方[result1, result2]。结果方可以只有一方，result1或者result2，那么data_flow_restrict需要相应地改动。
- 参与方的连接策略示意图
```
 data1             data2
   |                 |
   |                 |
   |                 |
   ↓                 ↓
compute1 -------> compute2
         <------- 
   |                 | 
   |                 |
   |                 |
   ↓                 ↓
 result1           result2
```

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型. 0: unknown, 1: origin_data, 2: model
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": []  # 自变量(特征列名), 该字段一直为空列表[]
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_alignment": false,
      "label_owner": "",   # 可无此字段
      "label_column": "",  # 可无此字段
      "psi_type": "T_V1_Basic_GLS254",  # 支持T_V1_Basic_GLS254, T_V1_Basic_SECP等
      "data_flow_restrict": {
        "data1": ["compute1"],
        "data2": ["compute2"],
        "compute1": ["result1"],
        "compute2": ["result2"]
      }
  }
}
```

## 4. Alignment

该模块专门用于训练和预测算法的对齐。与训练或者预测算法配套使用。与上一个算法不同的是use_alignment=true
cfg_dict参数由两部分组成，self_cfg_params参数的结构与逻辑回归训练相同，algorithm_dynamic_params是由本算法定制。
数据提供方有两方[data1, data2],  计算方有两方[compute1, compute2], 结果方有两方[result1, result2]。

- data1数据提供方
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型.
            "data_type": 1,      # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col1", "col2"]  # 自变量(特征列名), 训练或者预测算法选择的字段。
        }
      ]
  },
  "algorithm_dynamic_params": {
      "use_alignment": true,
      "label_owner": "data1",
      "label_column": "diagnosis",
      "psi_type": "T_V1_Basic_GLS254",  # 支持T_V1_Basic_GLS254, T_V1_Basic_SECP等
      "data_flow_restrict": {
        "data1": ["compute1"],
        "data2": ["compute2"],
        "compute1": ["result1"],
        "compute2": ["result2"]
      }
  }
}
```

## 5. information value

存在如下角色：数据提供方、计算方、结果接收方。每种角色都可能存在多个组织。下面分别按参与方角色说明配置：
- data1方(数据提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 0: unknown, 1: origin_data, 2: model
            "data_type": 1,      # 数据的格式. 
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",  # ID列名
            "selected_columns": ["col2", "col3"]  # 自变量(特征列名)
        }
      ]
  },
  "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 因变量(标签)
      "hyperparams": {           # 超参数
          "binning_type": 1,     # 分箱的方式。仅取值1或2。 1:等频分箱, 2:等距分箱
          "num_bin": 5,          # 分箱的箱数, 大于1的整数
          "postive_value": 1,    # 正例类别值, 整数或浮点数
          "negative_value": 0    # 负例类别值, 整数或浮点数
      },
      "calc_iv_columns": {   # 存储所有数据提供方的selected_columns，目的是给结果方使用
            "data1": ["col2", "col3"], # 数据提供方data1的selected_columns
            "data2": ["col4", "col5"]  # 数据提供方data2的selected_columns
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
      # 与数据提供方的该字段的内容完全相同，复制一份过来即可
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
      # 与数据提供方的该字段的内容完全相同，复制一份过来即可
  }
}
```

## 6. 评估指标说明

### 6.1 模型评估指标
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
指标取值都为[0,1]之间的小数

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
R2-score的取值为[0,1]之间的小数，其他指标的取值范围为[0, +∞)

### 6.2 信息价值(IV)评估指标
返回的是对每个已选择的特征计算的信息价值(IV)，格式如下：
```
{
    "feature_name_1": 0.01,
    "feature_name_2": 0.05,
    "feature_name_3": 0.2,
    "feature_name_4": 0.4,
    "feature_name_5": 1.3
}
```
IV值的取值范围为[0, +∞), 根据IV值评估特征的预测能力的评价基准如下表：
| IV范围 | 预测效果 |
| :----- | :------ |
| < 0.02 | 几乎没有 |
| 0.02 ~ 0.1 | 弱 |
| 0.1 ~ 0.3 | 中等 |
| 0.3 ~ 0.5 | 强 |
| > 0.5 | 难以置信，需确认 |

