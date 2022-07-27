# 非隐私算法(non-privacy algorithm)相关参数说明
算法所需输入的cfg_dict参数的结构由两大部分组成，一是本方的配置参数(self_cfg_params)，二是算法动态参数(algorithm_dynamic_params)。
+ 本方的配置参数(self_cfg_params)：

   暂时仅包含input_data字段，如果后面有新的需求变化，可以增加新的字段。input_data里面的字段含义如下：
  ```
  ① input_type：输入数据的类型. (算法用，标识数据使用方式).  0:unknown, 1:origin_data, 2:model等等。可以根据数据类型的增加而增加。暂时只有两种类型：源数据，模型结果
  ② access_type: 访问数据的方式, (fighter用，决定是否预先加载数据). 0:unknown, 1:local, 2:url等等。现阶段仅支持local
  ③ data_type：数据的格式, (算法用，标识数据格式). 0:unknown, 1:csv, 2:dir, 3:binary, 4:xls, 5:xlsx, 6:txt, 7:json等等。现阶段仅支持csv和folder。
  ④ data_path：如果数据在本地(access_type=local)，则这里是数据路径。如果数据在远程(access_type=url)，则这里是超链接
  ⑤ key_column：id列，作为样本的唯一标识。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
  ⑥ selected_columns：选择的列，指的是自变量(特征列)。如果数据的格式(data_type)是非二维表类型, 如folder/bin/图像/文本/音频/视频等格式，则无此字段
  ```
+ 算法动态参数(algorithm_dynamic_params)：

  所含字段随着算法变化而变化，需根据算法来定制

## 1. 训练和预测
训练与预测的相关算法有logistic regression、linear regression、DNN、XGBoost、SVM、KNN、KMeans，下面对每个算法说明：

### 1.1 logistic regression
#### 1.1.1 LR训练

​训练存在如下角色：数据提供方、计算方、结果接收方。暂时仅支持这三种角色各只有一个的场景。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
  "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的浮点数
          "init_method": "random_normal", # 模型初始化方法，可选['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'xavier_uniform', 'xavier_normal']
          "use_intercept": true,   # 模型结构是否使用bias，true-用，false-不用
          "optimizer": "Adam",     # 优化器。可选['SGD', 'Adam', 'RMSProp', 'Momentum', 'Adadelta', 'Adagrad', 'Ftrl']
          "activation": "softmax", # 激活函数。可选["sigmoid", "softmax"]
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
          "predict_threshold": 0.5     # 如果使用验证集，验证集预测结果的分类阈值，浮点数，值域[0,1]
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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

#### 1.1.2 LR预测

预测存在如下角色：数据提供方、计算方、结果接收方、模型提供方。暂时仅支持这三种角色各只有一个的场景。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
  "algorithm_dynamic_params": {
      "model_restore_party": "model1",  # 模型所在方
      "data_flow_restrict": {           # 数据流向限制
          "data1": ["compute1"],        # 数据提供方data1只流向计算方compute1
      	  "model1": ["compute1"],       # 模型提供方model1只流向计算方compute1
          "compute1": ["result1"]       # 计算方compute1只流向结果接收方result1
      }
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
            "input_type": 2,    # 输入数据的类型. 2表示输入数据是模型
            "data_type": 2      # 数据的格式. 2表示是目录
            "data_path": "path/to/data"   # 数据所在的本地路径
        }
      ]
  },
  "algorithm_dynamic_params": {
      # 与数据提供方的该字段的内容完全相同，复制一份过来即可
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



### 1.2 linear regression

#### 1.2.1 LinR训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：
- data1方(数据提供方)的配置
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
   "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的浮点数
          "init_method": "random_normal", # 模型初始化方法，可选['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'xavier_uniform', 'xavier_normal']
          "use_intercept": true,   # 模型结构是否使用bias，true-用，false-不用
          "optimizer": "Adam",     # 优化器。可选['SGD', 'Adam', 'RMSProp', 'Momentum', 'Adadelta', 'Adagrad', 'Ftrl']
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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

- result1方(计算方)的配置
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

#### 1.2.2 LinR预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。



### 1.3 DNN

#### 1.3.1 DNN训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

- data1方(数据提供方)
```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,       # 输入数据的类型.
            "data_type": 1,        # 数据的格式.
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
  "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "epochs": 10,            # 训练轮次，大于0的整数
          "batch_size": 256,       # 批量大小，大于0的整数
          "learning_rate": 0.1,    # 学习率，大于0的浮点数
          "layer_units": [32, 1],  # 隐藏层与输出层的每层单元数。假设列表有n个元素，那么前n-1层分别对应各个隐藏层，最后一层对应输出层。列表元素是大于0的整数
          "layer_activation": ["sigmoid", "sigmoid"],   # 隐藏层与输出层的每层的激活函数。列表元素值可选["sigmoid", "relu", "tanh", "softmax", ""], 其中""表示该层不使用激活函数
          "init_method": "random_normal",  # 指定模型参数初始化方法, 
                                          # 仅支持random_normal/random_uniform/zeros/ones
          "use_intercept": true,   # 模型结构中是否使用bias, true-用，false-不用
          "optimizer": "Adam",     # 优化器。可选['SGD', 'Adam', 'RMSProp', 'Momentum', 'Adadelta', 'Adagrad', 'Ftrl']
          "dropout_prob": 0.0,     # 训练过程中随机丢弃神经元的概率。浮点数，值域[0,1)。当取0时，表示不使用dropout。
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
          "predict_threshold": 0.5     # 如果使用验证集，验证集预测结果的分类阈值，浮点数，值域[0,1]
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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



#### 1.3.2 DNN预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。



### 1.4 XGBoost

#### 1.4.1 XGBoost训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

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
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
  "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # XGBoost的超参数
          "n_estimators": 3,       # 构建多少棵树，大于0的整数
          "max_depth": 4,          # 每棵树的最大深度，大于0的整数
          "max_bin": 5,            # 特征的分箱数，大于0的整数
          "learning_rate": 0.01,   # 学习率，大于0的浮点数          
          "subsample": 0.8,        # 样本采样率，即行方向的采样率，浮点数，值域(0,1]
          "colsample_bytree": 0.8, # 特征采样率，即列方向的采样率，浮点数，值域(0,1]
          "reg_lambda": 1.0,       # L2正则项系数, 浮点数，值域[0, +∞)
          "gamma": 0.0,            # 复杂度控制因子，用于防止过拟合。浮点数，值域[0, +∞)
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2,  # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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



#### 1.4.2 XGBoost预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。



### 1.5 SVM

#### 1.5.1 SVM训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置

```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
   "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "C": 1.0,          # L2正则的系数，大于0的浮点数
          "kernel": "rbf",   # 核函数类型。可选["rbf", "linear", "poly", "sigmoid"]，rbf为高斯核函数，linear为线性核函数，poly为多项式核函数，sigmoid为sigmoid核函数
          "degree": 3,       # 如果核函数为poly时，指定多项式的次数
          "max_iter": -1,    # 指定最大的迭代次数，-1或大于0的整数。当取值-1时表示不限制最大的迭代次数，由程序自行决定。
          "decision_function_shape": "ovr",   # 多分类时，指定多分类的方式，对二分类不起作用。可选['ovr', 'ovo']。即one vs rest, one vs one
          "tol": 0.001,    # 损失容忍度，当损失小于这个容忍度时，可以停止训练
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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

- result1方(计算方)的配置

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

#### 1.5.2 SVM预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。



### 1.6 KNN

#### 1.6.1 KNN训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置

```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
   "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "n_neighbors": 5,      # K的取值，选择多少个最邻近点。大于0的整数
          "distance_metric": "minkowski", # 距离的衡量方式。暂时仅支持"minkowski", 即明可夫斯基距离
          "metric_p": 2,   # 明可夫斯基距离的参数，当p=1时是曼哈顿距离，当p=2时是欧氏距离。大于0的整数
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
          "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
          "validation_set_rate": 0.2   # 如果使用验证集，验证集占输入数据集的比例，浮点数，值域(0,1)
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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

- result1方(计算方)的配置

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

#### 1.6.2 KNN预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。



### 1.7 KMeans

#### 1.7.1 KMeans训练

训练存在如下角色：数据提供方、计算方、结果接收方。下面分别按参与方角色说明配置：

- data1方(数据提供方)的配置

```
{
  "self_cfg_params": {
      "party_id": "data1",    # 本方party_id
      "input_data": [
        {
            "input_type": 1,     # 输入数据的类型. 数字的含义详见本文档开头
            "data_type": 1,      # 数据的格式. 数字的含义详见本文档开头
            "data_path": "path/to/data",  # 数据所在的本地路径
            "key_column": "col1",                 # ID列名
            "selected_columns": ["col2", "col3"]  # 特征列名
        }
      ]
  },
   "algorithm_dynamic_params": {
      "label_owner": "data1",    # 标签列所在方的party_id
      "label_column": "Y",       # 标签列名
      "hyperparams": {           # 与算法相关的超参数
          "n_clusters": 8,    # K的取值，聚类的类别数。取值大于等于2的整数
          "init_method": "k-means++",  # 聚类中心点初始化方法。可选["k-means++", "random"]
          "n_init": 10,      # 重复执行多少次KMeans算法，每次初始化不同的中心点。最后取最优的一次作为结果。大于0的整数
          "max_iter": 300,   # 单次运行KMeans算法的最大的迭代数。大于0的整数
          "tol": 0.0001,   # 收敛容忍度。对连续两次迭代的聚类中心相差小于这个值，可认为训练已完成。浮点数，值域(0,1)
          "random_seed": -1,     # 是否使用随机种子，可选[-1~2^32-1之间的所有整数]。-1表示不使用随机种子， 其他整数表示使用随机种子
      },
      "data_flow_restrict": {       # 数据流向限制
      	  "data1": ["compute1"],    # 数据提供方data1只流向计算方compute1
          "compute1": ["result1"]   # 计算方compute1只流向结果接收方result1
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

- result1方(计算方)的配置

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

#### 1.7.2 KMeans预测

格式与[逻辑回归的预测](# 1.1.2 LR预测)相同。




## 2. 评估指标说明

### 2.1 模型评估
现阶段模型评估有4种类型，如下：

+ **二分类模型评估**
  现在支持该类模型评估的训练算法有logistic regression, DNN, XGBoost, SVM, KNN.
  生成的评估指标的内容如下：

  ```
  {
      "accuracy": 0.91,
      "f1_score": 0.85,
      "precision": 0.93,
      "recall": 0.81
  }
  ```
  指标取值都为[0,1]之间的小数

+ **多分类模型评估**

  现在支持该类模型评估的训练算法有logistic regression, DNN, XGBoost, SVM, KNN.
  生成的评估指标的内容如下：

  ```
  {
  	"accuracy": 0.91,
  	"f1_score_micro": 0.9,
  	"precision_micro": 0.95,
  	"recall_micro": 0.85,
  	"f1_score_macro": 0.88,
  	"precision_macro": 0.93,
  	"recall_macro": 0.82
  }
  ```
  指标取值都为[0,1]之间的小数

+ **回归类型模型评估**
  现在支持该类模型评估的训练算法有linear regression, DNN, SVM.
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

+ **样本轮廓系数**

  这个评估指标是专门针对聚类算法的，如KMeans

  生成的评估指标的内容如下：

  ```
  {
  	"silhouette_score": 0.9
  }
  ```

  该指标称为轮廓系数，综合考虑了簇内的内聚度和簇外的分离度，该指标的计算方式为：
  ```
  for 每个样本点：
  	1. 计算当前样本点到它所属簇中的所有其他样本点的平均距离，记为distanceMeanIn，体现内聚度
  	2. 计算当前样本点到它所属簇之外的其他簇内的所有样本点的平均距离，记为distanceMeanOut，体现分离度
  	3. 计算该对象的轮廓系数 score = (distanceMeanOut - distanceMeanIn) / max(distanceMeanOut, distanceMeanIn)
  最后，对所有样本点的轮廓系数求平均，即可得到样本集的轮廓系数。
  由此可知该指标的取值范围为[-1, 1]. 值越大越好.
  ```

  

