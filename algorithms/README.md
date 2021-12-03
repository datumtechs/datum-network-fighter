# 算法相关参数说明

相关算法有logistic regression、linear regression、DNN、XGBoost等，下面对每个算法说明：

## logistic regression

+ **LR训练**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",    # 本方party_id
  "data_party": {
    "input_file": "path/to/train_data_input_file",    # 数据集文件的所在路径及文件名
    "key_column": "CLIENT_ID",            # ID列
    "selected_columns": ["col1", "col2"]  # 自变量(特征)
  },
  "dynamic_parameter": {
    "label_owner": "p0",       # 标签所在方的party_id
    "label_column": "Y",       # 因变量(标签)
    "algorithm_parameter": {   # 逻辑回归的一些参数
      "epochs": 10,            # 训练轮次
      "batch_size": 256,       # 批量大小
      "learning_rate": 0.1,    # 学习率
      "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
      "validation_set_rate": 0.2,  # 验证集占输入数据集的比例
      "predict_threshold": 0.5     # 验证集预测结果的分类阈值
    }
  }
}
```

cfg_dict结构中，动态参数dynamic_parameter根据算法的不同有所不同；除了dynamic_parameter之外的内容，剩下的结构在不同的算法中都是一样的。

dynamic_parameter参数里，label_owner是指标签拥有方，根据任务而变化。label_column是监督学习中的标签，对应于数据集中标签列的列名。algorithm_parameter参数逻辑回归训练算法相关的可调参数。



+ **LR预测**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",
  "data_party": {
      "input_file": "path/to/predict_data_input_file",  # 数据集文件的所在路径及文件名
      "key_column": "CLIENT_ID",             # ID列
      "selected_columns": ["col1", "col2"]   # 自变量(特征)
    },
  "dynamic_parameter": {
    "model_restore_party": "p0",  # 模型所在方
    "model_path": "file_path",    # 模型所在的路径，需填绝对路径。
    "predict_threshold": 0.5      # 预测结果的分类阈值
  }
}
```

cfg_dict结构中，动态参数dynamic_parameter根据算法的不同有所不同；除了dynamic_parameter之外的内容，剩下的结构在不同的算法中都是一样的。

dynamic_parameter参数里，模型提供方也被当做数据方，会被赋予一个单独的party_id，这个model_restore_party填的就是模型提供方的party_id。model_path指定了模型提供方中模型的所在路径及模型名。



## linear regression

+ **LinR训练**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",  # 本方party_id
  "data_party": {
    "input_file": "path/to/train_data_input_file",  # 数据集文件的所在路径及文件名
    "key_column": "CLIENT_ID",            # ID列
    "selected_columns": ["col1", "col2"]  # 自变量(特征)
  },
  "dynamic_parameter": {
    "label_owner": "p0",       # 标签所在方的party_id
    "label_column": "Y",       # 因变量(标签)
    "algorithm_parameter": {   # 逻辑回归的一些参数
      "epochs": 10,            # 训练轮次
      "batch_size": 256,       # 批量大小
      "learning_rate": 0.1,    # 学习率
      "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
      "validation_set_rate": 0.2   # 验证集占输入数据集的比例
    }
  }
}
```

cfg_dict结构中，动态参数dynamic_parameter根据算法的不同有所不同；除了dynamic_parameter之外的内容，剩下的结构在不同的算法中都是一样的。

dynamic_parameter参数里，label_owner是指标签拥有方，根据任务而变化。label_column是监督学习中的标签，对应于数据集中标签列的列名。algorithm_parameter参数线性回归训练算法相关的可调参数。



**LinR预测**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",
  "data_party": {
      "input_file": "path/to/predict_data_input_file",  # 数据集文件的所在路径及文件名
      "key_column": "CLIENT_ID",             # ID列
      "selected_columns": ["col1", "col2"]   # 自变量(特征)
    },
  "dynamic_parameter": {
    "model_restore_party": "p0",  # 模型所在方
    "model_path": "file_path"     # 模型所在的路径，需填绝对路径。
  }
}
```

cfg_dict结构中，动态参数dynamic_parameter根据算法的不同有所不同；除了dynamic_parameter之外的内容，剩下的结构在不同的算法中都是一样的。

dynamic_parameter参数里，模型提供方也被当做数据方，会被赋予一个单独的party_id，这个model_restore_party填的就是模型提供方的party_id。model_path指定了模型提供方中模型的所在路径及模型名。



## DNN

+ **DNN训练**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",  # 本方party_id
  "data_party": {
    "input_file": "path/to/train_data_input_file",  # 数据集文件的所在路径及文件名
    "key_column": "CLIENT_ID",            # ID列
    "selected_columns": ["col1", "col2"]  # 自变量(特征)
  },
  "dynamic_parameter": {
    "label_owner": "p0",       # 标签所在方的party_id
    "label_column": "Y",       # 因变量(标签)
    "algorithm_parameter": {   # 逻辑回归的一些参数
      "epochs": 10,            # 训练轮次
      "batch_size": 256,       # 批量大小
      "learning_rate": 0.1,    # 学习率
      "layer_units": [32, 128, 32, 1],  # 隐藏层与输出层的每层单元数，例子中有3个隐藏层，每层的单元数分别是32，128，32。输出层单元数是1。
      "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],   # 隐藏层与输出层的每层的激活函数
      "init_method": "random_normal",  # 指定模型参数初始化方法
      "use_intercept": true,       # 指定模型结构中是否使用bias
      "optimizer": "sgd",          # 优化器
      "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
      "validation_set_rate": 0.2,  # 验证集占输入数据集的比例
      "predict_threshold": 0.5     # 验证集预测结果的分类阈值
    }
  }
}
```



+ **DNN预测**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",
  "data_party": {
      "input_file": "path/to/predict_data_input_file",  # 数据集文件的所在路径及文件名
      "key_column": "CLIENT_ID",   # ID列
      "selected_columns": ["col1", "col2"]   # 自变量(特征)
    },
  "dynamic_parameter": {
    "model_restore_party": "p0",  # 模型所在方
    "model_path": "file_path",    # 模型所在的路径，需填绝对路径。
    "algorithm_parameter": {
        "layer_units": [32, 128, 32, 1],   # 隐藏层与输出层的每层单元数
        "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],  # 隐藏层与输出层的每层的激活函数
        "use_intercept": true,    # 指定模型结构中是否使用bias
        "predict_threshold": 0.5  # 二分类的阈值
    }
  }
}
```



## XGBoost

+ **XGBoost训练**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",  # 本方party_id
  "data_party": {
    "input_file": "path/to/train_data_input_file",  # 数据集文件的所在路径及文件名
    "key_column": "CLIENT_ID",            # ID列
    "selected_columns": ["col1", "col2"]  # 自变量(特征)
  },
  "dynamic_parameter": {
    "label_owner": "p0",       # 标签所在方的party_id
    "label_column": "Y",       # 因变量(标签)
    "algorithm_parameter": {   # 逻辑回归的一些参数
      "epochs": 10,            # 训练轮次
      "batch_size": 256,       # 批量大小
      "learning_rate": 0.01,   # 学习率
      "num_trees": 3,          # 多少棵树
      "max_depth": 4,          # 树的深度
      "num_bins": 5,           # 特征的分箱数
      "num_class": 2,          # 标签的类别数
      "lambd": 1.0,            # L2正则项系数, [0, +∞)
      "gamma": 0.0,            # 复杂度控制因子，用于防止过拟合。
      "use_validation_set": true,  # 是否使用验证集，true-用，false-不用
      "validation_set_rate": 0.2,  # 验证集占输入数据集的比例
      "predict_threshold": 0.5     # 验证集预测结果的分类阈值
    }
  }
}
```



+ **XGBoost预测**

算法所需输入的cfg_dict参数

```
{
  "party_id": "p0",
  "data_party": {
      "input_file": "path/to/predict_data_input_file",  # 数据集文件的所在路径及文件名
      "key_column": "CLIENT_ID",             # ID列
      "selected_columns": ["col1", "col2"]   # 自变量(特征)
    },
  "dynamic_parameter": {
    "model_restore_party": "p0",  # 模型所在方
    "model_path": "file_path",    # 模型所在的路径，需填绝对路径
    "algorithm_parameter": {
        "num_trees": 3,   # 多少棵树
        "max_depth": 4,   # 树的深度
        "num_bins": 5,    # 特征的分箱数
        "num_class": 2,   # 标签的类别数
        "lambd": 1.0,     # L2正则项系数, [0, +∞)
        "gamma": 0.0,     # 复杂度控制因子，用于防止过拟合。
        "predict_threshold": 0.5  # 二分类时的阈值
    }
  }
}
```



## Private Set Intersection

Doing