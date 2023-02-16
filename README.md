# RDGC论文代码

## 数据集下载
用MNIST举例，将`MNIST.mat`放入到数据集文件夹，比如说 `./dataset`。
修改 `./configs/mnist_0.1_pretrain.yaml` 和 `./configs/mnist_0.1_train.yaml` 中的 `Data/Dataset/root_path` 参数成 `./dataset`
例如

```
Data:
  Dataset:
    name: MNIST
    normalize: MinMaxScaler
    miss_rate: 0.1
    root_path: ./dataset
    num_view: 2
    num_sample: 10000
```

## 用法
### pretrain
```
python tools/pretrain.py -c ./configs/mnist_0.1_pretrain.yaml
```

### train
```
python tools/train.py -c ./configs/mnist_0.1_train.yaml
```

也可以使用 `scripts/mnist.sh` 中给的命令。

## 输出
`MNIST` 默认输出到 `./output/mnist_0.1/`中。
