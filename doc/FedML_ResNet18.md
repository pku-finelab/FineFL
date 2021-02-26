# FedML_ResNet18

## 1. FedML简介

- FedML介绍[查看](FedML.md)

## 2. Docker环境配置

- Docker环境配置[查看](FedML.md)
    
## 3. 生产环境配置建议

ResNet18+cifar10对配置要求远低于YOLOv3+COCO。
    

## 4. 物体识别任务（FedML_ResNet18）

该任务可用于测试环境是否配置完成。关于将ResNet18+cifar10联邦学习化的过程可参考FedML联邦学习框架  [查看](FedML.md)的第4节。

运行方法：

```
cd fedml_experiments/distributed/fedResNet18/
sh run_feavg_distributed_pytorch.sh 5 4 20
```
训练结果：准确率0.768.

