# FedML_YOLOv3

## 1. FedML简介

- FedML介绍[查看](FedML.md)

## 2. Docker环境配置

- Docker环境配置[查看](FedML.md)
    
## 3. 生产环境配置建议

   YOLOv3+COCO训练对显卡要求较高。建议GPU： V100或2080Ti。
   

## 4. 物体检测任务（YOLOv3+coco）

###  4.1 安装依赖包

    pip install -r requirements.txt

###  4.2 准备YOLOv3模型相关的数据和程序
    YOLOv3模型相关的资料都保存在/FedML/fedml_api/model/YOLOv3文件夹下。
####  4.2.1 数据集
YOLOv3使用的是coco数据集。为了模拟真实的生产环境，在实验时需要人工对数据进行切分。重新切分后的数据集在coco文件夹下。本次实验生成了10份相同规模的子数据集，以1...10命名，分别对应相应序号的client。每一个子数据集分为train集和test集，其中train集包含1641条数据，test集包含802条数据。

原YOLOv3项目通过读取数据文件列表遍历数据，实验改为传入文件夹路径，直接读取文件夹下所有文件。相应对/utils/datasets.py中ListDataset类的初始化函数进行了修改。

目前由于训练集和测试集较小，训练轮数少，无法涵盖所有的物体类型，训练效果差。在联邦学习化后遇到了一些bug，应该是由于原项目代码没有考虑到这些情况造成的。

####  4.2.2 模型类
对应原YOLOv3项目的models.py以及训练和测试依赖的工具类。基本与原YOLOv3项目相同。

####  4.2.3 YOLOv3Trainer类
get_model_params、set_model_params直接调用了model（Darknet）类的方法。

train函数修改自原项目中train.py，将定义参数的代码移动到main_fedavg.py中。由于model参数在接收server发来的全局参数时被更新，删除了从断点读取模型参数的代码。由于单轮训练的epoch一般较少，删除了保存断点的代码。删除了在训练过程中测试模型效果的代码。

test函数修改自test.py，将定义参数的代码移动到main_fedavg.py中。

####  4.2.4 YOLOv3ResultAggregator类
该类待完成。

###  4.3 定义联邦学习入口程序
从启动集群的命令可以看出，每个参与者（无论是server还是client）都会执行python3 ./main_fedavg.py，可见main_fedavg.py定义了联邦学习程序的入口。这个文件也在/FedML/fedml_experiments/distributed/fedYOLOv3文件夹下。

接下来对main函数的执行过程进行简要释义。在对其他机器学习任务进行联邦学习化时基本也遵循以下的步骤。
```
logging.getLogger().setLevel(logging.INFO)
```
定义logger，使打印内容在控制台显示。
```
comm, process_id, worker_number = FedML_init()
```
初始化进程的MPI参数。comm用于后续的MPI通讯，process_id获取进程的身份信息，worker_number实际表示client的总数。
```
# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)
logging.info(args)
```
获取参数。在执行run_fedavg_distributed_pytorch.sh脚本时指定了三个联邦学习相关参数，其他联邦学习过程与模型训练的参数都在add_args函数中被添加并储存在args中，之后args会作为参数传给相关函数和类的对象。

在add_args函数中可以看到，参数可以分为两类：第一类是联邦学习过程相关的参数，这类参数与具体任务无关，定义了联邦学习的聚集与分发过程。第二类是模型训练与测试相关的参数，这类参数与具体任务有关。对于YOLOv3模型，这部分参数对应了单机版本的YOLOv3代码中train.py和test.py两个文件中定义的opt包含的参数。需要注意的是模型定义相关的参数（/YOLOv3/config/yolov3.cfg）不需要包含在其中。
```
str_process_name = "FedAvg (distributed):" + str(process_id)
setproctitle.setproctitle(str_process_name)
```
初始化进程名。
```
logging.basicConfig...
hostname = socket.gethostname()
logging.info...
```
打印进程信息。
```
if process_id == 0:
wandb.init...
```
如果该进程的身份是server，在wandb平台注册任务。
```
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```
确保每次调试时初始化的值相同，使结果可复现。
```
logging.info...
device = init_training_device...
```
获取训练设备，如果服务器没有gpu则使用cpu训练，否则根据一定的的算法分配gpu。init_training_device函数可以根据生产环境重新定义。
```
train_path = os.path.join(os.path.join(args.data_path, str(process_id)), "train")
test_path = os.path.join(os.path.join(args.data_path, str(process_id)), "test")
```
通过args.data_path与该进程的process_id，生成client对应的数据集路径。4.2.2.1中说明了生成数据集的文件结构。由于server的process_id是0，所有client的process_id分别是1, ..., 10，因此client对应的数据集路径可以由coco文件夹路径与process_id拼接得到。数据集的定义可能需要根据生产环境进行修改。
```
model = create_model(args)
```
初始化model。create_model初始化并返回Darknet对象。
```
model_trainer = YOLOv3Trainer(model)
result_aggregator = YOLOv3ResultAggregator()
```
初始化model_trainer和result_aggregator。二者都是模型相关的，因此对于不同的模型要重新定义。见4.2.2.3和4.2.2.4.

###  4.4 定义联邦学习入口程序
从启动集群的命令可以看出，每个参与者（无论是server还是client）都会执行python3 ./main_fedavg.py，可见main_fedavg.py定义了联邦学习程序的入口。这个文件也在/FedML/fedml_experiments/distributed/fedYOLOv3文件夹下。

接下来对main函数的执行过程进行简要释义。在对其他机器学习任务进行联邦学习化时基本也遵循以下的步骤。
```
logging.getLogger().setLevel(logging.INFO)
```
定义logger，使打印内容在控制台显示。
```
comm, process_id, worker_number = FedML_init()
```
初始化进程的MPI参数。comm用于后续的MPI通讯，process_id获取进程的身份信息，worker_number实际表示client的总数。
```
# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)
logging.info(args)
```
获取参数。在执行run_fedavg_distributed_pytorch.sh脚本时指定了三个联邦学习相关参数，其他联邦学习过程与模型训练的参数都在add_args函数中被添加并储存在args中，之后args会作为参数传给相关函数和类的对象。

在add_args函数中可以看到，参数可以分为两类：第一类是联邦学习过程相关的参数，这类参数与具体任务无关，定义了联邦学习的聚集与分发过程。第二类是模型训练与测试相关的参数，这类参数与具体任务有关。对于YOLOv3模型，这部分参数对应了单机版本的YOLOv3代码中train.py和test.py两个文件中定义的opt包含的参数。需要注意的是模型定义相关的参数（/YOLOv3/config/yolov3.cfg）不需要包含在其中。
```
str_process_name = "FedAvg (distributed):" + str(process_id)
setproctitle.setproctitle(str_process_name)
```
初始化进程名。
```
logging.basicConfig...
hostname = socket.gethostname()
logging.info...
```
打印进程信息。
```
if process_id == 0:
wandb.init...
```
如果该进程的身份是server，在wandb平台注册任务。
```
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```
确保每次调试时初始化的值相同，使结果可复现。
```
logging.info...
device = init_training_device...
```
获取训练设备，如果服务器没有gpu则使用cpu训练，否则根据一定的的算法分配gpu。init_training_device函数可以根据生产环境重新定义。
```
train_path = os.path.join(os.path.join(args.data_path, str(process_id)), "train")
test_path = os.path.join(os.path.join(args.data_path, str(process_id)), "test")
```
通过args.data_path与该进程的process_id，生成client对应的数据集路径。4.2.1中说明了生成数据集的文件结构。由于server的process_id是0，所有client的process_id分别是1, ..., 10，因此client对应的数据集路径可以由coco文件夹路径与process_id拼接得到。数据集的定义可能需要根据生产环境进行修改。
```
model = create_model(args)
```
初始化model。create_model初始化并返回Darknet对象。
```
model_trainer = YOLOv3Trainer(model)
result_aggregator = YOLOv3ResultAggregator()
```
初始化model_trainer和result_aggregator。二者都是模型相关的，因此对于不同的模型要重新定义。见4.2.3和4.2.4.

###  4.5 启动集群
用于启动集群相关的程序在是./fedml_experiments/distributed/fedYOLOv3/run_fedavg_distributed_pytorch.sh。具体解释参考FedML联邦学习框架  [查看](FedML.md)的4.4节.

启动集群的命令示例为：
```
    sh run_fedavg_distributed_pytorch.sh 10 5 50
```
意为共有10个client，每轮训练有5个client参加，一共训练50轮。由于生成数据集的限制，所有训练者的个数最多是10.

