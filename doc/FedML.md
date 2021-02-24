## 1. FedML

  https://github.com/FedML-AI/FedML

## 2. 修改后的FedML

  https://github.com/yukizhao1998/FineFL

## 3. docker容器间部署的配置方法

#### 步骤1：拉取ubuntu16.04或cuda镜像
    docker pull ubuntu:16.04

如果训练需要使用gpu，则需要拉取对应版本的cuda镜像以及安装nvidia-docker。

#### 步骤2：使用Dockerfile为镜像安装必要的软件包并制作新镜像

##### a.创建Dockerfile文件夹及Dockerfile文件：
    mkdir Dockerfile
    cd Dockerfile
    vim Dockerfile

Dockerfile内容如下（如果使用cuda镜像则需要修改第一行的源镜像）：
```
FROM ubuntu:16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update
ADD ./Anaconda3-2019.07-Linux-x86_64.sh ./anaconda.sh
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN  /bin/bash ./anaconda.sh -b -p /opt/conda  && rm ./anaconda.sh && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && echo "conda activate base" >> ~/.bashrc && find /opt/conda/ -follow -type f -name '*.a' -delete && find /opt/conda/ -follow -type f -name '*.js.map' -delete &&  /opt/conda/bin/conda clean -afy
CMD [ "/bin/bash" ]
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
&& conda config --set show_channel_urls yes \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
&& conda install pytorch torchvision cudatoolkit=10.2 \
&& conda install -c anaconda mpi4py \
&& pip install --upgrade wandb \
&& conda install scikit-learn \
&& conda install numpy \
&& conda install h5py \
&& conda install setproctitle \
&& conda install networkx \
&& pip install paho-mqtt 
RUN apt-get update
RUN echo "Y" | apt-get install vim
RUN echo "Y" | apt-get install ssh
```

##### b.下载Anaconda安装包到当前目录：
    wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh

##### c.制作镜像
    sudo docker build -t fedml .

#### 步骤三：创建和运行容器
##### 1）创建容器
使用之前制作的fedml镜像，创建一个HeadNode及若干个ComputeNode容器。此处使用docker的文件映射机制将本地项目文件目录映射到容器的指定目录中，共享执行代码及数据。一下以创建一个HeadNode和两个ComputeNode为例。
```
    sudo docker run --gpus all -shm-size 8G -it -p 2001:80 -v [本地项目地址]:[容器中项目地址] --name HeadNode -h HeadNode -d fedml /bin/bash
    sudo docker run --gpus all -shm-size 8G -it -p 2002:80 -v [本地项目地址]:[容器中项目地址] --name ComputeNode1 -h ComputeNode1 -d fedml /bin/bash
    sudo docker run --gpus all -shm-size 8G -it -p 2003:80 -v [本地项目地址]:[容器中项目地址] --name ComputeNode2 -h ComputeNode2 -d fedml /bin/bash
```
##### 2）运行容器
```
    sudo docker exec -it HeadNode /bin/bash
    sudo docker exec -it ComputeNode1 /bin/bash
    sudo docker exec -it ComputeNode2 /bin/bash
```

#### 步骤四：配置容器间的ssh免密登陆

##### 1）修改密码和权限
对于每一个容器，进行如下操作：
运行passwd root，修改root密码
进入/etc/ssh/sshd_config文件，修改PermitRootLogin yes，然后重启ssh服务 /etc/init.d/ssh restart
进入/etc/hosts文件，找到并记录容器ip地址

##### 2）生成密钥，设置免密登陆
对于每一个容器，进行如下操作：

将其他容器的ip地址及host名添加到/etc/hosts的末尾，如：
    172.17.0.2      HeadNode
    172.17.0.3      ComputeNode1
    172.17.0.4      ComputeNode2

生成密钥：
    cd ~/.ssh（若不存在则创建）
    ssh-keygen -t rsa（一直回车）
    chmod 644 ~/.ssh/id_rsa.pub（与上一条语句中密钥的保存地址相同）
    cat ./id_rsa.pub >> ./authorized_keys

设置本容器免密登陆到其他容器（NodeName是所有其他容器的名字，有几个容器就执行几次语句）：
    ssh-copy-id -i ~/.ssh/id_rsa.pub [NodeName]

    如果ssh [NodeName]能直接登录到其他容器，则设置成功。


## 4.机器学习任务联邦学习化的一般方法

#### 4.1.1 准备环境
    为server和所有的client安装训练相关的依赖包。
#### 4.1.2 准备模型相关的数据和程序
    所有模型相关的资料保存在/FedML/fedml_api/model/[model_name]文件夹下。
##### 4.1.2.1 数据集
目前的框架中，所有client运行同一份代码，因此所有client的训练和测试数据应该储存在同一路径，或能通过同样的方法检索到的路径中。具体来说，在FedML-Experiments层的main_fedavg.py中定义了训练集和测试集的路径。在训练和测试的过程中，程序也使用同样的方法使用路径下的数据。


在实验环境中，可以通过切分数据集模拟生产环境中client生成的数据。

##### 4.1.2.2 模型类

对应单机版本的项目中对模型的定义，一般不需要修改。

##### 4.1.2.3 [MyModel]Trainer类

继承自fedml_core.trainer.model_trainer.ModelTrainer抽象类，封装了get_model_params（获取模型参数）、set_model_params（设置模型参数）、train（训练）、test（测试）四个接口。

[MyModel]Trainer类在main_fedavg.py中被初始化，初始化时会传入模型类的对象，保存为名为model的成员变量。之后[MyModel]Trainer类可以通过访问self.model使用模型的实例方法。在进行获取模型参数、设置模型参数、训练、测试等操作时使用的model都是self.model。

get_model_params：返回模型参数（dict类型）。

set_model_params：传入dict类型的模型参数，用其更新self.model.

train(self, device, args, train_path)：在client进行本地训练时被调用，作用是训练self.model。训练所需的参数，如epoch和batch_size都存储在args中。传入的参数都在main_fedavg.py中被定义。

test(self, device, args, test_path)：在client使用本地测试集测试全局模型的效果时被调用。测试所需的参数都存储在args中，传入的参数都在main_fedavg.py中被定义。返回值是被json序列化的字典，键为测试结果标签，值为对应的测试结果。如{“precision”:0.9, “recall”:”0.76”}。注意返回值必须被序列化，否则无法传输。

##### 4.1.2.4 [MyModel]ResultAggregator类

server对client更新后发回的模型进行聚集（aggregate）后，如果本轮需要对模型进行评估，server会将模型广播给所有client，client在本地测试后将测试结果发回给server。收到所有client的测试结果后，server会调用[MyModel]ResultAggregator的aggregate_result方法对测试结果进行统计，得到（一般是）加权平均后的测试结果。

aggregate_result函数收到的result_dict参数是一个字典，字典的键为client的序号client_idx，从0开始。result[client_idx]是一个字典，字典的键为”result”和”sample_number”。result[client_idx][”result”]是一个字典，与[MyModel]Trainer类中test的返回值的格式相同。result[client_idx][”sample_number”]是一个整数，是client数据集的规模，用于对训练结果进行加权平均。

#### 4.1.3 启动集群

启动集群相关的程序在/FedML/fedml_experiments/distributed/fed[MyModel]文件夹下。

run_fedavg_distributed_pytorch.sh定义了集群的启动方法，通过执行本脚本开始训练。进程间通过mpi协议进行通讯，脚本中的mpirun命令定义了进程数（-np $PROCESS_NUM）、参与通讯的hosts（-hostfile ./mpi_host_file）、每个host执行的程序（python3 ./main_fedavg.py）以及其他参数：

    CLIENT_NUM：所有参与训练者个数
    WORKER_NUM：每轮训练参与者的个数
    ROUND：联邦学习训练轮数

mpi_host_file中定义了所有hosts，作为mpirun命令的参数。所有参与训练的hosts名都应写在其中，每个一行。其中，第一行必须是server（aggregator）的host名，后面的每一行都是一个client（trainer）的host名，其数量一般应该和CLIENT_NUM相同。在MPI集群启动后，所有的host会依次启动一个进程（process），每个进程被分配一个唯一的process_id，process_id是从0开始的连续整数，分配顺序与mpi_host_file中的定义顺序相同。

举例来说，如果启动有1个server（host名为HeadNode）和2个client（host名为ComputeNode1和ComputeNode2）的联邦学习训练集群，每轮有1个client参与，训练50轮，那么mpi_host_name的内容为：

    HeadNode
    ComputeNode1
    ComputeNode2


启动集群的命令为：

    sh run_fedavg_distributed_pytorch.sh 10 5 50

集群启动后，HeadNode启动的server进程的process_id为0，ComputeNode1启动的进程的process_id为1，ComputeNode1启动的进程的process_id为2.

#### 4.1.4运行联邦学习程序

从启动集群的命令可以看出，每个参与者（无论是server还是client）都会执行python3 ./main_fedavg.py，可见main_fedavg.py定义了联邦学习程序的入口。这个文件也在/FedML/fedml_experiments/distributed/fed[MyTrainer]文件夹下。

对main函数的定义参照4.2.4。在对其他机器学习任务进行联邦学习化时基本遵循相同的步骤，只需对具体模型相关的内容进行修改。

## 5.FedML-API的主要接口介绍
FedML-API与具体模型无关的接口在./fedml_api/distributed/fedavg路径下。

### 5.1 FedAvgAPI.py
定义了server和client的初始化函数。如果不修改fedavg的流程（模型无关），这个文件一般不需要修改。
#### 5.1.1 FedML_init
获取MPI的相关参数。
#### 5.1.2 FedML_FedAvg_distributed
接收上一层定义的模型和参数，用于初始化server和client。函数根据process_id决定本进程的角色，从而初始化server（调用init_server）或client（init_client）。
#### 5.1.3 init_server
初始化server。server包含aggregator和manager，aggregator（FedAVGAggregator）定义了模型聚集的过程，manager（FedAvgServerManager中）定义了任务分发、通信等功能。
#### 5.1.4 init_client
初始化client。client包含trainer和manager，trainer（FedAVGTrainer）定义了模型训练的过程，manager（FedAvgClientManager）定义了任务接收、通信等功能。

### 5.2 FedAvgServerManager.py
FedAVGrServerManager类，继承自fedml_core.distributed.server.server_manager中的ServerManager。如果不修改fedavg的流程（模型无关），这个文件一般不需要修改。
#### 5.2.1 aggregator
FedAVGAggregator类的一个对象。
#### 5.2.2 run
继承自ServerManager，首先调用register_message_receive_handlers注册所有的handler函数，然后启动mqtt服务。
#### 5.2.3 send_init_message
在FedAvgAPI.init_server中被调用，首先选取第一轮参与训练的client，然后将模型的初始参数广播给所有的client。
#### 5.2.4 register_message_receive_handlers
调用继承自ServerManager的register_message_receive_handler，依次定义所有的handler（handle_massage_xxx形式的函数）。本质上是将所有消息类型与消息类型的handler存储在一个字典中，当收到某一类型的消息时进行查询和处理。所有相关的handler在下文中定义。
#### 5.2.5 handle_message_receive_model_from_client
收到client发来的模型更新后的参数以及参与训练的数据量时被调用。实际上定义了训练的流程，一般不需要修改。
#### 5.2.6 send_message_init_config
封装给client的初始化信息并发送，在所有训练（第一轮训练）开始前执行。
#### 5.2.7 send_message_sync_model_to_client
封装模型aggregate后更新的参数并发送，在每轮训练完成、aggregate结束后将更新后的参数发送给所有client时被调用。

### 5.3 FedAVGAggregator.py
定义了FedAVGAggregator类，即联邦学习中的aggregator。
#### 5.3.1 self.trainer
自定义[MyModel]Trainer类的一个对象。主要用来对模型参数进行全局更新等操作，并不实际进行训练。
#### 5.3.2 aggregate
当server收到了本轮训练中所有参与训练的client更新后的模型参数时被调用，对更新后的模型参数做加权平均，得到全局模型参数。



### 5.4 FedAvgClientManager.py
定义FedAVGClientManager类，继承自fedml_core.distributed.server.client_manager中的ClientManager。如果不修改fedavg的流程（模型无关），这个文件一般不需要修改。
#### 5.4.1 self.trainer
FedAVGTrainer类的一个对象。
#### 5.4.2 run
继承自ClientManager，首先调用register_message_receive_handlers注册所有的handler函数，然后启动mqtt服务。
#### 5.4.3 send_model_to_server
封装本地更新后的模型参数和参与训练的数据条数并发送给server，在__train函数的结尾被执行。
#### 5.4.4 handle_message_init
在服务刚启动，收到来自server的初始化信息时被调用。作用是更新trainer中的模型参数，初始化round_idx=0，然后调用__train进行一轮训练。
#### 5.4.5 handle_message_receive_model_from_server
在收到来自server的模型更新信息时被调用。作用是更新trainer中的模型参数，将round_idx加1，然后调用__train进行一轮训练。
#### 5.4.6 __train
调用self.trainer的train函数对模型进行训练，然后调用send_model_to_server，将本轮训练的参数更新结果和参与训练的数据条数发送给server。

### 5.5 FedAVGTrainer.py
#### 5.5.1 self.trainer
自定义Trainer类的一个对象。在main_fedavg.py中首先被定义（model_trainer），作为参数传给FedAvgAPI.FedML_FedAvg_distributed，用于初始化FedAVGTrainer。
#### 5.5.2 update_model
调用self.trainer.set_model_params，用全局模型参数更新本地模型参数。
#### 5.5.3 train
调用self.trainer.train进行本地训练，调用self.trainer.get_model_params获取本地模型参数，将其和参与训练的数据条数共同返回。

