## 联邦学习简介
联邦学习(Federated  Learning) 是一种特殊的分布式学习, 又称联邦学习，联合学习，联盟学习. 是2016年由Google率先提出, 在满足用户隐私安全、数据安全和政府法规的要求下使用数据进行建模, 进而解决数据孤岛问题。

作为分布式的机器学习范式，联邦学习能够有效解决数据孤岛问题，让参与方在不共享数据的基础上联合建模，挖掘数据价值。

## 主流开源联邦学习框架
- FATE:FATE (Federated AI Technology Enabler) 是微众银行AI部门发起的开源项目，为联邦学习生态系统提供了可靠的安全计算框架。FATE项目使用多方安全计算 (MPC) 以及同态加密 (HE) 技术构建底层安全计算协议，以此支持不同种类的机器学习的安全计算，包括逻辑回归、基于树的算法、深度学习和迁移学习等。
- PaddleFL: PaddleFL是一个基于PaddlePaddle的开源联邦学习框架. 飞桨(PaddlePaddle)以百度多年的深度学习技术研究和业务应用为基础, 是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。
- Fedlearner:字节跳动联邦学习平台 Fedlearner 已经在电商、金融、教育等行业多个落地场景实际应用。
- FedML: 美国南加州大学USC联合MIT、Stanford、MSU、UW-Madison、UIUC以及腾讯、微众银行等众多高校与公司联合发布了FedML联邦学习开源框架。
- TFF: TensorFlow Federated (TFF) is an open-source framework for machine learning and other computations on decentralized data.


## 实践教程
- FedML联邦学习框架  [查看](doc/FedML.md)
- 基于FedML的物体检测（YOLO v3+COCO）[查看](doc/FedML_YOLOv3.md)
- 基于FedML的物体识别（ResNet18+cifar）[查看](doc/FedML_ResNet18.md)
