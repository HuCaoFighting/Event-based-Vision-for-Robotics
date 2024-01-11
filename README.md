# Awsome-Event-based-Vision-for-Robotics (Autonomous-Driving & Robotic-Grasping)

Collect some papers about event-based Autonomous Driving & Event-based Robotic-Grasping. 

If you find some overlooked papers, please open issues or pull requests (recommended).


## Papers

### Survey papers

- <a name="Gallego20tpami"></a>Gallego, G., Delbruck, T., Orchard, G., Bartolozzi, C., Taba, B., Censi, A., Leutenegger, S., Davison, A., Conradt, J., Daniilidis, K., Scaramuzza, D.,  
**_[Event-based Vision: A Survey](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)_**,  
IEEE Trans. Pattern Anal. Machine Intell. (TPAMI), 2020.
- <a name="Chen20msp"></a>Chen, G., Cao, H., Conradt, J., Tang, H., Rohrbein, F., Knoll, A.,  
[Event-Based Neuromorphic Vision for Autonomous Driving: A Paradigm Shift for Bio-Inspired Visual Sensing and Perception](https://doi.org/10.1109/MSP.2020.2985815),  
IEEE Signal Processing Magazine, 37(4):34-49, 2020.


### Datasets
#### Datasets about Autonomous Driving

- <a name="Cheng19cvprw"></a>Cheng, W., Luo, H., Yang, W., Yu, L., Chen, S., Li, W.,  
*[DET: A High-resolution DVS Dataset for Lane Extraction](http://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Cheng_DET_A_High-Resolution_DVS_Dataset_for_Lane_Extraction_CVPRW_2019_paper.pdf),*  
IEEE Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2019. [Project page](https://spritea.github.io/DET/).
- <a name="ncars_dataset"></a>[N-CARS Dataset](http://www.prophesee.ai/dataset-n-cars/): A large real-world event-based dataset for car classification.     [Sironi et al., CVPR 2018](#Sironi18cvpr).
- <a name="Zhu18mvsec"></a>Zhu, A., Thakur, D., Ozaslan, T., Pfrommer, B., Kumar, V., Daniilidis, K.,  
*[The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception](https://doi.org/10.1109/LRA.2018.2800793),*  
IEEE Robotics and Automation Letters (RA-L), 3(3):2032-2039, Feb. 2018. [PDF](https://arxiv.org/abs/1801.10202), [Dataset](https://daniilidis-group.github.io/mvsec/), [YouTube](https://youtu.be/9FaUvvzaHW8).
- <a name="Binas17icmlw"></a>Binas, J., Neil, D., Liu, S.-C., Delbruck, T.,  
*[DDD17: End-To-End DAVIS Driving Dataset](https://www.openreview.net/pdf?id=HkehpKVG-),*  
Int. Conf. Machine Learning, Workshop on Machine Learning for Autonomous Vehicles, 2017. [Dataset](http://sensors.ini.uzh.ch/databases.html)
- <a name="Hu20itsc"></a>Hu, Y., Binas, J., Neil, D., Liu, S.-C., Delbruck, T.,  
*[DDD20 End-to-End Event Camera Driving Dataset: Fusing Frames and Events with Deep Learning for Improved Steering Prediction](https://arxiv.org/abs/2005.08605)*,  
IEEE Intelligent Transportation Systems Conf. (ITSC), 2020. [Dataset](https://sites.google.com/view/davis-driving-dataset-2020/home), [More datasets](http://sensors.ini.uzh.ch/databases.html)
- <a name="Klenk21iros"></a>Klenk S., Chui, J., Demmel, N., Cremers, D.,  
*[TUM-VIE: The TUM Stereo Visual-Inertial Event Data Set](https://vision.in.tum.de/data/datasets/visual-inertial-event-dataset)*,  
IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2021.
- <a name="deTournemire20arxiv"></a>de Tournemire, P., Nitti, D., Perot, E., Migliore, D., Sironi, A.,  
*[A Large Scale Event-based Detection Dataset for Automotive](https://arxiv.org/abs/2001.08499)*,  
arXiv, 2020. [Code](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox), [News](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)
- <a name="1mpx_detection_dataset"></a> Perot, E., de Tournemire, P., Nitti, D., Masci, J., Sironi, A.,  [1Mpx Detection Dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/): [Learning to Detect Objects with a 1 Megapixel Event Camera. NeurIPS 2020](#Perot20nips).
- <a name="MGehrig21ral"></a>Gehrig, M., Aarents, W., Gehrig, D., Scaramuzza, D.,  
*[DSEC: A Stereo Event Camera Dataset for Driving Scenarios](https://doi.org/10.1109/LRA.2021.3068942)*,  
IEEE Robotics and Automation Letters (RA-L), 2021. [Dataset](http://rpg.ifi.uzh.ch/dsec.html), [PDF](http://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf), [Code](https://github.com/uzh-rpg/DSEC), [Video](https://youtu.be/W4yW78y4F7A).
- <a name="Chen20tits"></a>Chen, G., Wang, F., Li, W., Hong, L., Conradt, J., Chen, J., Zhang, Z., Lu, Y., Knoll, A., 
*[NeuroIV: Neuromorphic Vision Meets Intelligent Vehicle Towards Safe Driving With a New Database and Baseline Evaluations](https://doi.org/10.1109/TITS.2020.3022921)*,  
IEEE Trans. Intelligent Transportation Systems (TITS), 2020.

#### Datasets about Robotic Grasping
- <a name="E-Grasping"></a>Bin Li, Hu Cao*, Zhongnan Qu, Yingbai Hu, Zhenke Wang, Zichen Liang,  
*[Event-Based Robotic Grasping Detection With Neuromorphic Vision Sensor and Event-Grasping Dataset](https://www.frontiersin.org/articles/10.3389/fnbot.2020.00051/full)*,  
Frontiers in Neurorobotics, 2021. [Dataset](https://github.com/HuCaoFighting/DVS-GraspingDataSet).
- <a name="NeuroGrasp"></a>Hu Cao , Guang Chen , Zhijun Li , Yingbai Hu ,  Alois Knoll,  
*[NeuroGrasp: Multimodal Neural Network With Euler Region Regression for Neuromorphic Vision-Based Grasp Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9787342)*,  
 IEEE Transactions on Instrumentation and Measurement (TIM), 2021. [Dataset](https://github.com/HuCaoFighting/DVS-GraspingDataSet).
### Papers about Autonomous Driving
- <a name="Hidalgo20threedv"></a>Hidalgo-Carrió J., Gehrig D., Scaramuzza, D.,  
*[Learning Monocular Dense Depth from Events](https://arxiv.org/pdf/2010.08350.pdf)*,  
IEEE Int. Conf. on 3D Vision (3DV), 2020. [PDF](http://rpg.ifi.uzh.ch/docs/3DV20_Hidalgo.pdf), [YouTube](https://youtu.be/Ne1KyyXd3_A), [Code](https://github.com/uzh-rpg/rpg_e2depth), [Project Page](http://rpg.ifi.uzh.ch/e2depth).
- <a name="Gehrig21ral"></a>Gehrig, D., Rüegg, M., Gehrig, M., Hidalgo-Carrió J., Scaramuzza, D.,   
*[Combining Events and Frames Using Recurrent Asynchronous Multimodal Networks for Monocular Depth Prediction](https://doi.org/10.1109/LRA.2021.3060707)*,  
IEEE Robotics and Automation Letters (RA-L), 2021. [PDF](http://rpg.ifi.uzh.ch/docs/RAL21_Gehrig.pdf), [Code](http://rpg.ifi.uzh.ch/rpg_ramnet), [Project Page](http://rpg.ifi.uzh.ch/RAMNet.html).
- <a name="Alonso19cvprw"></a>Alonso I., Murillo A.,  
*[EV-SegNet: Semantic Segmentation for Event-based Cameras](http://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Alonso_EV-SegNet_Semantic_Segmentation_for_Event-Based_Cameras_CVPRW_2019_paper.pdf)*,  
IEEE Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2019. [PDF](https://arxiv.org/pdf/1811.12039.pdf). [Project page](https://github.com/Shathe/Ev-SegNet). [Video pitch](https://youtu.be/AuXN7y3bMqo)
- <a name="Wang21cvpr"></a>Wang, L., Chae, Y., Yoon, S.-H., Kim, T.-K., Yoon, K.-J.,  
*[EvDistill: Asynchronous Events To End-Task Learning via Bidirectional Reconstruction-Guided Cross-Modal Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_EvDistill_Asynchronous_Events_To_End-Task_Learning_via_Bidirectional_Reconstruction-Guided_Cross-Modal_CVPR_2021_paper.pdf)*,  
IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2021. [Code](https://github.com/addisonwang2013/evdistill).
- <a name="Cheng19fnbot"></a>Chen, G., Cao, H., Ye, C., Zhang, Z., Liu, X., Mo, X., Qu, Z., Conradt, J., Röhrbein, F., Knoll, A.,  
*[Multi-Cue Event Information Fusion for Pedestrian Detection With Neuromorphic Vision Sensors](https://doi.org/10.3389/fnbot.2019.00010)*,  
Front. Neurorobot. 13:10, 2019.
- <a name="Jiang19icra"></a> Jiang, Z., Xia, P., Huang, K., Stechele, W., Chen, G., Bing, Z., Knoll, A.,  
*[Mixed Frame-/Event-Driven Fast Pedestrian Detection](https://doi.org/10.1109/ICRA.2019.8793924)*,  
IEEE Int. Conf. Robotics and Automation (ICRA), 2019.
- <a name="Li19icme"></a>Li, J., Dong, S., Yu, Z., Tian, Y., Huang, T.,  
*[Event-Based Vision Enhanced: A Joint Detection Framework in Autonomous Driving](https://doi.org/10.1109/ICME.2019.00242),*  
IEEE Int. Conf. Multimedia and Expo (ICME), 2019.
- <a name="Perot20nips"></a>Perot, E., de Tournemire, P., Nitti, D., Masci, J., Sironi, A.,  
[Learning to Detect Objects with a 1 Megapixel Event Camera](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf),  
Advances in Neural Information Processing Systems 33 (NeurIPS), 2020. [1Mpx Detection Dataset](#1mpx_detection_dataset)
- <a name="Kim21iccv"></a>Kim, J., Bae, J., Park, G., Zhang, D., and Kim, Y.,  
*[N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_With_Event_Cameras_ICCV_2021_paper.pdf),*  
IEEE Int. Conf. Computer Vision (ICCV), 2021.
[Suppl. Mat.](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Kim_N-ImageNet_Towards_Robust_ICCV_2021_supplemental.pdf), [Github Page](https://github.com/82magnolia/n_imagenet), [N-ImageNet Dataset](https://www.dropbox.com/sh/47we7z2gff5barh/AADU4GyWnzLFzMzBDjLP00baa?dl=0).
- <a name="Chen20jsen"></a>Chen, G., Hong, L., Dong, J., Liu, P., Conradt, J., Knoll, A.,  
*[EDDD: Event-based drowsiness driving detection through facial motion analysis with neuromorphic vision sensor](https://doi.org/10.1109/JSEN.2020.2973049)*,  
IEEE Sensors Journal, 20(11):6170-6181, 2020.
- <a name="Maqueda18cvpr"></a>Maqueda, A.I., Loquercio, A., Gallego, G., Garcia, N., Scaramuzza, D.,  
*[Event-based Vision meets Deep Learning on Steering Prediction for Self-driving Cars](http://openaccess.thecvf.com/content_cvpr_2018/papers/Maqueda_Event-Based_Vision_Meets_CVPR_2018_paper.pdf)*,  
IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018. [PDF](http://rpg.ifi.uzh.ch/docs/CVPR18_Maqueda.pdf), [Poster](http://rpg.ifi.uzh.ch/docs/CVPR18_Maqueda_poster.pdf),  [YouTube](https://youtu.be/_r_bsjkJTHA).
- <a name="Yuhuang Hu"></a>Yuhuang Hu, Tobi Delbruck, Shih-Chii Liu,
*[Learning to Exploit Multiple Vision Modalities by Using Grafted Networks](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610086.pdf)*,  
ECCV 2020.
- <a name="Chen20tits"></a>Chen, G., Wang, F., Li, W., Hong, L., Conradt, J., Chen, J., Zhang, Z., Lu, Y., Knoll, A., 
*[NeuroIV: Neuromorphic Vision Meets Intelligent Vehicle Towards Safe Driving With a New Database and Baseline Evaluations](https://doi.org/10.1109/TITS.2020.3022921)*,  
IEEE Trans. Intelligent Transportation Systems (TITS), 2020.
- <a name="Cao"></a>Hu Cao, Guang Chen, Jiahao Xia, Genghang Zhuang, Alois Knoll, 
*[Fusion-based Feature Attention Gate Component for Vehicle Detection based on Event Camera](https://ieeexplore.ieee.org/abstract/document/9546775)*,  
IEEE Sensors Journal ( Volume: 21, Issue: 21, 01 November 2021).
- <a name="Shixiong Zhang"></a>Shixiong Zhang, Wenmin Wang, Honglei Li,Shenyong Zhang,
*[Fusing Asynchronous Event and Synchronization Frame Data with Multi-layer Feature Attention Guided Filtering](https://doi.org/10.21203/rs.3.rs-1770323/v1)*,  
Research Square (June 30th, 2022).
- <a name="Abhishek Tomy"></a>Abhishek Tomy, Anshul Paigwar, Khushdeep S. Mann, Alessandro Renzaglia, Christian Laugier,
*[Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions](https://ieeexplore.ieee.org/document/9812059)*,  
2022 International Conference on Robotics and Automation (ICRA) (23-27 May 2022).
- <a name="Abhishek Tomy"></a>Jianing Li, Xiao Wang, Lin Zhu, Jia Li, Tiejun Huang, Yonghong Tian,
*[Retinomorphic Object Detection in Asynchronous Visual Streams](https://www.aaai.org/AAAI22Papers/AAAI-1396.LiJ.pdf)*,  
Proceedings of the AAAI Conference on Artificial Intelligence, 36(2), 1332-1340.
- <a name="Munir, Farzeen"></a>Munir Farzeen, Azam Shoaib, Jeon Moongu, Lee Byung-Geun, Pedrycz Witold,
*[LDNet: End-to-End Lane Marking Detection Approach Using a Dynamic Vision Sensor](https://ieeexplore.ieee.org/document/9518365)*,  
IEEE Transactions on Intelligent Transportation Systems ( Volume: 23, Issue: 7, July 2022).
- <a name="Jianing Li"></a>Jianing Li, Jia Li, Lin Zhu, Xijie Xiang, Tiejun Huang, Yonghong Tian,
*[Asynchronous Spatio-Temporal Memory Network for Continuous Event-Based Object Detection](https://ieeexplore.ieee.org/document/9749022)*,  
IIEEE Transactions on Image Processing ( Volume: 31, 04 April 2022).
- <a name="Zhang, Jiaming"></a>Zhang, Jiaming and Yang, Kailun and Stiefelhagen, Rainer
*[ISSAFE: Improving Semantic Segmentation in Accidents by Fusing Event-based Data](https://ieeexplore.ieee.org/document/9636109)*,  
2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
- <a name="Zhaoning Sun"></a>Nico Messikommer, Daniel Gehrig, Davide Scaramuzza
*[ESS: Learning Event-based Semantic Segmentation from Still Images](https://https://arxiv.org/abs/2203.10016)*,  
European Conference on Computer Vision (ECCV), 2022  [Code](https://github.com/uzh-rpg/ess)
- <a name="Yansong Peng"></a>Yueyi Zhang, Zhiwei Xiong, Xiaoyan Sun, Feng Wu
*[GET: Group Event Transformer for Event-Based Vision](https://arxiv.org/abs/2310.02642)*,  
ICCV 2023  [Code](https://github.com/Peterande/GET-Group-Event-Transformer)
- <a name="Zexi Jia"></a>Kaichao You, Weihua He, Yang Tian, Yongxiang Feng, Yaoyuan Wang, Xu Jia, Yihang Lou,
Jingyi Zhang, Guoqi Li, and Ziyang Zhang
*[Event-Based Semantic Segmentation With Posterior Attention](https://ieeexplore.ieee.org/document/10058930)*,  
IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 32, 2023  [Code](https://github.com/zexiJia/EvSegFormer)
- <a name="Jiaming Zhang"></a>Jiaming Zhang, Kailun Yang, Rainer Stiefelhagen
*[Exploring Event-driven Dynamic Context for Accident Scene Segmentation](https://arxiv.org/abs/2112.05006)*,  
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 23, NO. 3, MARCH 2022  [Code](https://github.com/jamycheung/ISSAFE)

### Papers about Robotic Grasping
- <a name="E-Grasping"></a>Bin Li, Hu Cao*, Zhongnan Qu, Yingbai Hu, Zhenke Wang, Zichen Liang,  
*[Event-Based Robotic Grasping Detection With Neuromorphic Vision Sensor and Event-Grasping Dataset](https://www.frontiersin.org/articles/10.3389/fnbot.2020.00051/full)*,  
Frontiers in Neurorobotics, 2021. [Dataset](https://github.com/HuCaoFighting/DVS-GraspingDataSet).
- <a name="NeuroGrasp"></a>Hu Cao , Guang Chen , Zhijun Li , Yingbai Hu ,  Alois Knoll,  
*[NeuroGrasp: Multimodal Neural Network With Euler Region Regression for Neuromorphic Vision-Based Grasp Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9787342)*,  
 IEEE Transactions on Instrumentation and Measurement (TIM), 2021. [Dataset](https://github.com/HuCaoFighting/DVS-GraspingDataSet).

### LLM VLM for Robotics
- <a name=""></a>Kechun Xu, Shuqi Zhao, Zhongxiang Zhou, Zizhang Li, Huaijin Pi, Yifeng Zhu, Yue Wang, Rong Xiong,  
*[A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter](https://arxiv.org/abs/2302.12610)*,  
 ICRA 2023. [Code](https://github.com/xukechun/Vision-Language-Grasping).
- <a name=""></a>Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei,  
*[VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://arxiv.org/abs/2307.05973)*,  
  [Code](https://github.com/huangwl18/VoxPoser).
- <a name=""></a>Chao Tang, Dehao Huang, Wenqi Ge, Weiyu Liu, Hong Zhang,  
*[GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping](https://arxiv.org/abs/2307.13204)*,  
  [Code](https://www.catalyzex.com/paper/arxiv%3A2307.13204).
- <a name=""></a>Xinyu Chen, Jian Yang, Zonghan He, Haobin Yang, Qi Zhao, Yuhui Shi,  
*[QwenGrasp: A Usage of Large Vision-Language Model for Target-Oriented Grasping](https://arxiv.org/abs/2309.16426)*,  
  [Code](https://www.catalyzex.com/paper/arxiv%3A2309.16426).
- <a name=""></a>Reihaneh Mirjalili, Michael Krawez, Simone Silenzi, Yannik Blei, Wolfram Burgard,  
*[LAN-grasp: Using Large Language Models for Semantic Object Grasping](https://arxiv.org/abs/2310.05239)*,  
  [Code](https://www.catalyzex.com/paper/arxiv%3A2310.05239).
  
### arXiv papers
- **[DRFuser]** Multi-Modal Fusion for Sensorimotor Coordination in Steering Angle Prediction [[paper](https://arxiv.org/abs/2202.05500)] [[code](https://github.com/azamshoaib/drfuser)]
- **[CMX]** CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers [[paper](https://arxiv.org/pdf/2203.04838v2.pdf)] [[code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)]
-  Traffic Sign Detection With Event Cameras and DCNN [[paper](https://arxiv.org/abs/2207.13345)]
-  RGB-Event Fusion for Moving Object Detection in Autonomous Driving [[paper](https://arxiv.org/abs/2209.08323)]

