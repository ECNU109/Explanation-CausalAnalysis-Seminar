# Explanation-CausalAnalysis-Seminar
2022Spring Seminar: 1. Explanation Technique 2. Graph Explainability 3. Causal Analysis

时间：周二 9:00 am

地点：理科大楼A1716

---

## 论文报告安排

|  时间   | 报告人  | 主题  | 论文  |
|  ----  | ---- | ----  | ----  |
| 3.8      | 纪焘 | Intro. | [AAAI21 Tutorial] [Explaining Machine Learning Predictions: State-of-the-art, Challenges, and Opportunities](https://explainml-tutorial.github.io/aaai21) |
| 3.15      | 刘文炎	 |   |  |
| 3.15      | 屈稳稳 |   |  |
| 3.22      | 步一凡	 |  | |
| 3.22      | 胡梦琦 |  |  |
| 3.29      | 李放	 |   |  |
| 3.29      | 李文锋 |  |  |
| 4.12       | 潘金伟	 |  |  |
| 4.12       | 郑海坤 |  |  |
| 4.19      | 朱威	 |  |  |
| 4.19      | 岑黎彬 |  |  |
| 4.26      | 江宇辉	 |  |  |
| 4.26      | 孔维璟 |  |  |
| 5.10      | 毛炜	|  |  |
| 5.10      | 王鹏飞 |  |  |
| 5.17      | 岳文静	 |  |  |
| 5.17      | 钟博 |  |  |
| 5.24      | 郑焕然	 |  |  |
| 5.24      | 李靖东 |  |  |
| 5.31      | 纪焘 |  |  |
| 5.31      | 刘文炎 |  |  |
| 6.7      | 屈稳稳	 |  |  |
| 6.7      | 张启凡 |  |  |

## 论文主题列表
| 主题  | 论文  |
| ----  | ----  |
| ~~Intro.~~ | ~~[AAAI21 Tutorial] [Explaining Machine Learning Predictions: State-of-the-art, Challenges, and Opportunities](https://explainml-tutorial.github.io/aaai21)~~ |
|  LIME | [KDD16] ["Why Should I Trust You?": Explaining the Predictions of Any Classifier.](https://arxiv.org/abs/1602.04938) <br> [AAAI18] [Anchors: High-Precision Model-Agnostic Explanations.](http://sameersingh.org/files/papers/anchors-aaai18.pdf) |
|  SHAP | [NeurIPS17] [A Unified Approach to Interpreting Model Predictions.](https://arxiv.org/abs/1705.07874) <br> [ICML19] [Data Shapley: Equitable Valuation of Data for Machine Learning.](http://proceedings.mlr.press/v97/ghorbani19c.html) |
| Saliency Map | [ICLR ws14] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.](https://arxiv.org/abs/1312.6034) <br> [ICML17] [Learning Important Features Through Propagating Activation Differences.](https://arxiv.org/pdf/1704.02685.pdf) |
 | Saliency Map | [ICMLws18] [Noise-adding Methods of Saliency Map as Series of Higher Order Partial Derivative.](https://arxiv.org/pdf/1806.03000.pdf) <br> [ICML17] [Axiomatic Attribution for Deep Networks.](https://arxiv.org/abs/1703.01365) |
|  IF | [ICML17] [Understanding Black-box Predictions via Influence Functions.](https://arxiv.org/pdf/1703.04730.pdf) <br> [ICLR21] [Influence Functions in Deep Learning Are Fragile.](https://arxiv.org/pdf/2006.14651.pdf) |
|  |  |

---

## 阅读资源列表

> ⭐代表推荐阅读

## I. Explanation Technique

⭐[AAAI21 Tutorial] **Explaining Machine Learning Predictions: State-of-the-art, Challenges, and Opportunities** [Talk](https://explainml-tutorial.github.io/aaai21)

⭐[CVPR21 Tutorial] **Interpretable Machine Learning for Computer Vision** [Talk](https://interpretablevision.github.io)

Slides [Part1](https://interpretablevision.github.io/slide/cvpr21_samek.pdf) [Part2](https://interpretablevision.github.io/slide/cvpr21_rudin.pdf) [Part3](https://interpretablevision.github.io/slide/cvpr21_morcos.pdf) [Part4](https://interpretablevision.github.io/slide/cvpr21_zhou.pdf)

### 方法介绍（LIME、SHAP、Saliency Map、IF、Counterfactual）

1. ⭐**Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.** *Cynthia Rudin.* Nature Machine Intelligence 2019. [paper](https://arxiv.org/pdf/1811.10154.pdf)
2. ⭐**The Mythos of Model Interpretability.** *Zachary C. Lipton.* Machine Learning 2018. [paper](https://arxiv.org/abs/1606.03490)
3. ⭐[LIME] **"Why Should I Trust You?": Explaining the Predictions of Any Classifier.** *MT Ribeiro, S Singh, C Guestrin.* KDD 2016. [paper](https://arxiv.org/abs/1602.04938)
4. ⭐[Anchors] **Anchors: High-Precision Model-Agnostic Explanations.** *MT Ribeiro, S Singh, C Guestrin.* AAAI 2018. [paper](http://sameersingh.org/files/papers/anchors-aaai18.pdf)
5. ⭐[SHAP] **A Unified Approach to Interpreting Model Predictions.** *Scott Lundberg, Su-In Lee.* NeurIPS 2017. [paper](https://arxiv.org/abs/1705.07874)
6. ⭐[SHAP] **Data Shapley: Equitable Valuation of Data for Machine Learning.** *Amirata Ghorbani, James Zou.* ICML 2019. [paper](http://proceedings.mlr.press/v97/ghorbani19c.html)
7. ⭐[Saliency Map] **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.** *Karen Simonyan, Andrea Vedaldi, Andrew Zisserman.* ICLR ws 2014. [paper](https://arxiv.org/abs/1312.6034)
8. ⭐[Saliency Map-DeepLIFT] **Learning Important Features Through Propagating Activation Differences.** *Avanti Shrikumar, Peyton Greenside, Anshul Kundaje*. ICML 2017. [paper](https://arxiv.org/pdf/1704.02685.pdf)
9. ⭐[SmoothGrad] **Noise-adding Methods of Saliency Map as Series of Higher Order Partial Derivative.** *Junghoon Seo, Jeongyeol Choe, Jamyoung Koo, Seunghyeon Jeon, Beomsu Kim, Taegyun Jeon.* ICML ws 2018. [paper](https://arxiv.org/pdf/1806.03000.pdf)
10. ⭐[Saliency Map-积分梯度] **Axiomatic Attribution for Deep Networks.** *Mukund Sundararajan, Ankur Taly, Qiqi Yan*. ICML 2017. [paper](https://arxiv.org/abs/1703.01365)
11. ⭐[Saliency Map-RISE] **RISE: randomized input sampling for explanation of black-box models.** BMVC 2018. [paper](https://arxiv.org/abs/1806.07421)
12. ⭐[IF] **Understanding Black-box Predictions via Influence Functions.** ICML 2017. [paper](https://arxiv.org/pdf/1703.04730.pdf)
13. [IF] **Influence Functions in Deep Learning Are Fragile.** ICLR 2021. [paper](https://arxiv.org/pdf/2006.14651.pdf)
14. [IF] **Representer Point Selection for Explaining Deep Neural Networks.** NeurIPS 2018. [paper](https://arxiv.org/abs/1811.09720)
15. [IF] **Estimating Training Data Influence by Tracing Gradient Descent.** NeurIPS 2020. [paper](https://arxiv.org/abs/2002.08484)
16. ⭐[Counterfactual] **Counterfactual Explanations for Machine Learning: A Review.** [paper](https://arxiv.org/abs/2010.10596)
17. [Counterfactual] **Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR**. [paper](https://arxiv.org/abs/1711.00399)
18. [Counterfactual] **Model-agnostic counterfactual explanations for consequential decisions**. *AISTATS 2020.* [paper](https://arxiv.org/abs/1905.11190)
19. [Counterfactual] **Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers.** NeurIPS 2019. [paper](https://arxiv.org/pdf/1912.03277.pdf)
20. [Distill] **Faithful and Customizable Explanations of Black Box Models.** AIES 2019. [paper](https://cs.stanford.edu/people/jure/pubs/explanations-aies19.pdf)
21. ⭐[Distill] **“How do I fool you?": Manipulating User Trust via Misleading Black Box Explanations.** AIES 2020. [paper](https://www.aies-conference.com/2020/wp-content/papers/182.pdf)
22. [Distill] **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML 2019. [paper](https://arxiv.org/pdf/1905.11946.pdf)
23. ⭐[Counterfactual] **Beyond Individualized Recourse: Interpretable and Interactive Summaries of Actionable Recourses.** NeurIPS 2020. [paper](https://arxiv.org/pdf/2009.07165)
24. [Dissection] **Network Dissection: Quantifying Interpretability of Deep Visual Representations.** CVPR 2017. [paper](http://netdissect.csail.mit.edu)
25. ⭐[TCAV] **Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).** ICML 2018. [paper](https://arxiv.org/pdf/1711.11279.pdf)
26. [TCAV] **Regression Concept Vectors for Bidirectional Explanations in Histopathology.** MICCAI 2018. [paper](https://arxiv.org/abs/1904.04520)
27. [TCAV] **Towards Automatic Concept-based Explanations.** NeurIPS 2019. [paper](https://arxiv.org/pdf/1902.03129.pdf)

### 解释性方法评测

1. ⭐**On Human Predictions with Explanations and Predictions of Machine Learning Models: A Case Study on Deception Detection.** FAT 2019. [paper](https://arxiv.org/pdf/1811.07901.pdf)
2. ⭐**Teaching Categories to Human Learners with Visual Explanations.** CVPR 2018. [paper](https://arxiv.org/pdf/1802.06924.pdf)
3. ⭐**Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?.** ACL 2020. [paper](https://arxiv.org/pdf/2005.01831)
4. **Visualizing Deep Networks by Optimizing with Integrated Gradients.** AAAI 2020. [paper](https://arxiv.org/pdf/1905.00954.pdf)
5. **Towards A Rigorous Science of Interpretable Machine Learning.** arxiv 2017. [paper](https://arxiv.org/abs/1702.08608)
6. **Manipulating and Measuring Model Interpretability.** CHI 2021. [paper](https://arxiv.org/pdf/1902.03129.pdf)
7. **Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications.** NAACL 2019. [paper](https://arxiv.org/pdf/1905.00563.pdf)


## II. Graph Explainability

### 网络结构解释
1. ⭐**Gnnexplainer: Generating explanations for graph neural networks**. *Ying Rex, Bourgeois Dylan, You Jiaxuan, Zitnik Marinka, Leskovec Jure*. NeurIPS 2019. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7138248/) [code](https://github.com/RexYing/gnn-model-explainer)
2. ⭐**PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks**. *Vu Minh, Thai My T.*. NeurIPS 2020. [paper](https://arxiv.org/pdf/2010.05788.pdf)
3. ⭐**Xgnn: Towards model-level explanations of graph neural networks**. *Yuan Hao, Tang Jiliang, Hu Xia, Ji Shuiwang*. KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403085)
4. ⭐**When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods**. *Faber Lukas, K. Moghaddam Amin, Wattenhofer Roger*. KDD 2021. [paper](https://dl.acm.org/doi/10.1145/3447548.3467283)
5. ⭐**Generative Causal Explanations for Graph Neural Networks**. *Lin Wanyu, Lan Hao, Li Baochun*. ICML 2021. [paper](https://arxiv.org/pdf/2104.06643.pdf)
6. ⭐**Higher-order explanations of graph neural networks via relevant walks**. *Schnake Thomas, Eberle Oliver, Lederer Jonas, Nakajima Shinichi, Schütt Kristof T, Müller Klaus-Robert, Montavon Grégoire*. TPAMI 2021. [paper](https://arxiv.org/pdf/2006.03589.pdf)
7. ⭐**Discovering Invariant Rationales for Graph Neural Networks.** ICLR 2022. [paper](https://arxiv.org/abs/2201.12872)
8. ⭐**Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking.** ICLR 2021. [paper](https://arxiv.org/abs/2010.00577)
9. ⭐**Multi-objective Explanations of GNN Predictions.** ICDM 2021. [paper](https://arxiv.org/abs/2111.14651)
10. ⭐**Graph Information Bottleneck for Subgraph Recognition.** ICLR 2021. [paper](https://arxiv.org/pdf/2010.05563.pdf)
11. **ProtGNN: Towards Self-Explaining Graph Neural Networks.** AAAI 2022. [paper](https://arxiv.org/abs/2112.00911)
12. **Deconfounding to Explanation Evaluation in Graph Neural Networks.** ICLR 2022. [paper](https://arxiv.org/abs/2201.08802)
13. **GNN-SubNet: disease subnetwork detection with explainable Graph Neural Networks.** BioRxiv 2022. [paper](https://www.biorxiv.org/content/10.1101/2022.01.12.475995v1)
14. **Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning.** The Webconf 22. [paper](https://arxiv.org/abs/2202.08816)
15. **EGNN: Constructing explainable graph neural networks via knowledge distillation. KBS 2022.** [paper](https://www.sciencedirect.com/science/article/pii/S0950705122001289?via%3Dihub)
16. **Reinforcement Learning Enhanced Explainer for Graph Neural Networks.** NeurIPS 2021. [paper](http://recmind.cn/papers/explainer_nips21.pdf)
17. **Towards Multi-Grained Explainability for Graph Neural Networks.** NeurIPS 2021. [paper](http://staff.ustc.edu.cn/~hexn/papers/nips21-explain-gnn.pdf)
18. **Robust Counterfactual Explanations on Graph Neural Networks.** NeurIPS 2021. [paper](https://arxiv.org/abs/2107.04086)
19. **On Explainability of Graph Neural Networks via Subgraph Explorations.** ICML 2021. [paper](https://arxiv.org/abs/2102.05152)
20. **Automated Graph Representation Learning with Hyperparameter Importance Explanation.** ICML 2021. [paper](http://proceedings.mlr.press/v139/wang21f/wang21f.pdf)


### 图解释性的下游应用
### [1-4]计算病理  [5-7]社交网络  [8-10]推荐系统

1. ⭐**Quantifying Explainers of Graph Neural Networks in Computational Pathology**. *Jaume Guillaume, Pati Pushpak, Bozorgtabar Behzad, Foncubierta Antonio, Anniciello Anna Maria, Feroce Florinda, Rau Tilman, Thiran Jean-Philippe, Gabrani Maria, Goksel Orcun*. CVPR 2021. [paper](https://arxiv.org/pdf/2011.12646.pdf)
2. ⭐**Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network**. *Wu Haoran, Chen Wei, Xu Shuang, Xu Bo*. NAACL 2021. [paper](https://aclanthology.org/2021.naacl-main.156.pdf)
3. ⭐**Counterfactual Graphs for Explainable Classification of Brain Networks**. *Abrate Carlo, Bonchi Francesco*. KDD 2021. [paper](https://arxiv.org/pdf/2106.08640.pdf)
4. **Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity.** ICML 2021**.** [paper](https://arxiv.org/abs/2105.04854)
5. ⭐**GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media**. *Lu, Yi-Ju and Li, Cheng-Te*. ACL 2020. [paper](https://arxiv.org/pdf/2004.11648.pdf)
6. ⭐**HENIN: Learning Heterogeneous Neural Interaction Networks for Explainable Cyberbullying Detection on Social Media**. *Chen, Hsin-Yu and Li, Cheng-Te*. EMNLP 2020. [paper](https://www.aclweb.org/anthology/2020.emnlp-main.200/)
7. ⭐**SCARLET: Explainable Attention based Graph Neural Network for Fake News spreader prediction.** PAKDD 21 **.** [paper](https://arxiv.org/abs/2102.04627)
8. ⭐**MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems.** *Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, Jie Tang*. KDD 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408) [code](https://github.com/huangtinglin/MixGCF)
9. ⭐**Structured Graph Convolutional Networks with Stochastic Masks for Recommender Systems.** *Chen, Huiyuan and Wang, Lan and Lin, Yusan and Yeh, Chin-Chia Michael and Wang, Fei and Yang, Hao*. SIGIR 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3462868) [code](https://github.com/THUDM/cogdl/blob/master)
10. ⭐**Sequential Recommendation with Graph Convolutional Networks.** *Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, Depeng Jin, Yong Li*. SIGIR 2021. [paper](https://arxiv.org/abs/2106.14226) [code](https://github.com/THUDM/cogdl/blob/master)

## III.  Causal Inference

1. ⭐[**Survey] Toward Causal Representation Learning**, IEEE 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9363924)
2. ⭐[**Survey] A Survey of Learning Causality with Data: Problems and Methods**, ACM 2020. [paper](https://arxiv.org/abs/1809.09337)

### 因果推断方法介绍

1. ⭐**Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning.** PNAS 2019. [paper](https://arxiv.org/abs/1706.03461)
2. ⭐**Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments.** NeurIPS 2019. [paper](https://arxiv.org/abs/1905.10176)
3. ⭐**VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments.** ICLR 2021. [paper](https://arxiv.org/abs/2103.07861)  [code](https://github.com/lushleaf/varying-coefficient-net-with-functional-tr)
4. ⭐**Learning Counterfactual Representations for Estimating Individual Dose-Response Curves.** AAAI 2020. [paper](https://arxiv.org/abs/1902.00981) [code](https://github.com/d909b/drnet)
5. ⭐**Estimating the Effects of Continuous-valued Interventions using Generative Adversarial Networks.** NeurIPS 2020. [paper](https://arxiv.org/abs/2002.12326) [code](https://github.com/ioanabica/SCIGAN)
6. **Learning Individual Causal Effects from Networked Observational Data.** WSDM 2020. [paper](https://arxiv.org/abs/1906.03485) [code](https://github.com/rguo12/network-deconfounder-wsdm20)
7. ⭐[Temporal data] **Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders.** ICML 2020. [paper](https://arxiv.org/abs/1902.00450) [code](https://github.com/ioanabica/Time-Series-Deconfounder)
8. ⭐[Temporal data] **Estimating Counterfactual Treatment Outcomes over Time through Adversarially Balanced Representations.** ICLR 2020. [paper](https://openreview.net/pdf?id=BJg866NFvB) [code](https://github.com/ioanabica/Counterfactual-Recurrent-Network)
9. ⭐**Deep Structural Causal Models for Tractable Counterfactual Inference.** NeurIPS 2020. [paper](https://arxiv.org/abs/2006.06485) [code](https://github.com/biomedia-mira/deepscm)
10. **Representation Learning for Treatment Effect Estimation from Observational Data.** NeurIPS 2019. [paper](https://papers.nips.cc/paper/7529-representation-learning-for-treatment-effect-estimation-from-observational-data.pdf)
11. **Differentiable Causal Discovery Under Unmeasured Confounding.** AISTATS 2021. [paper](https://arxiv.org/abs/2010.06978)

### 因果推断的下游应用

1. ⭐**Counterfactual Data Augmentation for Neural Machine Translation.** ACL 2021. [paper](https://www.aclweb.org/anthology/2021.naacl-main.18/) [code](https://github.com/xxxiaol/GCI)
2. ⭐**Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis.** NAACL 2021. [paper](https://arxiv.org/abs/2104.09420) [code](https://github.com/xxxiaol/GCI)
3. ⭐**Causal Effects of Linguistic Properties.** NAACL, 2021. [paper](https://arxiv.org/abs/2010.12919)
4. **Sketch and Customize: A Counterfactual Story Generator.** AAAI, 2021. [paper](https://arxiv.org/abs/2104.00929)
5. **Counterfactual Generator: A Weakly-Supervised Method for Named Entity Recognition.** EMNLP 2020. [paper](https://github.com/xijiz/cfgen/blob/master/docs/cfgen.pdf) [code](https://github.com/xijiz/cfgen)
6. ⭐**The Deconfounded Recommender: A Causal Inference Approach to Recommendation.** *arXiv*, 2019. [paper](https://arxiv.org/abs/1808.06581)  [code](https://github.com/blei-lab/deconfounder_tutorial)
7. **Using Embeddings to Estimate Peer Influence on Social Networks. NeurlPS 2021. [paper](https://why21.causalai.net/papers/WHY21_41.pdf)**
8. **Unsupervised Causal Binary Concepts Discovery with VAE for Black-box Model Explanation. NeurlPS 2021. [paper](https://why21.causalai.net/papers/WHY21_3.pdf)**
9. ⭐**Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models.** ACL 2021. [paper](https://arxiv.org/abs/2106.06087)
