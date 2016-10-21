# Deep Learning Papers Reading Roadmap

>If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

>Here is a reading roadmap of Deep Learning papers!

The roadmap is constructed in accordance with the following four guidelines:

- From outline to detail
- From old to state-of-the-art
- from generic to specific areas
- focus on state-of-the-art

You will find many papers that are quite new but really worth reading.

I would continue adding papers to this roadmap.


---------------------------------------

# 1 Deep Learning History and Basics

## 1.0 Book

**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book in preparation. Draft chapters available at http://www. iro. umontreal. ca/∼ bengioy/dlbook (2015).[[pdf]](https://github.com/HFTrader/DeepLearningBook) **(Deep Learning Bible, you can read this book while reading following papers.)** :star::star::star::star::star:

## 1.1 Survey

**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(Three Giants' Survey)** :star::star::star::star::star:

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554.[[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(Deep Learning Eve)** :star::star::star:

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(Milestone, Show the promise of deep learning)** :star::star::star:

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014).[[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015).[[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

## 1.4 Speech Recognition Evolution

**[8]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97.[[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(Breakthrough in speech recognition)**:star::star::star::star:

**[9]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN)**:star::star::star:

**[10]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014.[[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:

**[11]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015).[[pdf]](http://arxiv.org/pdf/1507.06947) **(Google Speech Recognition System)** :star::star::star:

**[12]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015).[[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(Baidu Speech Recognition System)** :star::star::star::star:

**[13]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016).[[pdf]](https://arxiv.org/pdf/1610.05256v1) **(State-of-the-art in speech recognition, Microsoft)** :star::star::star::star:

>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to image and speech recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

#2 Deep Learning Method

## 2.1 Model

**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012).[[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** :star::star::star:

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958.[[pdf]](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) :star::star::star:

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015).[[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)** :star::star::star::star:

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016).[[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)** :star::star::star::star:

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**  :star::star::star:

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

## 2.2 Optimization

**[20]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147.[[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** :star::star:

**[21]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014).[[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)** :star::star::star:

**[22]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016).[[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[23]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015).[[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[24]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[25]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013.[[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[26]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013).[[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[27]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[28]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[29]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[29]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[30]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[31]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[32]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[33]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[34]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[35]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:

## 2.5 Neural Turing Machine

**[36]** Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) **(Basic Prototype of Future Computer)** :star::star::star::star::star:

**[37]** Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf) :star::star::star:

**[38]** Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014).[[pdf]](http://arxiv.org/pdf/1410.3916) :star::star::star:


**[39]** Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) :star::star::star::star:

**[40]** Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) :star::star::star::star:

**[41]** Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) **(Milestone,combine above papers' ideas)** :star::star::star::star::star:

## 2.6 Deep Reinforcement Learning

**[42]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[43]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf) **(Milestone)** :star::star::star::star::star:

**[44]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)**  :star::star::star::star:

**[45]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** :star::star::star::star::star:

**[46]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** :star::star::star::star:

**[47]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[48]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[49]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)** :star::star::star::star::star:

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

**[50]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(A Tutorial)** :star::star::star:

**[51]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(A brief discussion about lifelong learning)**  :star::star::star:

**[52]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(Godfather's Work)** :star::star::star::star:

**[53]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)** :star::star::star:

**[54]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** :star::star::star:

**[55]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(Outstanding Work, A novel idea)** :star::star::star::star::star:


## 2.8 One Shot Deep Learning

**[56]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[57]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[58]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)** :star::star::star::star:

**[59]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[60]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[61]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[62]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) :star::star::star:

**[63]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[64]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[65]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[66]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) :star::star::star:

**[67]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) :star::star:

## 3.2 Object Detection

**[68]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[69]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** :star::star::star::star::star:

**[70]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** :star::star::star::star:

**[71]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

**[72]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5638-analysis-of-variational-bayesian-latent-dirichlet-allocation-weaker-sparsity-than-map.pdf) :star::star::star::star:

**[73]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical)** :star::star::star::star::star:

**[74]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

## 3.3 Visual Tracking

**[75]** Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) **(First Paper to do visual tracking using Deep Learning,DLT Tracker)** :star::star::star:

**[76]** Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](http://arxiv.org/pdf/1501.04587) **(SO-DLT)** :star::star::star::star:

**[77]** Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) **(FCNT)** :star::star::star::star:

**[78]** Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](http://arxiv.org/pdf/1604.01802) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)** :star::star::star::star:

**[79]** Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](https://arxiv.org/pdf/1606.09549) **(SiameseFC,New state-of-the-art for real-time object tracking)** :star::star::star::star:

**[80]** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) **(C-COT)** :star::star::star::star:

**[81]** Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](https://arxiv.org/pdf/1608.07242) **(VOT2016 Winner,TCNN)** :star::star::star::star:




## 3.4 Image/Video Caption
**[75]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf)

**[76]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[77]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014.[[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[78]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[79]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[80]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[81]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[82]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[83]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014.[[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[84]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## 3.5 Machine Translation

## 3.6 Audio

## 3.7 Art

## 3.8 Game

## 3.9 Robotics

## 3.10 Other Frontiers












