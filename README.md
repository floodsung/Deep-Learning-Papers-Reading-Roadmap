<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Deep Learning Papers Reading Roadmap](#deep-learning-papers-reading-roadmap)
* [1 Deep Learning History and Basics](#1-deep-learning-history-and-basics)
  * [1.0 Book](#10-book)
  * [1.1 Survey](#11-survey)
  * [1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)](#12-deep-belief-networkdbnmilestone-of-deep-learning-eve)
  * [1.3 ImageNet Evolution（Deep Learning broke out from here）](#13-imagenet-evolutiondeep-learning-broke-out-from-here)
  * [1.4 Speech Recognition Evolution](#14-speech-recognition-evolution)
  * [2.1 Model](#21-model)
  * [2.2 Optimization](#22-optimization)
  * [2.3 Unsupervised Learning / Deep Generative Model](#23-unsupervised-learning--deep-generative-model)
  * [2.4 RNN / Sequence-to-Sequence Model](#24-rnn--sequence-to-sequence-model)
  * [2.5 Neural Turing Machine](#25-neural-turing-machine)
  * [2.6 Deep Reinforcement Learning](#26-deep-reinforcement-learning)
  * [2.7 Deep Transfer Learning / Lifelong Learning / especially for RL](#27-deep-transfer-learning--lifelong-learning--especially-for-rl)
  * [2.8 One Shot Deep Learning](#28-one-shot-deep-learning)
* [3 Applications](#3-applications)
  * [3.1 NLP(Natural Language Processing)](#31-nlpnatural-language-processing)
  * [3.2 Object Detection](#32-object-detection)
  * [3.3 Visual Tracking](#33-visual-tracking)
  * [3.4 Image Caption](#34-image-caption)
  * [3.5 Machine Translation](#35-machine-translation)
  * [3.6 Robotics](#36-robotics)
  * [3.7 Art](#37-art)
  * [3.8 Object Segmentation](#38-object-segmentation)

<!-- tocstop -->

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

**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.0%20Book/0%20Deep%20learning.pdf) **(Deep Learning Bible, you can read this book while reading following papers.)** :star::star::star::star::star:

## 1.1 Survey

**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.1%20Survey/1%20Deep%20learning.pdf) **(Three Giants' Survey)** :star::star::star::star::star:

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.2%20Deep%20Belief%20Network%28DBN%29%28Milestone%20of%20Deep%20Learning%20Eve%29/2%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)**(Deep Learning Eve)** :star::star::star:

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.2%20Deep%20Belief%20Network%28DBN%29%28Milestone%20of%20Deep%20Learning%20Eve%29/3%20Reducing%20the%20dimensionality%20of%20data%20with%20neural%20networks.pdf) **(Milestone, Show the promise of deep learning)** :star::star::star:

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.3%20ImageNet%20Evolution%EF%BC%88Deep%20Learning%20broke%20out%20from%20here%EF%BC%89/4%20Imagenet%20classification%20with%20deep%20convolutional%20neural%20networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.3%20ImageNet%20Evolution%EF%BC%88Deep%20Learning%20broke%20out%20from%20here%EF%BC%89/5%20Very%20deep%20convolutional%20networks%20for%20large-scale%20image%20recognition.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.3%20ImageNet%20Evolution%EF%BC%88Deep%20Learning%20broke%20out%20from%20here%EF%BC%89/6%20Going%20deeper%20with%20convolutions.pdf) **(GoogLeNet)** :star::star::star:

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.3%20ImageNet%20Evolution%EF%BC%88Deep%20Learning%20broke%20out%20from%20here%EF%BC%89/7%20Deep%20residual%20learning%20for%20image%20recognition.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

## 1.4 Speech Recognition Evolution

**[8]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/8%20Deep%20neural%20networks%20for%20acoustic%20modeling%20in%20speech%20recognition%3A%20The%20shared%20views%20of%20four%20research%20groups.pdf) **(Breakthrough in speech recognition)**:star::star::star::star:

**[9]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/9%20Speech%20recognition%20with%20deep%20recurrent%20neural%20networks.pdf) **(RNN)**:star::star::star:

**[10]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/10%20Towards%20End-To-End%20Speech%20Recognition%20with%20Recurrent%20Neural%20Networks.pdf):star::star::star:

**[11]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/11%20Fast%20and%20accurate%20recurrent%20neural%20network%20acoustic%20models%20for%20speech%20recognition.pdf) **(Google Speech Recognition System)** :star::star::star:

**[12]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/12%20Deep%20speech%202%3A%20End-to-end%20speech%20recognition%20in%20english%20and%20mandarin.pdf) **(Baidu Speech Recognition System)** :star::star::star::star:

**[13]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](pdfs/1%20Deep%20Learning%20History%20and%20Basics/1.4%20Speech%20Recognition%20Evolution/13%20Achieving%20Human%20Parity%20in%20Conversational%20Speech%20Recognition.pdf) **(State-of-the-art in speech recognition, Microsoft)** :star::star::star::star:

>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to image and speech recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

#2 Deep Learning Method

## 2.1 Model

**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/14%20Improving%20neural%20networks%20by%20preventing%20co-adaptation%20of%20feature%20detectors.pdf) **(Dropout)** :star::star::star:

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/15%20Dropout%3A%20a%20simple%20way%20to%20prevent%20neural%20networks%20from%20overfitting.pdf) :star::star::star:

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/16%20Batch%20normalization%3A%20Accelerating%20deep%20network%20training%20by%20reducing%20internal%20covariate%20shift.pdf) **(An outstanding Work in 2015)** :star::star::star::star:

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/17%20Layer%20normalization.pdf) **(Update of Batch Normalization)** :star::star::star::star:

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/18%20Binarized%20Neural%20Networks%3A%20Training%20Neural%20Networks%20with%20Weights%20and%20Activations%20Constrained%20to%2B%201%20or%E2%88%921.pdf) **(New Model,Fast)**  :star::star::star:

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.1%20Model/19%20Decoupled%20neural%20interfaces%20using%20synthetic%20gradients.pdf) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

**[20]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)** :star::star::star:

**[21]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)** :star::star::star:

## 2.2 Optimization

**[22]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.2%20Optimization/22%20On%20the%20importance%20of%20initialization%20and%20momentum%20in%20deep%20learning.pdf) **(Momentum optimizer)** :star::star:

**[23]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.2%20Optimization/23%20Adam%3A%20A%20method%20for%20stochastic%20optimization.pdf) **(Maybe used most often currently)** :star::star::star:

**[24]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.2%20Optimization/24%20Learning%20to%20learn%20by%20gradient%20descent%20by%20gradient%20descent.pdf) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[25]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.2%20Optimization/25%20Deep%20compression%3A%20Compressing%20deep%20neural%20network%20with%20pruning%2C%20trained%20quantization%20and%20huffman%20coding.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[26]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.2%20Optimization/26%20SqueezeNet%3A%20AlexNet-level%20accuracy%20with%2050x%20fewer%20parameters%20and%3C%201MB%20model%20size.pdf) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[27]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/27%20Building%20high-level%20features%20using%20large%20scale%20unsupervised%20learning.pdf) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/28%20Auto-encoding%20variational%20bayes.pdf) **(VAE)** :star::star::star::star:

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/29%20Generative%20adversarial%20nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/30%20Unsupervised%20representation%20learning%20with%20deep%20convolutional%20generative%20adversarial%20networks.pdf) **(DCGAN)** :star::star::star::star:

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/31%20DRAW%3A%20A%20recurrent%20neural%20network%20for%20image%20generation.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.3%20Unsupervised%20Learning%20%26%20Deep%20Generative%20Model/32%20Pixel%20recurrent%20neural%20networks.pdf) **(PixelRNN)** :star::star::star::star:

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[34]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.4%20RNN%20%26%20Sequence-to-Sequence%20Model/34%20Generating%20sequences%20with%20recurrent%20neural%20networks.pdf) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[35]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.4%20RNN%20%26%20Sequence-to-Sequence%20Model/35%20Learning%20phrase%20representations%20using%20RNN%20encoder-decoder%20for%20statistical%20machine%20translation.pdf) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[36]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.4%20RNN%20%26%20Sequence-to-Sequence%20Model/36%20Sequence%20to%20sequence%20learning%20with%20neural%20networks.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[37]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.4%20RNN%20%26%20Sequence-to-Sequence%20Model/37%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.pdf) :star::star::star::star:

**[38]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.4%20RNN%20%26%20Sequence-to-Sequence%20Model/38%20A%20neural%20conversational%20model.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:

## 2.5 Neural Turing Machine

**[39]** Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/39%20Neural%20turing%20machines.pdf) **(Basic Prototype of Future Computer)** :star::star::star::star::star:

**[40]** Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/40%20Reinforcement%20learning%20neural%20Turing%20machines.pdf) :star::star::star:

**[41]** Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/41%20Memory%20networks.pdf) :star::star::star:


**[42]** Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/42%20End-to-end%20memory%20networks.pdf) :star::star::star::star:

**[43]** Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/43%20Pointer%20networks.pdf) :star::star::star::star:

**[44]** Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.5%20Neural%20Turing%20Machine/44%20Hybrid%20computing%20using%20a%20neural%20network%20with%20dynamic%20external%20memory.pdf) **(Milestone,combine above papers' ideas)** :star::star::star::star::star:

## 2.6 Deep Reinforcement Learning

**[45]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/45%20Playing%20atari%20with%20deep%20reinforcement%20learning.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[46]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/46%20Human-level%20control%20through%20deep%20reinforcement%20learning.pdf) **(Milestone)** :star::star::star::star::star:

**[47]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/47%20Dueling%20network%20architectures%20for%20deep%20reinforcement%20learning.pdf) **(ICLR best paper,great idea)**  :star::star::star::star:

**[48]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/48%20Asynchronous%20methods%20for%20deep%20reinforcement%20learning.pdf) **(State-of-the-art method)** :star::star::star::star::star:

**[49]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/49%20Continuous%20control%20with%20deep%20reinforcement%20learning.pdf) **(DDPG)** :star::star::star::star:

**[50]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/50%20Continuous%20Deep%20Q-Learning%20with%20Model-based%20Acceleration.pdf) **(NAF)** :star::star::star::star:

**[51]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/51%20Trust%20region%20policy%20optimization.pdf) **(TRPO)** :star::star::star::star:

**[52]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.6%20Deep%20Reinforcement%20Learning/52%20Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search.pdf) **(AlphaGo)** :star::star::star::star::star:

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

**[53]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/53%20Deep%20Learning%20of%20Representations%20for%20Unsupervised%20and%20Transfer%20Learning.pdf) **(A Tutorial)** :star::star::star:

**[54]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/54%20Lifelong%20Machine%20Learning%20Systems%3A%20Beyond%20Learning%20Algorithms.pdf) **(A brief discussion about lifelong learning)**  :star::star::star:

**[55]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/55%20Distilling%20the%20knowledge%20in%20a%20neural%20network.pdf) **(Godfather's Work)** :star::star::star::star:

**[56]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/56%20Policy%20distillation.pdf) **(RL domain)** :star::star::star:

**[57]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/57%20Actor-mimic%3A%20Deep%20multitask%20and%20transfer%20reinforcement%20learning.pdf) **(RL domain)** :star::star::star:

**[58]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.7%20Deep%20Transfer%20Learning%20%26%20Lifelong%20Learning%20%26%20especially%20for%20RL/58%20Progressive%20neural%20networks.pdf) **(Outstanding Work, A novel idea)** :star::star::star::star::star:


## 2.8 One Shot Deep Learning

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.8%20One%20Shot%20Deep%20Learning/59%20Human-level%20concept%20learning%20through%20probabilistic%20program%20induction.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[60]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.8%20One%20Shot%20Deep%20Learning/60%20Siamese%20Neural%20Networks%20for%20One-shot%20Image%20Recognition.pdf) :star::star::star:

**[61]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.8%20One%20Shot%20Deep%20Learning/61%20One-shot%20Learning%20with%20Memory-Augmented%20Neural%20Networks.pdf) **(A basic step to one shot learning)** :star::star::star::star:

**[62]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.8%20One%20Shot%20Deep%20Learning/62%20Matching%20Networks%20for%20One%20Shot%20Learning.pdf) :star::star::star:

**[63]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](pdfs/2%20Deep%20Learning%20Method/2.8%20One%20Shot%20Deep%20Learning/63%20Low-shot%20visual%20object%20recognition.pdf) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/1%20Joint%20Learning%20of%20Words%20and%20Meaning%20Representations%20for%20Open-Text%20Semantic%20Parsing.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/2%20Distributed%20representations%20of%20words%20and%20phrases%20and%20their%20compositionality.pdf) **(word2vec)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/3%20%E2%80%9CSequence%20to%20sequence%20learning%20with%20neural%20networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/4%20%E2%80%9CAsk%20Me%20Anything%3A%20Dynamic%20Memory%20Networks%20for%20Natural%20Language%20Processing.pdf) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/5%20Character-Aware%20Neural%20Language%20Models.pdf) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/6%20Towards%20AI-Complete%20Question%20Answering%3A%20A%20Set%20of%20Prerequisite%20Toy%20Tasks.pdf) **(bAbI tasks)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/7%20Teaching%20Machines%20to%20Read%20and%20Comprehend.pdf) **(CNN/DailyMail cloze style questions)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/8%20Very%20Deep%20Convolutional%20Networks%20for%20Natural%20Language%20Processing.pdf) **(state-of-the-art in text classification)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](pdfs/3%20Applications/3.1%20NLP%28Natural%20Language%20Processing%29/9%20Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.pdf) **(slightly worse than state-of-the-art, but a lot faster)** :star::star::star:

## 3.2 Object Detection

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/1%20Deep%20neural%20networks%20for%20object%20detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/2%20Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation.pdf) **(RCNN)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/3%20Spatial%20pyramid%20pooling%20in%20deep%20convolutional%20networks%20for%20visual%20recognition.pdf) **(SPPNet)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/4%20Fast%20r-cnn.pdf) :star::star::star::star:

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/5%20Faster%20R-CNN%3A%20Towards%20real-time%20object%20detection%20with%20region%20proposal%20networks.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/6%20You%20only%20look%20once%3A%20Unified%2C%20real-time%20object%20detection.pdf) **(YOLO,Oustanding Work, really practical)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/7%20SSD%3A%20Single%20Shot%20MultiBox%20Detector.pdf) :star::star::star:

**[8]** Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409) :star::star::star::star:

**[9]** He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](pdfs/3%20Applications/3.2%20Object%20Detection/9%20Mask%20R-CNN.pdf) :star::star::star::star:
## 3.3 Visual Tracking

**[1]** Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/1%20Learning%20a%20deep%20compact%20image%20representation%20for%20visual%20tracking.pdf) **(First Paper to do visual tracking using Deep Learning,DLT Tracker)** :star::star::star:

**[2]** Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/2%20Transferring%20rich%20feature%20hierarchies%20for%20robust%20visual%20tracking.pdf) **(SO-DLT)** :star::star::star::star:

**[3]** Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/3%20Visual%20tracking%20with%20fully%20convolutional%20networks.pdf) **(FCNT)** :star::star::star::star:

**[4]** Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/4%20Learning%20to%20Track%20at%20100%20FPS%20with%20Deep%20Regression%20Networks.pdf) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)** :star::star::star::star:

**[5]** Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/5%20Fully-Convolutional%20Siamese%20Networks%20for%20Object%20Tracking.pdf) **(SiameseFC,New state-of-the-art for real-time object tracking)** :star::star::star::star:

**[6]** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/6%20Beyond%20Correlation%20Filters%3A%20Learning%20Continuous%20Convolution%20Operators%20for%20Visual%20Tracking.pdf) **(C-COT)** :star::star::star::star:

**[7]** Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](pdfs/3%20Applications/3.3%20Visual%20Tracking/7%20Modeling%20and%20Propagating%20CNNs%20in%20a%20Tree%20Structure%20for%20Visual%20Tracking.pdf) **(VOT2016 Winner,TCNN)** :star::star::star::star:

## 3.4 Image Caption
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/1%20Every%20picture%20tells%20a%20story%3A%20Generating%20sentences%20from%20images.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/2%20Baby%20talk%3A%20Understanding%20and%20generating%20image%20descriptions.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/3%20Show%20and%20tell%3A%20A%20neural%20image%20caption%20generator.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/4%20Long-term%20recurrent%20convolutional%20networks%20for%20visual%20recognition%20and%20description.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/5%20Deep%20visual-semantic%20alignments%20for%20generating%20image%20descriptions.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/6%20Deep%20fragment%20embeddings%20for%20bidirectional%20image%20sentence%20mapping.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/7%20From%20captions%20to%20visual%20concepts%20and%20back.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/8%20Learning%20a%20recurrent%20visual%20representation%20for%20image%20caption%20generation.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/9%20Deep%20captioning%20with%20multimodal%20recurrent%20neural%20networks%20%28m-rnn%29.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](pdfs/3%20Applications/3.4%20Image%20Caption/10%20Show%2C%20attend%20and%20tell%3A%20Neural%20image%20caption%20generation%20with%20visual%20attention.pdf):star::star::star::star::star:

## 3.5 Machine Translation

> Some milestone papers are listed in RNN / Seq-to-Seq topic.

**[1]** Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/1%20Addressing%20the%20rare%20word%20problem%20in%20neural%20machine%20translation.pdf) :star::star::star::star:


**[2]** Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/2%20Neural%20Machine%20Translation%20of%20Rare%20Words%20with%20Subword%20Units.pdf):star::star::star:

**[3]** Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/3%20Effective%20approaches%20to%20attention-based%20neural%20machine%20translation.pdf) :star::star::star::star:

**[4]** Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/4%20A%20Character-Level%20Decoder%20without%20Explicit%20Segmentation%20for%20Neural%20Machine%20Translation.pdf):star::star:

**[5]** Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/5%20Fully%20Character-Level%20Neural%20Machine%20Translation%20without%20Explicit%20Segmentation.pdf):star::star::star::star::star:

**[6]** Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](pdfs/3%20Applications/3.5%20Machine%20Translation/6%20Google%27s%20Neural%20Machine%20Translation%20System%3A%20Bridging%20the%20Gap%20between%20Human%20and%20Machine%20Translation.pdf) **(Milestone)** :star::star::star::star:

## 3.6 Robotics

**[1]** Koutník, Jan, et al. "**Evolving large-scale neural networks for vision-based reinforcement learning**." Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013. [[pdf]](pdfs/3%20Applications/3.6%20Robotics/1%20Evolving%20large-scale%20neural%20networks%20for%20vision-based%20reinforcement%20learning.pdf) :star::star::star:

**[2]** Levine, Sergey, et al. "**End-to-end training of deep visuomotor policies**." Journal of Machine Learning Research 17.39 (2016): 1-40. [[pdf]](pdfs/3%20Applications/3.6%20Robotics/2%20End-to-end%20training%20of%20deep%20visuomotor%20policies.pdf) :star::star::star::star::star:

**[3]** Pinto, Lerrel, and Abhinav Gupta. "**Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**." arXiv preprint arXiv:1509.06825 (2015). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/3%20Supersizing%20self-supervision%3A%20Learning%20to%20grasp%20from%2050k%20tries%20and%20700%20robot%20hours.pdf) :star::star::star:

**[4]** Levine, Sergey, et al. "**Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**." arXiv preprint arXiv:1603.02199 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/4%20Learning%20Hand-Eye%20Coordination%20for%20Robotic%20Grasping%20with%20Deep%20Learning%20and%20Large-Scale%20Data%20Collection.pdf) :star::star::star::star:

**[5]** Zhu, Yuke, et al. "**Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**." arXiv preprint arXiv:1609.05143 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/5%20Target-driven%20Visual%20Navigation%20in%20Indoor%20Scenes%20using%20Deep%20Reinforcement%20Learning.pdf) :star::star::star::star:

**[6]** Yahya, Ali, et al. "**Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**." arXiv preprint arXiv:1610.00673 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/6%20Collective%20Robot%20Reinforcement%20Learning%20with%20Distributed%20Asynchronous%20Guided%20Policy%20Search.pdf) :star::star::star::star:

**[7]** Gu, Shixiang, et al. "**Deep Reinforcement Learning for Robotic Manipulation**." arXiv preprint arXiv:1610.00633 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/7%20Deep%20Reinforcement%20Learning%20for%20Robotic%20Manipulation.pdf) :star::star::star::star:

**[8]** A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell."**Sim-to-Real Robot Learning from Pixels with Progressive Nets**." arXiv preprint arXiv:1610.04286 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/8%20Sim-to-Real%20Robot%20Learning%20from%20Pixels%20with%20Progressive%20Nets.pdf) :star::star::star::star:

**[9]** Mirowski, Piotr, et al. "**Learning to navigate in complex environments**." arXiv preprint arXiv:1611.03673 (2016). [[pdf]](pdfs/3%20Applications/3.6%20Robotics/9%20Learning%20to%20navigate%20in%20complex%20environments.pdf) :star::star::star::star:

## 3.7 Art

**[1]** Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**
:star::star::star::star:

**[2]** Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](pdfs/3%20Applications/3.7%20Art/2%20A%20neural%20algorithm%20of%20artistic%20style.pdf) **(Outstanding Work, most successful method currently)** :star::star::star::star::star:

**[3]** Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](pdfs/3%20Applications/3.7%20Art/3%20Generative%20Visual%20Manipulation%20on%20the%20Natural%20Image%20Manifold.pdf) **(iGAN)** :star::star::star::star:

**[4]** Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/4%20Semantic%20Style%20Transfer%20and%20Turning%20Two-Bit%20Doodles%20into%20Fine%20Artworks.pdf) **(Neural Doodle)** :star::star::star::star:

**[5]** Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/5%20Colorful%20Image%20Colorization.pdf) :star::star::star::star:

**[6]** Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/6%20Perceptual%20losses%20for%20real-time%20style%20transfer%20and%20super-resolution.pdf) :star::star::star::star:

**[7]** Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/7%20A%20learned%20representation%20for%20artistic%20style.pdf) :star::star::star::star:

**[8]** Gatys, Leon and Ecker, et al."**Controlling Perceptual Factors in Neural Style Transfer**." arXiv preprint arXiv:1611.07865 (2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/8%20Controlling%20Perceptual%20Factors%20in%20Neural%20Style%20Transfer.pdf) **(control style transfer over spatial location,colour information and across spatial scale)**:star::star::star::star:

**[9]** Ulyanov, Dmitry and Lebedev, Vadim, et al. "**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**." arXiv preprint arXiv:1603.03417(2016). [[pdf]](pdfs/3%20Applications/3.7%20Art/9%20Texture%20Networks%3A%20Feed-forward%20Synthesis%20of%20Textures%20and%20Stylized%20Images.pdf) **(texture generation and style transfer)** :star::star::star::star:


## 3.8 Object Segmentation

**[1]** J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](pdfs/3%20Applications/3.8%20Object%20Segmentation/1%20Fully%20convolutional%20networks%20for%20semantic%20segmentation.pdf) :star::star::star::star::star:

**[2]** L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](pdfs/3%20Applications/3.8%20Object%20Segmentation/2%20Semantic%20image%20segmentation%20with%20deep%20convolutional%20nets%20and%20fully%20connected%20crfs.pdf) :star::star::star::star::star:

**[3]** Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](pdfs/3%20Applications/3.8%20Object%20Segmentation/3%20Learning%20to%20segment%20object%20candidates..pdf) :star::star::star::star:

**[4]** Dai, J., He, K., Sun, J. "**Instance-aware semantic segmentation via multi-task network cascades**." in CVPR. 2016 [[pdf]](pdfs/3%20Applications/3.8%20Object%20Segmentation/4%20Instance-aware%20semantic%20segmentation%20via%20multi-task%20network%20cascades.pdf) :star::star::star:

**[5]** Dai, J., He, K., Sun, J. "**Instance-sensitive Fully Convolutional Networks**." arXiv preprint arXiv:1603.08678 (2016). [[pdf]](pdfs/3%20Applications/3.8%20Object%20Segmentation/5%20Instance-sensitive%20Fully%20Convolutional%20Networks.pdf) :star::star::star:
