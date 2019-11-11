# Deep Learning for Biology course materials / HSE 2019

This is a repository of course materials for the Deep Learning for Biology course. 

The course is taught Fall 2019 at Higher School of Economics (Moscow), Faculty of Computer Science, Master’s Programme 'Data Analysis in Biology and Medicine'.

## The contents
* Course [slides](slides)
* Course Jupyter [notebooks](notebooks) (using Tensorflow 2.0)


## Syllabus

### (10/09/2019) **[1. Artificial Intelligence: Current state and Overview](slides/%231.%20%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9%20%D0%B8%D0%BD%D1%82%D0%B5%D0%BB%D0%BB%D0%B5%D0%BA%D1%82%2C%20%D0%BE%D0%B1%D0%B7%D0%BE%D1%80%20%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D0%B8.pdf)**
- Short history
- Current results in Deep Learning
- Images and Video
- Speech and Sound
- Text and Language
- Robotic control
- ML for systems
- Problems with DL
- Other approaches to AI
- Knowledge and Representation
- Symbolic approaches
- Evolutionary computations and Swarm intelligence
- Hardware

**Video**:
- [part 1](https://www.youtube.com/watch?v=4lJ_JX_ig_Y)
- [part 2](https://www.youtube.com/watch?v=78-eNtdGd28)
- [part 3](https://www.youtube.com/watch?v=h3___u1rEwo)

### (17/09/2019) **[2. Introduction to Neural Networks](slides/%232.%20%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D0%B8.pdf)**
- Intro into NN: neuron, neural network, backpropagation, 
- Feed-forward NNs (FNN)
- Autoencoders (AE)

**Video**:
- [part 1](https://www.youtube.com/watch?v=tAeNQfvPDvI) (including the last topic from previous lecture, the Hardware)
- [part 2](https://www.youtube.com/watch?v=x5D2eKDFALM)


### (24/09/2019) **3. Tensorflow 2/Keras practice**
- Tensorflow 2 Intro (FFN: Binary classification, Multi-class classification, Regression)
  - [Jupyter notebook](notebooks/1%20-%20tf2_nn_intro.ipynb)
  - [Colab notebook](https://colab.research.google.com/drive/1YVrUhphhL_CoTY-aNCJGJLlXmeJW_d62)
- Autoencoders (shallow, deep, regularized/sparse, denoising)
  - [Jupyter notebook](notebooks/2%20-%20tf2_autoencoders.ipynb)
  - [Colab notebook](https://colab.research.google.com/drive/1sWx65xJJTP_xg2HnbwVjKj3YdNSXYQRy)

**Video**:
- [part 1](https://www.youtube.com/watch?v=AkTG7U2Wsxs)
- [part 2](https://www.youtube.com/watch?v=1v9OtgmrakM)
- [alternative recording](https://youtu.be/32O0rQth0WE)


### (01/10/2019) **[4. Convolutional NNs (CNN) and Image processing](slides/%234.%20Convolutional%20Neural%20Networks%20(CNNs).pdf)**
- What is CNN
- CNN for classification, CNN autoencoders, Saving and Loading models, How to use pretrained models in Tensorflow
  - [Jupyter notebook](notebooks/4%20-%20tf2_cnn.ipynb)
  - [Colab notebook](https://colab.research.google.com/drive/1vu-ZUHCVzPnvhohcPsPD96p4K5VcAVke)
  
**Video**:
- [part 1](https://youtu.be/6mgm2sDoHIQ)
- other parts are missing :(


### (08/10/2019) **[5. Real-life modern CNNs](slides/%235.%20%D0%9A%D0%B0%D0%BA%20%D1%83%D1%81%D1%82%D1%80%D0%BE%D0%B5%D0%BD%D1%8B%20%D1%80%D0%B5%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D1%81%D0%BE%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.pdf)**
- Activations, Regularization, Augmentation, Optimization etc
- Models: LeNet, AlexNet, VGG, GoogLeNet, Inception, ResNet, DenseNet, XCeption, NASNet
  
**Video**:
- [video 1, ~75%](https://www.youtube.com/watch?v=aWnrBejs79I)
- waiting for other recordings
  

### (15/10/2019) **Journal Club #1**
**Video**:
- [video](https://www.youtube.com/watch?v=ZXrlG742sT0)


### (29/10/2019) **Journal Club #2**
**Video**:
- [video](https://www.youtube.com/watch?v=Ze1QUAeYz-k)


### (05/11/2019) **[6. Guest Lecture: Artur Kadurin, GANs](slides/%236.%20Artur_Kadurin%20-%20GANs.pdf)**
- GANs
  
**Video**:
- [video](https://youtu.be/K7qIkl0eJPM)

# *--- NOT YET READY ---*
- Variational autoencoder
  - [Jupyter notebook](notebooks/3%20-%20tf2_vae.ipynb)
  - [Colab notebook](https://colab.research.google.com/drive/1rgUVgs7YnluhwaNDfDb6vGjXElb5QXn8)
- Keras practice. [Notebook: Visualizing CNNs: Saliency maps, grad-CAM, FCNs](notebooks/keras_cnn.ipynb)
- Keras practice. [Notebook: Playing with autoencoders](notebooks/playing_with_autoencoders.ipynb)

**[7. Transfer Learning](slides/%237.%20Transfer%20Learning.pdf)**
- Theory of Transfer Learning
- How to use pretrained models in Tensorflow
  - [Jupyter notebook](notebooks/4%20-%20tf2_cnn.ipynb)
  - [Colab notebook](https://colab.research.google.com/drive/1vu-ZUHCVzPnvhohcPsPD96p4K5VcAVke)  
- Keras practice. [Notebook: Transfer learning using VGG](notebooks/keras_cnn_transfer_learning.ipynb)

**[8. Advanced CNNs](slides/%238.%20%D0%9F%D1%80%D0%BE%D0%B4%D0%B2%D0%B8%D0%BD%D1%83%D1%82%D1%8B%D0%B5%20%D1%81%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D0%B5%20%D1%81%D0%B5%D1%82%D0%B8.pdf)**
- 1D, 3D, dilated convolutions
- Detection: R-CNN, Fast R-CNN, Faster R-CNN, YOLO
- Fully-convolutional CNNs (FCNs)
- Deconvolutional networks (Transposed convolution)
- Generative Adversarial Networks (GANs)
- Style Transfer
- [Notebook: FCN example, classification using only convolutions](notebooks/keras_cnn.ipynb)

**[9. Recurrent NNs (RNNs)](slides/%239.%20%D0%9E%D1%81%D0%BD%D0%BE%D0%B2%D1%8B%20%D1%80%D0%B5%D0%BA%D1%83%D1%80%D1%80%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D1%85%20%D1%81%D0%B5%D1%82%D0%B5%D0%B9.pdf)**
- RNN basics, Backpropagation through time
- Long short-term memory (LSTM)
- Advanced RNNs: Bidirectional RNNs, Multidimensional RNNs

**[10. Practice: Generating text using RNNs](slides/%2310.%20%D0%9F%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B0%20-%20%D0%93%D0%B5%D0%BD%D0%B5%D1%80%D0%B0%D1%86%D0%B8%D1%8F%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%B0%20%D1%81%20%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D0%B8.pdf)**
- Keras example. [Notebook: Text generation](notebooks/keras_text_generation.ipynb)

**[11. Practice: Text classification using RNNs](slides/%2311.%20%D0%9F%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B0%20-%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2.pdf)**
- Working with texts: vectorizing, one-hot encoding, word embeddings, word2vec etc
- Keras example: sentence-based classification using RNN/LSTM/BLSTM
- Keras example: sentence-based classification using 1D CNN
- Keras example: sentence-based classification using RNN+CNN
- [Notebook with examples](notebooks/keras_text_classification.ipynb)

**[12. Sequence Learning (seq2seq)](slides/%2312.%20Sequence%20Learning.pdf)**
- Multimodal Learning
- Seq2seq
- Encoder-Decoder
- Beam search
- Attention mechanisms, Visualizing attention, Hard and Soft attention, Self-Attention
- Augmented RNNs
- Connectionist Temporal Classification (CTC)
- Non-RNN Sequence Learning, problems with RNNs
- Convolutional Sequence Learning
- Self-Attention Neural Networks (SAN): Transformer Architecture
- Transformer: The next steps (Image Transformer, BERT, Universal Transformer)
