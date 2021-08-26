# DCGAN
In this repo I have implemented a paper titled by Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. It is about introducing a version of GAN that considers the training stability of the model.

And this is done by removing the fully connected layers in GANs and replacing the pooling layers with strided convolutional layers, and using batch normalization, Relu activation for the generator layers except for the output layer using tanh, and leakyRelu for the discriminator and the last layer is fed into a single sigmid output, also using dropout to avoid overfitting. The paper link for more details https://arxiv.org/pdf/1511.06434.pdf

For the implelmentation aiding by pytorch tutorials, I used the same settings illustrated in the paper cosidering the stability of the model, I have built the generator and discriminator networks in model.py file. And in the train.py file is the training of DCGAN network on MNIST dataset with Discriminator and Generator imported from model.py.

To get the samples results, I run the code in colab using GPU and i used the celebA dataset, which contains faces of celebrities. This can be found in the notebook file. the source refernce from https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dcgan_faces_tutorial.ipynb

Area of applications:
1. Generation of higher resolution images.
2. Vector arithmetic can be performed on images in Z space to get results, like man with glasses — normal man + normal woman = woman with glasses.
3. The use of vector arithmetic could decrease the amount of data needed for modelling complex image distributions.
