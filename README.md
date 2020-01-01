# MNIST neural network TypeScript

Neural network written in TypeScript. This is a rewrite of the [original neural network](http://neuralnetworksanddeeplearning.com/) written in Python, by [Michael Nielsen](http://michaelnielsen.org/).

It is generally a bad, bad idea to have the CPU train a neural network. It's especially bad if vectorization isn't used while computing anything. This is the case with using TypeScript to train a neural network.

Why is it a bad idea? It's insanely slow. At least 10 folds slower than Michael Nielsen's original implementation.

The whole purpose of this rewrite was for me to go through the process of writing a neural network. This is entirely educational, and likely writing a neural network from scratch is not going to be of any real world productive value.

You can view the demo here: https://gifted-swanson-103ffa.netlify.com/