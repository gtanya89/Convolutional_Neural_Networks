## Handwriting Recognition on MNIST Dataset using Convolutional Neural Networks

Model Architecture is as follows:

Input image size 28 * 28 

1. Layer 1
    1. Convolutional Layer 5 * 5 * 1 32 filters; padding=SAME
    2. ReLU activation
    3. Max pooling layer with 2 * 2 size and stride=2
2. Layer 2
    1. Convolutional Layer 5 * 5 * 32 64 filters; padding=SAME
    2. ReLU activation
    3. Max pooling layer with 2 * 2 size and stride=2
3. Fully Connected Layer
    1. 1024 neurons
    2. Flatten the output volume from Layer 2
4. Apply dropout keep_prob=0.5
5. Softmax

6. Compute cross-entropy cost and minimize cost by updating weights and biases using Adam Optimizer.

7. Run using mini batch gradient descent.
    
