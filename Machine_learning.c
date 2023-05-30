#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Single-layer perceptron
void perceptron_train(double inputs[][2], double outputs[], double weights[], int num_inputs, int num_epochs, double learning_rate) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < num_inputs; i++) {
            double predicted_output = sigmoid(inputs[i][0] * weights[0] + inputs[i][1] * weights[1]);
            double error = outputs[i] - predicted_output;
            total_error += fabs(error);

            // Update weights
            weights[0] += learning_rate * error * sigmoid_derivative(predicted_output) * inputs[i][0];
            weights[1] += learning_rate * error * sigmoid_derivative(predicted_output) * inputs[i][1];
        }

        // Print the total error for this epoch
        printf("Epoch %d - Total Error: %.4f\n", epoch, total_error);

        // If the total error is below a threshold, stop training
        if (total_error < 0.001) {
            printf("Training completed.\n");
            break;
        }
    }
}

int main(void) {
    // Training data
    double training_inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double training_outputs[4] = {0, 0, 0, 1};

    // Initialize weights randomly
    double weights[2] = {0.5, 0.5};

    // Train the perceptron
    int num_epochs = 10000;
    double learning_rate = 0.1;
    perceptron_train(training_inputs, training_outputs, weights, 4, num_epochs, learning_rate);

    // Test the perceptron
    printf("Input\t\tOutput\n");
    for (int i = 0; i < 4; i++) {
        double predicted_output = sigmoid(training_inputs[i][0] * weights[0] + training_inputs[i][1] * weights[1]);
        printf("%d %d\t\t%.2f\n", (int)training_inputs[i][0], (int)training_inputs[i][1], predicted_output);
    }

    return 0;
}
