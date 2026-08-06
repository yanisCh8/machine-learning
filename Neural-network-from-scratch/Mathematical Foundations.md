# Mathematical Foundations

## 1. Activation Functions

### Sigmoid Drawback
The sigmoid function $\sigma(z)$ slows down learning because the gradient becomes extremely flat at high or low magnitudes, leading to vanishing gradients.

### ReLU (Rectified Linear Unit)
Used in the hidden layer to introduce non-linearity. Without it, the network is just a linear regression model.
$$ReLU(a) = \max(0, a)$$

### Softmax
Used in the output layer to convert raw scores into a probability distribution across the 10 digit classes.

$$A^{[2]}_i = \frac{e^{Z^{[2]}_i}}{\sum_{j=1}^{10} e^{Z^{[2]}_j}}$$

---

## 2. Forward Propagation

We process $m$ training examples simultaneously using matrix operations:

* **Input Layer ($A^{[0]}$):** $$A^{[0]} = X \quad \text{(Shape: } 784 \times m\text{)}$$
* **Hidden Layer ($Z^{[1]}, A^{[1]}$):**
  $$Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]}$$
  $$A^{[1]} = ReLU(Z^{[1]})$$
* **Output Layer ($Z^{[2]}, A^{[2]}$):**
  $$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$
  $$A^{[2]} = \text{Softmax}(Z^{[2]})$$

---

## 3. Backward Propagation

We calculate gradients starting from the output layer and moving backward:

* **Output Layer Gradients:**
  $$dZ^{[2]} = A^{[2]} - Y$$
  $$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
  $$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

* **Hidden Layer Gradients:**
  $$dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g'^{[1]}(Z^{[1]})$$
  $$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$
  $$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

---

## 4. Update Parameters

Finally, we update our weights and biases using a learning rate ($\alpha$):

$$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
$$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$
$$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
$$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
