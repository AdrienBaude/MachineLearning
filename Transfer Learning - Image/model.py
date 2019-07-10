# Reference https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg19
from keras import backend as K


def preprocess_image(image_path):
    return vgg19.preprocess_input(
        np.expand_dims(img_to_array(load_img(image_path, target_size=(img_nrows, img_ncols))), axis=0))


def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


content_path = 'content.jpg'
style_path = 'style.jpg'

iterations = 601
total_variation_weight = 1.0
style_weight = 1.0
content_weight = 0.025

width, height = load_img(content_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

content_image = K.variable(preprocess_image(content_path))
style_image = K.variable(preprocess_image(style_path))
result_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([content_image, style_image, result_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

layer_features = outputs_dict['block5_conv2']
content_features = layer_features[0, :, :, :]
result_features = layer_features[2, :, :, :]

loss = K.variable(0.0)
loss += content_weight * content_loss(content_features, result_features)

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    result_features = layer_features[2, :, :, :]
    loss += (style_weight / len(feature_layers)) * style_loss(style_features, result_features)

loss += total_variation_weight * total_variation_loss(style_image)

grads = K.gradients(loss, result_image)
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([result_image], outputs)

evaluator = Evaluator()

#x = vgg19.preprocess_input(np.expand_dims(np.random.randint(256, size=(img_nrows, img_ncols, 3)).astype('float64'), axis=0))
x = preprocess_image(content_path)


for i in range(iterations):
    print('Start of iteration', i)
    x, _, _ = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    img = deprocess_image(x.copy())
    fname = 'generated_at_iteration_%d.jpg' % i
    if i % 50 == 0:
        save_img(fname, img)
