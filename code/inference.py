import numpy as np
from PIL import Image
from loader import resize_image


def patchify_test_image(img, size_patch):
    """Extracts patches from testing image."""
    h, w, _ = img.shape
    n_vertical_patch = int(h / size_patch)
    n_horizontal_patch = int(w / size_patch)
    list_patch = []
    for i in range(n_vertical_patch):
        for j in range(n_horizontal_patch):
            list_patch.append(img[i * size_patch : (i + 1) * size_patch, j * size_patch : (j + 1) * size_patch])
    return np.stack(list_patch)


def reconstruct_image_from_patches(img, np_patches, size_patch):
    """Reconstruct image from stack of patches N x size_patch x size_patch."""
    h, w, _ = img.shape
    n_vertical_patch = int(h / size_patch)
    n_horizontal_patch = int(w / size_patch)
    img_reconstructed = []
    p = 0
    for i in range(n_vertical_patch):
        row = []
        for j in range(n_horizontal_patch):
            row.append(np_patches[p])
            p += 1
        img_reconstructed.append(np.concatenate(row, axis=1))
    return np.concatenate(img_reconstructed, axis=0)


def load_andy_model(path_model = "../dataset/pretrained_model.h5"):
    """Loads model trained by Andy."""
    from tensorflow.keras.optimizers import RMSprop
    from keras.layers import Input
    from keras.models import Model

    from model import create_tiramisu

    print("Loading pretrained model ...")
    size_patch = 224
    img_input = Input(shape=(size_patch, size_patch, 3))

    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(1e-3),
                  metrics=["accuracy"],
                  sample_weight_mode='temporal')

    model.load_weights(path_model)

    return model


def display_predicitons_transparent(image, predictions):
    if np.max(image) <= 1:
        image *= 255
    layer1 = Image.fromarray((image).astype('uint8'))
    layer2 = Image.fromarray(
        np.concatenate(
            4*[np.expand_dims((225*(1-predictions)).astype('uint8'), axis=-1)],
            axis=-1))
    result = Image.new("RGBA", layer1.size)
    result = Image.alpha_composite(result, layer1.convert('RGBA'))
    return Image.alpha_composite(result, layer2)


def predict_with_pretrained_model(image, model, size_patch=224):
    print("Resizing input image ...")
    image_test_resized = resize_image(image, size_patch=size_patch)
    # Normalise intensities
    mu = 0.57729064767952254
    std = 0.31263486676782137
    image_test_resized = (image_test_resized - mu) / std

    print("Extracting patches ...")
    stack_patch = patchify_test_image(image_test_resized, size_patch)

    def reshape_pred(pred, size_patch):
        return pred.reshape(size_patch, size_patch, 2)[:, :, 1]

    print("Running predictions ...")
    pred_patch = model.predict(stack_patch, batch_size=6)
    np_pred_patch = np.stack([reshape_pred(pred, size_patch=size_patch) for pred in pred_patch])

    pred_full = reconstruct_image_from_patches(image_test_resized, np_pred_patch, 224)

    image_test_resized_rescaled = (image_test_resized * std + mu)
    pred_transparent = display_predicitons_transparent(image_test_resized_rescaled, pred_full)

    return image_test_resized_rescaled.astype(int), pred_full, pred_transparent
