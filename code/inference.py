import numpy as np


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

def predict_image(path_image, size_patch, path_model = "../dataset/pretrained_model.h5"):
    from model import create_tiramisu
    import keras

    img_input = Input(shape=(size_patch, size_patch, 3))
    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(1e-3),
                  metrics=["accuracy"],
                  sample_weight_mode='temporal')

    model.load_weights(path_model)

    full_img = load_image(path_image, img_sz)
    full_img_r, full_pred = waldo_predict(full_img)
    mask = prediction_mask(full_img_r, full_pred)
    #mask.save(os.path.join(args.output_path, 'output_' + str(i) + '.png'))
    return
