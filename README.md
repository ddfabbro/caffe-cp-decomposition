This is a heavy modification of the original [cp-decomposition](https://github.com/vadim-v-lebedev/cp-decomposition) algorithm that implements the method from their paper [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition](https://arxiv.org/pdf/1412.6553.pdf).

The most important modification is that now you can decompose more than one layer, according to given tensor rank (refer to  	`cpd_example.py`).

It also fixes some bugs and remove unnecessary complexity, making the usage and extensibility much simpler. In other words, it does what it is supposed to do: [CP decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) of convolutional layers. No more, no less.

## Requirements

- `pycaffe`
- `scikit-tensor`

## Limitations

With simplicity in mind, some limitations arises such as:

- Convolutional layer paramaters that are non-uniform (e.g `kernel_h`, `kernel_w`, `pad_h`, `pad_w`, `stride_h` and `stride_w` ) are not supported. However, you can easily modify the code to your needs.
- Multi branch networks are not supported.
