## Stock Prediction with TensorFlow & TypeScript

### Optional additions
* GPU for faster training (note: replace `tfjs-node` with `tfjs-node-gpu`)
	* [NVIDIA Cuda](https://developer.nvidia.com/cuda-downloads) & [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (for NVIDIA graphics cards)
	* [OpenCL](https://www.khronos.org/opencl/) (must build TensorFlow with [SYCL](https://www.khronos.org/sycl/) or [triSYCL](https://github.com/triSYCL/triSYCL) support)

### Closing
```shell
$ git clone https://github.com/matteopolak/stock-predict.git
```

### Installing
```shell
$ yarn install # or `npm install`
```

### Building (with TypeScript)
```shell
$ npx tsc
```

### Running
```shell
$ npm start -- <ticker>
```