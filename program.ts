import Network from './network';
import * as idx from './idx/idx-data';
import { scalarMultiply } from './vector';
import { fstat, writeFileSync } from 'fs';

function loadData() {
  return Promise.all([
    idx.loadBits('train-images-idx3-ubyte'),
    idx.loadBits('train-labels-idx1-ubyte'),
    idx.loadBits('t10k-images-idx3-ubyte'),
    idx.loadBits('t10k-labels-idx1-ubyte'),
  ]);
}

function getImg(data: idx.IdxTensor, index: number, size: number) {
  const img = data.data.slice(
    index * size * size,
    (index + 1) * size * size
  );
  return img;
}

function numberToArray(value: number) {
  if (value < 0 || value > 9) {
    throw new Error('Expected an integer between 0 and 9');
  }
  return Array(10).fill(0).map((_, i) => i === (value | 0) ? 1 : 0);
}

function max(n: number[]) {
  return n.reduce((prev, next) => next > prev ? next : prev, 0);
}

function min(n: number[]) {
  return n.reduce((prev, next) => next < prev ? next : prev, 0);
}

function normalize(arr: number[]) {
  const maximum = max(arr);
  const minimum = min(arr);

  return arr.map(v => (v - minimum) / (maximum - minimum));
}

const network = new Network([ 784, 30, 10 ]);

(async () => {
  const [
    trainImages,
    trainLabels,
    testImages,
    testLabels,
  ] = await loadData();

  const training: [ number[], number[] ][] = [];
  const test: [ number[], number ][] = [];

  // const cardinality = 1000;
  // const byteToFloat = 1 / 254;

  for (let i = 0; i < trainImages.shape[0]; i++) {
  // for (let i = 0; i < cardinality; i++) {
    training.push([
      normalize(Array.from(getImg(trainImages, i, 28))),
      numberToArray(trainLabels.data[i])
    ]);
  }

  for (let i = 0; i < testImages.shape[0]; i++) {
    test.push([
      normalize(Array.from(getImg(testImages, i, 28))),
      testLabels.data[i]
    ])
  }

  // network.sgd(training, 2, 1, 3.0, test);
  network.sgd(training, 30, 10, 3.0, test);
  
  writeFileSync('model.json', JSON.stringify(network.dump()));
  
})().catch(console.error);
