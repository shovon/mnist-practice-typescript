import Matrix from './matrix';

/**
 * Combines two iterables into a single iterable. At each iteration, the value
 * from the first and second iterables are placed inside a two-item array
 * (a.k.a. tuple).
 * 
 * The first iterable to end iteration ends the entire iteration.
 * @param it1 The first iterable.
 * @param it2 The second iterable.
 */
function * zip<T, V>(it1: Iterable<T>, it2: Iterable<V>): Iterable<[T, V]> {
  // TODO: unit test this.

  const iterable1 = it1[Symbol.iterator]();
  const iterable2 = it2[Symbol.iterator]();

  let next1 = iterable1.next();
  let next2 = iterable2.next();
  while (!next1.done && !next2.done) {
    yield [ next1.value, next2.value ];
    next1 = iterable1.next();
    next2 = iterable2.next();
  }
}

/**
 * Destructively shuffles the given array, using the Fisher–Yates shuffle.
 * @param arr The array to shuffle.
 */
function shuffle<T>(arr: T[]) {
  // TODO: unit test this.

  for (let i = arr.length - 1; i >= 1; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[j];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**
 * Computes the sigmoid of the given number.
 * @param z The number to compute its sigmoid.
 */
function sigmoid(z: number) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Computes the derivative of the signmoid, of the given number.
 * @param z The number to compute the derivative of the sigmoid.
 */
function sigmoidPrime(z: number) {
  return sigmoid(z) * (1 - sigmoid(z));
}

/**
 * Maps through each items in the iterable, returning a new iterable with the
 * supplied function applied.
 * @param it An iterable.
 * @param mapper The function to apply to each value of the iterable.
 */
function * map<T, V>(it: Iterable<T>, mapper: (t: T) => V): Iterable<V> {
  for (const value of it) {
    yield mapper(value);
  }
}

/**
 * Generates a new iterator, that will iterate from the start value, to the end
 * value, in steps of the step value.
 * @param start The number to start with
 * @param end The number to end with
 * @param step The steps to iterate by
 */
function * range(start: number, end: number, step: number = 1) {
  // TODO: unit test this.
  for (let i = start; i < end; i += step) {
    yield i;
  }
}

/**
 * Iterates through the iterable, yielding each item, along with their index.
 * @param iterable The iterable to iterate.
 */
function * iterate<T>(iterable: Iterable<T>): Iterable<[T, number]> {
  let i = 0;
  for (const item of iterable) {
    yield [ item, i ];
    i++;
  }
}

/**
 * The callback to the reduce function to compute the sum of all items.
 * @param result The sum of all numbers in the array
 * @param next The next result in the array
 */
function sumFn(result: number, next: number) {
  // TODO: unit test this.
  return result + next;
}

function indexMax(array: number[]) {
  return array.reduce((max, next, i) => next > array[max] ? i : max, 0);
}

type Model = {
  weights: number[][][],
  biases: number[][][]
}

/**
 * The neural network class.
 */
export default class Network {

  private biases: Matrix[]
  private weights: Matrix[]

  /**
   * Initializes a new instance of the neural network.
   * @param sizes The input sizes of each layer of the neural network.
   */
  constructor(private sizes: number[]) {
    this.biases = sizes
      .slice(1)
      .map(y => Matrix.random(y, 1));
    this.weights = [
      ...map(
        zip(sizes.slice(0, -1), sizes.slice(1)),
        ([x, y]) => Matrix.random(y, x)
      )
    ];
  }

  static load(model: Model) {
    if (model.weights.length !== model.biases.length) {
      throw new Error('The number of weights does not match the number of biases');
    }

    const activationSize = model.weights[0][0].length;

    const sizes = [ activationSize ];

    for (const [ weight, i ] of iterate(model.weights)) {
      const rowCount = weight.length;
      if (rowCount !== model.biases[i].length) {
        throw new Error('Dimensionality of bias does not match the number of rows in the weight');
      }
      sizes.push(rowCount);
    }

    const network = new Network(sizes);
    network.biases = model.biases.map(bias => new Matrix(bias));
    network.weights = model.weights.map(weight => new Matrix(weight));
    return network;
  }

  /**
   * Provide input to the network, to get an output.
   * @param activation The input activation.
   */
  feedForward(activation: number[]) {
    let a = Matrix.fromVector(activation);
    for (const [ b, w ] of zip(this.biases, this.weights)) {
      a = w.multiply(a).add(b).apply(sigmoid);
    }
    return a;
  }

  sgd(trainingData: [ number[], number[] ][], epochs: number, miniBatchSize, eta, testData: [ number[], number ][] | null = null) {
    console.log('Training started');
    let nTest: number | null = testData ? testData.length : null;
    const n = trainingData.length;
    for (const j of range(0, epochs)) {
      shuffle(trainingData);
      const miniBatches = [
        ...map(
          range(0, n, miniBatchSize),
          k => trainingData.slice(k, k + miniBatchSize)
        )
      ];
      for (const miniBatch of miniBatches) {
        this.updateMiniBatch(miniBatch, eta);
      }
      if (testData) {
        console.log(`Epoch ${j}: ${this.evaluate(testData)} / ${nTest}`)
      } else {
        console.log(`Epoch ${j} complete`)
      }
    }
  }

  updateMiniBatch(miniBatch: [ number[], number[] ][], eta) {
    let nablaB = this.biases.map<Matrix>(b => Matrix.zeros(b.rowCount, b.colCount));
    let nablaW = this.weights.map<Matrix>(w => Matrix.zeros(w.rowCount, w.colCount));

    for (const [ x, y ] of miniBatch) {
      const [
        deltaNablaB,
        deltaNablaW
      ] = this.backprop(x, y);
      nablaB = [...map<[Matrix, Matrix], Matrix>(
        zip<Matrix, Matrix>(nablaB, deltaNablaB),
        ([ nb, dnb ]) => nb.add(dnb)
      )];
      nablaW = [...map<[Matrix, Matrix], Matrix>(
        zip<Matrix, Matrix>(nablaW, deltaNablaW),
        ([ nw, dnw ]) => nw.add(dnw)
      )];
    }

    this.weights = [
      ...map(
        zip(this.weights, nablaW),
        ([ w, nw ]) => w.subtract(nw.scalarMultiply(eta/miniBatch.length))
      )
    ];
    this.biases = [
      ...map(
        zip(this.biases, nablaB),
        ([ b, nb ]) => b.subtract(nb.scalarMultiply(eta/miniBatch.length))
      )
    ];
  }

  backprop(x: number[], y: number[]): [ Matrix[], Matrix[] ] {
    let nablaB = this.biases.map<Matrix>(b => Matrix.zeros(b.rowCount, b.colCount));
    let nablaW = this.weights.map<Matrix>(w => Matrix.zeros(w.rowCount, w.colCount));

    let activation = Matrix.fromVector(x);
    let activations = [activation];
    let zs: Matrix[] = []

    for (const [ b, w ] of zip(this.biases, this.weights)) {
      const z = w.multiply(activation).add(b);
      zs.push(z);
      activation = z.apply(sigmoid);
      activations.push(activation);
    }

    let delta = this
      .costDerivative(
        activations[activations.length - 1],
        Matrix.fromVector(y)
      )
      .hadamard(zs[zs.length - 1].apply(sigmoidPrime));
    nablaB[nablaB.length - 1] = delta
    nablaW[nablaW.length - 1] = delta.multiply(activations[activations.length - 2].transposed);

    for (const l of range(2, this.sizes.length)) {
      const z = zs[zs.length - l];
      const sp = z.apply(sigmoidPrime);
      delta = this.weights[this.weights.length - l + 1].transposed.multiply(delta).hadamard(sp);
      nablaB[nablaB.length - l] = delta;
      nablaW[nablaW.length - l] = delta.multiply(activations[activations.length - l - 1].transposed);
    }

    return [ nablaB, nablaW ];
  }

  evaluate(testData: [ number[], number ][]) {
    const testResults = testData
      .map(([ x, y ]) => {
        const computed = this.feedForward(x).transposed.elements[0];
        return [ indexMax(computed), y ];
      });
    return testResults.map(([ x, y ]) => x === y ? 1 : 0).reduce(sumFn, 0);
  }

  private costDerivative(outputActivations: Matrix, y: Matrix) {
    return outputActivations.subtract(y);
  }

  dump(): Model {
    return {
      weights: this.weights.map(m => m.elements),
      biases: this.biases.map(m => m.elements)
    }
  }

}
