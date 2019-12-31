/**
 * Returns a new array of numbers that is the vector dot product of the two
 * supplied (a and b) vectors.
 * @param a The left operand of the dot product
 * @param b The right operand of the dot product
 */
export function dot(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error(`The dimensions of the vectors do not match. a.length: ${a.length}, b.length: ${b.length}`)
  }
  return a.map((el, i) => el * b[i]).reduce((sum, next) => sum + next, 0);
}

/**
 * Returns a new array of numbers that is the addition of the two supplied (a
 * and b) vectors.
 * @param a The left operand of the dot product
 * @param b The right operand of the dot product
 */
export function add(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error(`The dimensions of the vectors do not match. a.length: ${a.length}, b.length: ${b.length}`)
  }

  return a.map((el, i) => el + b[i]);
}

/**
 * Returns a new array of numbers that is the subtraction of the two supplied (a
 * and b) vectors.
 * @param a The left operand of the dot product
 * @param b The right operand of the dot product
 */
export function subtract(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error(`The dimensions of the vectors do not match. a.length: ${a.length}, b.length: ${b.length}`)
  }

  return a.map((el, i) => el - b[i]);
}

/**
 * Returns a new array of numbers that is the elementwise multiplication of the
 * two supplied (a and b) vectors.
 * @param a The left operand of the dot product
 * @param b The right operand of the dot product
 */
export function hadamard(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error(`The dimensions of the vectors do not match. a.length: ${a.length}, b.length: ${b.length}`)
  }

  return a.map((el, i) => el * b[i]);
}

/**
 * Returns a new array of numbeers that is the elementwise multiplication by the
 * supplied scalar.
 * @param scalar The scalar that will multiply each element
 * @param vector The vector that will have each element multiplied by the scalar
 */
export function scalarMultiply(scalar: number, vector: number[]) {
  return vector.map(el => el * scalar);
}

/**
 * Applies the supplied function to the supplied vector.
 * @param vector The vector to apply the function to
 * @param fn The function that is invoked to apply an operation on vector
 *   Component
 */
export function vectorApply(vector: number[], fn: (number) => number) {
  return vector.map(fn);
}