import * as vectorLib from './vector';

function randn_bm() {
  var u = 0, v = 0;
  while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while(v === 0) v = Math.random();
  return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

/**
 * A helper class that 
 */
export default class Matrix {

  /**
   * Initializes a new instance of a Matrix anemic class.
   * @param values The individual rows for our matrix.
   */
  constructor(private values: number[][]) {}

  /**
   * Returns a Matrix version of the supplied vector.
   * 
   * The matrix will consist of a single column, that has all of the elements
   * of the vector.
   * @param vector The vector from which to convert to a matrix type.
   */
  static fromVector(vector: number[]) {
    return new Matrix([ vector ]).transposed;
  }

  /**
   * Gets a random matrix, given the number of rows, and the number of colums.
   * @param rows The number of rows
   * @param columns The number of columns
   */
  static random(rows: number, columns: number) {
    const arr = Array(rows)
      .fill([])
      .map(() => Array(columns).fill(0).map(() => randn_bm()));
    return new Matrix(arr);
  }

  /**
   * Gets a matrix, given the number of rows, and the number of columns.
   * @param rows The number of rows
   * @param columns The number of columns
   */
  static zeros(rows: number, columns: number) {
    const arr = Array(rows)
      .fill([])
      .map(() => Array(columns).fill(0));
    return new Matrix(arr);
  }

  /**
   * The number of rows in the matrix.
   */
  get rowCount() { return this.values.length; }

  /**
   * The number of columns in the matrix.
   */
  get colCount() { return this.values[0].length; }

  /**
   * Gets a tuple that represents the shape of the array.
   */
  get shape() { return [ this.rowCount, this.colCount ]; }

  /**
   * Gets the elements in array form.
   */
  get elements() { return this.values.map(row => row.slice()); }

  /**
   * Gets the matrix in transposed form.
   */
  get transposed() {
    // Iterate through each column of each row
    const mat = [];

    for (let col = 0; col < this.values[0].length; col++) {
      const vector = [];
      for (let row = 0; row < this.values.length; row++) {
        vector.push(this.values[row][col]);
      }
      mat.push(vector);
    }

    return new Matrix(mat);
  }

  // In order to get the column vector: m.transposed.elements[0]

  /**
   * Returns a new matrix that is the matrix multiplication of this matrix, with
   * the supplied m matrix.
   * @param m The right operand in the matrix multiplication.
   */
  multiply(m: Matrix) {
    if (this.colCount !== m.rowCount) {
      throw new Error('The right operand row count does not match the left operand\'s column count.');
    }

    const mt = m.transposed;
    const mat = [];
    
    for (let row = 0; row < this.values.length; row++) {
      const vector = [];
      for (let rowM = 0; rowM < mt.values.length; rowM++) {
        vector.push(vectorLib.dot(this.values[row], mt.values[rowM]));
      }
      mat.push(vector);
    }

    return new Matrix(mat);
  }
  
  /**
   * Returns a new matrix that is the matrix addition of this matrix, with the
   * supplied m matrix.
   * @param m The right operand in the matrix addition.
   */
  add(m: Matrix) {
    if (this.colCount !== m.colCount || this.rowCount !== m.rowCount) {
      throw new Error('The dimensions of the right operand does not match the dimensions of the left operand')
    }

    return new Matrix(
      this.values.map((row, i) => vectorLib.add(row, m.values[i]))
    );
  }

  /**
   * Returns a new matrix that is the matrix subtraction of this matrix, with
   * the supplied m matrix.
   * @param m The right operand in the matrix addition.
   */
  subtract(m: Matrix) {
    if (this.colCount !== m.colCount || this.rowCount !== m.rowCount) {
      throw new Error('The dimensions of the right operand does not match the dimensions of the left operand')
    }

    return new Matrix(
      this.values.map((row, i) => vectorLib.subtract(row, m.values[i]))
    );
  }

  /**
   * Returns a new matrix that is the elementwise multiplication of this matrix,
   * with the supplied m matrix.
   * @param m The right operand in the matrix addition.
   */
  hadamard(m: Matrix) {
    if (this.colCount !== m.colCount || this.rowCount !== m.rowCount) {
      throw new Error('The dimensions of the right operand does not match the dimensions of the left operand')
    }

    return new Matrix(
      this.values.map((row, i) => vectorLib.hadamard(row, m.values[i]))
    );
  }

  /**
   * Returns a new matrix that is each element of this matrix, multiplied by the
   * supplied scalar.
   * @param m The right operand in the matrix addition.
   */
  scalarMultiply(scalar: number) {
    return new Matrix(
      this.values.map((row) => vectorLib.scalarMultiply(scalar, row))
    );
  }

  /**
   * Applies the given function to each element in this matrix.
   * @param fn The function to apply the matrix onto
   */
  apply(fn: (number) => number) {
    return new Matrix(this.values.map((row) => row.map(fn)));
  }

  /**
   * Compares this Matrix with the supplied Matrix m.
   * @param m The matrix to compare with this Matrix
   */
  areEqual(m: Matrix) {
    // TODO: bear in mind, floating points are a bit finicky. A hard equality
    // check will fail. Ideally, you want to employ some floating point
    // tolerance.

    if (this.colCount !== m.colCount || this.rowCount !== m.rowCount) {
      return false;
    }

    return this.values.every((row, i) =>
      row.every((value, j) => value === m.values[i][j])
    );
  }

}
