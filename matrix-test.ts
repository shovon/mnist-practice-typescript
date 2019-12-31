import * as assert from 'assert';
import Matrix from './matrix';

function assertArrayEqual(a: number[], b: number[]) {
  assert(a.length === b.length);
  for (let i = 0; i < a.length; i++) {
    assert(a[i] === b[i]);
  }
}

{
  const a = new Matrix([[2]]);
  const b = new Matrix([[3]]);
  const result = a.multiply(b);
  assert(result.elements[0][0] === 6);
}

{
  const a = new Matrix([ [ 2, 3 ] ]);
  const b = new Matrix(
    [
      [ 4 ],
      [ 6 ]
    ]
  );
  const result = a.multiply(b);
  assert(result.elements[0][0] === 26);
}

{
  const a = new Matrix([ [ 2, 3 ] ]);
  const b = new Matrix([
    [ 4, 7 ],
    [ 6, 9 ]
  ]);
  const result = a.multiply(b);
  // const expected
  assertArrayEqual(result.elements[0], [ 26, 41 ]);
}

{
  const a = new Matrix([
    [ 1, 2 ],
    [ 2, 3 ]
  ]);
  const b = new Matrix([
    [ 5, 1 ],
    [ 3, 4 ]
  ]);
  const expected = new Matrix([
    [ 11, 9 ],
    [ 19, 14 ]
  ]);
  assert(a.multiply(b).areEqual(expected));
}

{
  const a = new Matrix([
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
  ]);
  const b = new Matrix([
    [ 0, 1, 3 ],
    [ 2, 4, 5 ],
    [ 6, 7, 8 ]
  ]);
  const expected = new Matrix([
    [ 1, 3, 6 ],
    [ 6, 9, 11 ],
    [ 13, 15, 17 ]
  ]);
  assert(a.add(b).areEqual(expected));
}

{
  const a = new Matrix([
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
  ]);
  const expected = new Matrix([
    [ 2, 4, 6 ],
    [ 8, 10, 12 ],
    [ 14, 16, 18 ]
  ]);
  assert(a.apply(n => n * 2).areEqual(expected));
}
