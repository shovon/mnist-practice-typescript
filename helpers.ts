export function indexMax(array: number[]) {
  return array.reduce((max, next, i) => next > array[max] ? i : max, 0);
}