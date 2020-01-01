import Network from "./network";
import { indexMax } from './helpers';
import 'babel-polyfill';
import "core-js/stable";
import "regenerator-runtime/runtime";

const model = require('./model.json');

const drawing = document.getElementById('drawing') as HTMLCanvasElement;
const preview = document.getElementById('preview') as HTMLCanvasElement;
const grayscale = document.getElementById('grayscale') as HTMLCanvasElement;
const clearButton = document.getElementById('clear');
const predictionOutput = document.getElementById('prediction');

const drawingCtx = drawing.getContext('2d');
const previewCtx = preview.getContext('2d');
const grayscaleCtx = grayscale.getContext('2d');

let mousedown = false;

function clear() {
  drawingCtx.fillStyle = 'white';
  drawingCtx.fillRect(0, 0, drawing.width, drawing.height);
}

clear();

function drawCircle(event: MouseEvent) {
  mousedown = true;
  drawingCtx.beginPath();
  drawingCtx.fillStyle = 'black';
  const thickness = 12;
  drawingCtx.arc(event.x - thickness * 2, event.y - thickness * 2, thickness, 0, 2 * Math.PI);
  drawingCtx.fill();
  drawingCtx.closePath();

  previewCtx.drawImage(drawing, 0, 0, 28, 28);
}

drawing.addEventListener('mousedown', drawCircle);
drawing.addEventListener('mouseup', () => {
  mousedown = false;
});
drawing.addEventListener('mousemove', (event) => {
  if (mousedown) {
    drawCircle(event);
  }
});
clearButton.addEventListener('click', () => {
  clear();
});

const network = Network.load(model);
setInterval(() => {
  const imageData = previewCtx.getImageData(0, 0, preview.width, preview.height);
  const pixels: number[] = [];
  for (let i = 0; i < preview.width * preview.height; i++) {
    const r = imageData.data[i * 4];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];
    const pixel = ((r + g + b) / 3) / 255;
    pixels.push(1 - pixel);
  }
  const grayscaleData = grayscaleCtx.getImageData(0, 0, 28, 28);
  for (let i = 0; i < pixels.length; i++) {
    grayscaleData.data[i * 4] = (pixels[i] * 255) | 0;
    grayscaleData.data[i * 4 + 1] = (pixels[i] * 255) | 0;
    grayscaleData.data[i * 4 + 2] = (pixels[i] * 255) | 0;
    grayscaleData.data[i * 4 + 3] = 255;
  }
  grayscaleCtx.putImageData(grayscaleData, 0, 0);
  predictionOutput.innerText = indexMax(network.feedForward(pixels).transposed.elements[0]).toString();
}, 1000);
