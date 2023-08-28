const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { promisify } = require('util');
const { createCanvas, loadImage } = require('canvas');

async function readCSV(filePath) {
  try {
    const data = await promisify(fs.readFile)(filePath, 'utf8');
    const lines = data.trim().split('\n');
    const headers = lines[0].split(',');
    const rows = lines.slice(1).map(line => {
      const values = line.split(',');
      return headers.reduce((obj, header, index) => {
        obj[header] = values[index];
        return obj;
      }, {});
    });
    return rows;
  } catch (err) {
    console.error('Error reading CSV:', err);
    throw err;
  }
}

async function loadImagesAndLabels(data) {
  const images = [];
  const labels = [];
  
  for (const row of data) {
    const image = await loadImage(row.ruta);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, image.width, image.height);

    // Obtiene los datos de píxeles del canvas y crea un tensor de TensorFlow
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const pixels = new Uint8Array(imageData.data.buffer);
    const tensor = tf.tensor(pixels, [image.height, image.width, 4], 'int32');

    images.push(tensor);
    labels.push(Number(row.etiqueta));
  }

  // Convertir las etiquetas a un arreglo plano (flatten) o TypedArray
  const flattenedLabels = [].concat(...labels);
  
  return { images, labels: tf.tensor1d(flattenedLabels, 'int32') };
}

function preprocessData(data, imageSize) {
  const resizedImages = data.images.map(image => tf.image.resizeBilinear(image, [imageSize, imageSize]));
  const preprocessedImages = tf.stack(resizedImages);

  // Obtener los valores del tensor y convertirlos en un arreglo plano
  const flattenedLabels = data.labels.arraySync().flat();

  const preprocessedLabels = tf.tensor1d(flattenedLabels, 'int32');
  return { images: preprocessedImages, labels: preprocessedLabels };
}

function disposeData(data) {
  data.images.forEach(image => image.dispose());
  data.labels.dispose();
}

function createModel(inputShape, numClasses) {
  const model = tf.sequential();

  // Agregar capas convolucionales y de agrupación
  model.add(tf.layers.conv2d({
    inputShape,
    filters: 32, // Número de filtros
    kernelSize: 3, // Tamaño del filtro
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Agregar otra capa convolucional y de agrupación
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Agregar una capa de aplanado para convertir la salida 2D en 1D
  model.add(tf.layers.flatten());

  // Agregar capas densas para la clasificación
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // Agregar la capa de salida con activación softmax para la clasificación de múltiples clases
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

  return model;
}

const load_scr = async () => {
  const MODEL_PATH = './models';
  const CSV_PATH = './tgs/0.csv';
  const IMAGE_SIZE = 28;
  const NUM_CLASSES = 10;
  const BATCH_SIZE = 32;
  const EPOCHS = 10;

  // Cargar y preparar los datos
  const data = await readCSV(CSV_PATH);
  const img_data = await loadImagesAndLabels(data)
  const preprocessedData = preprocessData(img_data, IMAGE_SIZE, NUM_CLASSES);
  disposeData(img_data);

  // Definir el modelo
  const inputShape = [IMAGE_SIZE, IMAGE_SIZE, 4];
  const model = createModel(inputShape, NUM_CLASSES);

  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(),
    metrics: ['accuracy'],
  });

  // Entrenar el modelo
  const history = await model.fit(
    preprocessedData.images,
    tf.oneHot(preprocessedData.labels, NUM_CLASSES),
    {
      batchSize: BATCH_SIZE,
      epochs: EPOCHS,
      shuffle: true,
      validationSplit: 0.2,
      callbacks: tf.node.tensorBoard('/tmp/tflogs'),
    }
  );

  console.log(history.history); // Muestra el historial del entrenamiento

  // Guardar el modelo entrenado
  if (!fs.existsSync(MODEL_PATH)) {
    fs.mkdirSync(MODEL_PATH);
  }
  await model.save(`file://${path.resolve(MODEL_PATH)}`);

}

load_scr();