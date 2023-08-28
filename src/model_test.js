const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');

// Función para cargar el modelo entrenado
async function loadModel(modelPath) {
  console.log('Cargando el modelo...');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  console.log('Modelo cargado con exito');
  return model;
}

// Función para cargar la imagen de prueba con 4 canales de color
async function loadImageWithChannels(imagePath, numChannels) {
  const image = await loadImage(imagePath);
  if(!image) {
    console.log('No se encuentra la imagen!')
  } 
  const canvas = createCanvas(image.width, image.height);
  console.log(canvas)
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);

  // Obtener los valores de píxeles de la imagen
  const imageData = ctx.getImageData(0, 0, image.width, image.height);

  // Obtener los valores de píxeles y crear un tensor 3D con los valores y las dimensiones adecuadas
  const pixels = new Uint8Array(imageData.data);
  const shape = [imageData.height, imageData.width, numChannels];
  return tf.tensor3d(pixels, shape, 'int32');
}

// Función para redimensionar y normalizar una imagen
async function preprocessImage(imagePath, imageSize) {
  const image = await loadImage(imagePath);

  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);

  // Obtener los valores de píxeles de la imagen
  const imageData = ctx.getImageData(0, 0, image.width, image.height);

  // Obtener los valores de píxeles y crear un tensor 3D con los valores y las dimensiones adecuadas
  const pixels = new Uint8Array(imageData.data);
  const shape = [imageData.height, imageData.width, 4]; // Usar 4 canales de color
  const preprocessedImage = tf.tensor3d(pixels, shape, 'int32');

  // Redimensionar la imagen
  const resizedImage = tf.image.resizeBilinear(preprocessedImage, [imageSize, imageSize]);

  // Normalizar los valores de píxeles entre 0 y 1
  const normalizedImage = resizedImage.div(255);

  console.log('Preprocessed Image:', normalizedImage.arraySync());

  return normalizedImage;
}

// Función para realizar predicciones con el modelo
async function predictWithModel(model, imagePath, imageSize) {
  const preprocessedImage = await preprocessImage(imagePath, imageSize);

  // Agregar una dimensión extra para que el modelo pueda hacer la predicción
  const input = preprocessedImage.expandDims();

  // Realizar la predicción
  const predictions = model.predict(input);

  // Obtener el índice de la clase predicha
  const predictedClassIndex = predictions.argMax(1).dataSync()[0];

  console.log('Shape of input tensor:', input.shape);
  console.log('Input tensor:', input.arraySync());
  console.log('Predictions:', predictions.arraySync());

  return predictedClassIndex;
}

// Ruta donde guardaste el modelo entrenado
const MODEL_PATH = './models/model.json';

// Ruta de la imagen de prueba que quieres usar para la predicción
const IMAGE_PATH = './assets/imgs/1679343557663.jpg';

// Tamaño de las imágenes que se utilizaron durante el entrenamiento
const IMAGE_SIZE = 28;

// Función principal para probar el modelo
async function main() {
  // Cargar el modelo entrenado
  const model = await loadModel(MODEL_PATH);

  console.log('Cargando la imagen de prueba...');
  // Cargar la imagen de prueba con 4 canales de color
  const preprocessedImage = await loadImageWithChannels(IMAGE_PATH, 4);
  console.log('Imagen de prueba cargada.');

  // console.log(preprocessedImage.shape);
  console.log('Realizando predicción con el modelo...');
  // Realizar predicciones con el modelo en la imagen de prueba
  const predictedClassIndex = await predictWithModel(
    model,
    preprocessedImage,
    IMAGE_SIZE
  );
  console.log('Predicción realizada.');
  console.log('Predicción:', predictedClassIndex);

  // Liberar memoria del modelo
  model.dispose();
}

main();
