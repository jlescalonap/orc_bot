const Tesseract = require('tesseract.js');
const sharp = require('sharp');

// Ruta de la imagen de entrada (asegúrate de que la imagen exista)
// 0021995m3
const imagePath = 'src/3.jpeg';

// Función para extraer texto de la imagen
async function extractTextFromImage() {
  try {
    // const preprocessedImage = await sharp(imagePath).modulate({ brightness: 1.2, contrast: 1.5 }).toBuffer();
    const preprocessedImage = await sharp(imagePath).modulate({ brightness: 1.5, contrast: 1.4 }).toBuffer();
    const result = await Tesseract.recognize(
      preprocessedImage,
      'eng', // Lenguaje: 'eng' para inglés, puedes usar otros idiomas según tu necesidad
      {
        logger: info => console.log(info), // Para obtener información adicional de proceso si es necesario
        tessedit_char_whitelist: '0123456789',
        tessedit_ocr_engine_mode: 'Tesseract_LSTM_Compositing',
      }
    );
    
    console.log('Texto extraído:', result.data.text);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Llamada a la función de extracción
extractTextFromImage();
