const fs = require('fs');
const csv = require('csv-parser');

const data = [];

fs.createReadStream('./tgs/0.csv')
  .pipe(csv())
  .on('data', (row) => {
    // Agregar cada fila del CSV al array de datos
    data.push(row);
  })
  .on('end', () => {
    // El array "data" ahora contiene las rutas y etiquetas de las imágenes
    console.log(data);
  });