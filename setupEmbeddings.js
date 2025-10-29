// Cargamos las variables de entorno desde el archivo .env
import 'dotenv/config';

// Módulo para trabajar con el sistema de archivos (leer/escribir archivos)
import fs from 'fs';

// Cliente de OpenAI
import OpenAI from 'openai';

// Necesario para importar módulos CommonJS dentro de un proyecto ESM
import { createRequire } from 'module';
const requireCJS = createRequire(import.meta.url);

// Importamos pdf-parse usando require porque es CommonJS
const pdfParseModule = requireCJS('pdf-parse');
const pdfParse = pdfParseModule.default || pdfParseModule; // Nos aseguramos de obtener la función

// Inicializamos OpenAI con la API Key de nuestro .env
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Función principal para generar embeddings de los PDFs
async function main() {
  // Leemos todos los archivos de la carpeta 'docs' que terminen en .pdf
  const docs = fs.readdirSync('docs').filter(f => f.endsWith('.pdf'));

  // Array donde guardaremos los embeddings de cada PDF
  const embeddings = [];

  // Recorremos cada PDF
  for (const file of docs) {
    // Leemos el contenido binario del archivo PDF
    const data = fs.readFileSync(`docs/${file}`);

    // Extraemos el texto del PDF
    const pdf = await pdfParse(data);
    const text = pdf.text;

    // Generamos el embedding usando la API de OpenAI
    const emb = await openai.embeddings.create({
      model: 'text-embedding-3-small', // Modelo de embeddings
      input: text                        // Texto extraído del PDF
    });

    // Guardamos el nombre del archivo, su texto y su embedding
    embeddings.push({ file, text, embedding: emb.data[0].embedding });
    console.log(`Procesado: ${file}`); // Indicamos que se ha procesado este PDF
  }

  // Guardamos todos los embeddings en un archivo JSON
  fs.writeFileSync('embeddings.json', JSON.stringify(embeddings, null, 2));
  console.log('Embeddings guardados en embeddings.json');
}

// Ejecutamos la función principal
main();
