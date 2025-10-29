import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import multer from 'multer';
import fetch from 'node-fetch';
import fs from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.static('public'));
app.use(express.json());

// === Cargar embeddings de tus PDFs ===
const docsEmbeddings = JSON.parse(fs.readFileSync('embeddings.json', 'utf-8'));

// === Memoria de conversación ===
let historial = [
  { role: "system", content: "Eres un asistente inteligente. Puedes usar los PDFs cargados si son relevantes, pero también responder preguntas generales." }
];

// === Función de similitud coseno ===
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// === Generar audio con ElevenLabs ===
async function generarVozElevenLabsBase64(text, voiceId) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      voice_settings: { stability: 0.75, similarity_boost: 0.75 }
    })
  });
  const arrayBuffer = await response.arrayBuffer();
  const base64Audio = Buffer.from(arrayBuffer).toString('base64');
  return `data:audio/mpeg;base64,${base64Audio}`;
}

// === Generar respuesta con contexto ===
async function generarRespuesta(userText) {
  // Guardar el mensaje del usuario
  historial.push({ role: "user", content: userText });

  // Crear embedding de la consulta
  const queryEmb = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: userText
  });
  const qEmbedding = queryEmb.data[0].embedding;

  // Buscar documento más similar
  let topDoc = docsEmbeddings[0];
  let maxSim = -1;
  for (const doc of docsEmbeddings) {
    const sim = cosineSimilarity(qEmbedding, doc.embedding);
    if (sim > maxSim) {
      maxSim = sim;
      topDoc = doc;
    }
  }

  // Si el documento es relevante, añadirlo como contexto
  const UMBRAL_SIMILITUD = 0.7;
  if (maxSim > UMBRAL_SIMILITUD) {
    historial.push({
      role: "system",
      content: `Información relevante del documento:\n${topDoc.text}`
    });
  }

  // Generar respuesta con el historial completo
  const responseGPT = await openai.responses.create({
    model: "gpt-4.1-mini",
    input: historial
  });

  const answer = responseGPT.output_text;

  // Guardar la respuesta del asistente
  historial.push({ role: "assistant", content: answer });

  // Generar voz
  const VOICE_ID = process.env.ELEVENLABS_VOICE_ID;
  const audioBase64 = await generarVozElevenLabsBase64(answer, VOICE_ID);

  return { question: userText, reply: answer, audio: audioBase64 };
}

// === Endpoint para texto ===
app.post('/api/texto', async (req, res) => {
  try {
    const userText = req.body.text;
    const responseData = await generarRespuesta(userText);
    res.json(responseData);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error procesando el texto" });
  }
});

// === Endpoint para audio ===
app.post('/api/voz', upload.single('audio'), async (req, res) => {
  try {
    const tempPath = join(tmpdir(), `${Date.now()}.webm`);
    fs.writeFileSync(tempPath, req.file.buffer);

    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempPath),
      model: "gpt-4o-mini-transcribe",
    });

    fs.unlinkSync(tempPath);

    const responseData = await generarRespuesta(transcription.text);
    res.json(responseData);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error procesando el audio" });
  }
});

app.listen(3000, () => console.log("Servidor en http://localhost:3000"));
