// 📡 Conecta con tu API de FastAPI
// Ruta relativa "/api": mismo codigo en dev y en produccion. En produccion
// Apache hace proxy /api -> :8080; en local el dev server de CRA hace lo mismo
// (ver "proxy" en package.json). Sobrescribir con REACT_APP_API_URL solo si
// hace falta apuntar a otro host.
const API_URL = process.env.REACT_APP_API_URL || "/api";

// Aqui estamos declarando el servicio de nuestra nueva API, habra que modificarla ya que por ahora unicamente sirve para
// fetch el data que viene de una direccion concreta de nuestra API, es decir, solo incluimos un endpoint y no es dinamico
export const fetchSampleData = async () => {
  try {
    const response = await fetch(`${API_URL}/wbgt/sample`);
    if (!response.ok) {
      throw new Error("Error obteniendo datos");
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
};
