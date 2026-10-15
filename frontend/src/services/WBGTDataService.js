import pako from "pako";

// Base URL of the FastAPI backend. Relative "/api" by default so the SAME code
// works everywhere:
//   - production: served from rhialarm.sdsu.edu, Apache proxies /api -> :8080
//   - local dev:  CRA dev server proxies /api -> :8080 (see "proxy" in package.json)
// Only set REACT_APP_API_URL if you ever need to point at a different host.
const API_URL = process.env.REACT_APP_API_URL || "/api";

export async function fetchWBGTData(datetimeStr, forecastHour) {
  // Backend route: GET /api/wbgt/{datetime_str}/{forecast_hour}
  // datetimeStr must be formatted as YYYY-MM-DD_HH, forecastHour is 1..36.
  const url = `${API_URL}/wbgt/${datetimeStr}/${forecastHour}`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const json = await response.json();

    // ✅ Si viene comprimido, lo descomprimimos
    if (!json.values && json.values_compressed) {
      const binaryString = atob(json.values_compressed);
      const byteArray = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        byteArray[i] = binaryString.charCodeAt(i);
      }

      const decompressed = pako.inflate(byteArray, { to: "string" });
      json.values = JSON.parse(decompressed);
    }

    return json;
  } catch (error) {
    console.error("❌ Error fetching WBGT data:", error);
    return null;
  }
}
