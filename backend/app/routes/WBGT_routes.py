from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from app.pipelines.WBGT.wbgt_pipeline import generate_wbgt  # 👈 Importa la función
import time

router = APIRouter()


@router.get("/wbgt/sample")
async def get_wbgt_sample():
    """Convenience endpoint used by the debug page (/wbgt in the frontend).

    Returns a WBGT response for the most recent likely-available model run
    (now - 2h, forecast hour 1), matching what the map SliderControl requests.
    """
    try:
        start_time = time.time()
        dt = datetime.utcnow() - timedelta(hours=2)
        dt = dt.replace(minute=0, second=0, microsecond=0)

        result = generate_wbgt(dt, forecast_hour=1)

        print(f"⏱️ /wbgt/sample generado en: {time.time() - start_time:.2f}s")
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="❌ Archivo WBGT no encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error interno: {str(e)}")


@router.get("/wbgt/{datetime_str}/{forecast_hour}")
async def get_wbgt_data(datetime_str: str, forecast_hour: int):
    try:
        # ⏱️ Medir tiempo
        start_time = time.time()

        # Convertir string a datetime
        dt = datetime.strptime(datetime_str, "%Y-%m-%d_%H")

        # Llamar a tu función
        result = generate_wbgt(dt, forecast_hour)

        end_time = time.time()
        print(f"⏱️ Tiempo de respuesta total: {end_time - start_time:.2f} segundos")

        return result

    except ValueError:
        raise HTTPException(status_code=400, detail="❌ Formato de fecha inválido. Usa YYYY-MM-DD_HH")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="❌ Archivo WBGT no encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error interno: {str(e)}")
