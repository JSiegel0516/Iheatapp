import unittest
from unittest.mock import AsyncMock, patch

from app.routes import chat_routes as chat


class ForecastCitations(unittest.IsolatedAsyncioTestCase):
    async def run_chat(self, answer, matches):
        with (
            patch.object(chat, "lookup", return_value={
                "wbgt_f": 65.6, "location": "Brawley",
                "forecast_valid_at": "2026-10-19T22:00:00-07:00",
            }),
            patch.object(chat, "_enso_scopes", return_value=([], None)),
            patch.object(chat, "_retrieve", return_value=matches),
            patch.object(chat, "generate_answer", new_callable=AsyncMock,
                         return_value=answer) as generate,
        ):
            result = await chat.chat(chat.ChatRequest(
                message="WBGT in Brawley today at 10 PM"))
            prompt = generate.call_args.args[0][0]["content"]
            self.assertNotIn("NOAA-NBM-POINT", prompt)
            self.assertIn("without citation markers or a Sources list", prompt)
            self.assertIn('"wbgt_f": 65.6', prompt)
            return result

    async def test_direct_forecast_has_no_source(self):
        result = await self.run_chat("Brawley's forecast WBGT is 65.6°F at 10 PM Pacific.", [])
        self.assertEqual(result.citations, [])

    async def test_document_guidance_keeps_citations(self):
        result = await self.run_chat(
            "Forecast WBGT: 65.6°F. Take breaks as recommended [1].",
            [{"source_id": "doc", "title": "Heat guidance", "organization": "Test",
              "url": "https://example.com/heat", "text": "Take breaks."}])
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].title, "Heat guidance")
