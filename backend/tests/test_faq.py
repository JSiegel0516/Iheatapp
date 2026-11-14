import unittest
from unittest.mock import patch

from app.services.faq import entries, match_faq
from app.routes import chat_routes as chat


class FAQTests(unittest.TestCase):
    def test_all_curated_variants(self):
        for entry in entries():
            for question in entry["questions"]:
                with self.subTest(question=question):
                    self.assertEqual(match_faq(question), entry["answer"])

    def test_similar_wording(self):
        self.assertIsNotNone(match_faq("Please tell me where the WBGT data comes from?"))

    def test_specific_and_unrelated_questions_fall_through(self):
        for question in (
            "What is the WBGT in Brawley tomorrow at 2 PM?",
            "Where does iHeat get its data and is it safe to work?",
            "What timezone does NOAA use?",
            "What timezone does iHeat not use?",
            "Can I request historical WBGT for 1997?",
            "What is heat stroke?",
            "SST for January 1997",
        ):
            with self.subTest(question=question):
                self.assertIsNone(match_faq(question))

    def test_context_and_languages_bypass(self):
        self.assertIsNone(match_faq("How do I check WBGT?", has_context=True))
        self.assertIsNone(match_faq("How do I check WBGT?", language="es"))

    def test_el_nino_accents(self):
        self.assertEqual(match_faq("What is El Niño?"), match_faq("What is El Nino?"))
        self.assertIsNotNone(match_faq("What is El Niño?"))

    def test_specific_el_nino_requests_bypass_faq(self):
        for question in (
            "How long did El Nino last in 1997?",
            "What is El Nino doing today?",
            "What is the SST anomaly for December 1997?",
            "Compare the 1997-98 and 2015-16 El Nino",
            "How long does El Nino last in California?",
            "Is El Nino happening now?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(match_faq(question))


class FAQRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_faq_skips_expensive_services(self):
        with (patch.object(chat, "_retrieve") as retrieve,
              patch.object(chat, "generate_answer") as generate,
              patch.object(chat, "lookup") as lookup):
            result = await chat.chat(chat.ChatRequest(
                message="How do I check WBGT?", map_context={"wbgt_f": None, "category": None}))
            self.assertEqual(result.model_family, "FAQ")
            self.assertIn("Brawley", result.answer)
            retrieve.assert_not_called()
            generate.assert_not_called()
            lookup.assert_not_called()
