import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import xarray as xr

from app.services import wbgt_lookup as wbgt


class PointLookupTests(unittest.TestCase):
    now = datetime(2026, 10, 19, 18, tzinfo=timezone.utc)

    @patch("herbie.Herbie")
    def test_unknown_units_only_recovered_for_ncep_wbgt(self, herbie):
        attrs = {"units": "unknown", "GRIB_centre": "kwbc",
                 "GRIB_discipline": 0, "GRIB_parameterCategory": 0,
                 "GRIB_parameterNumber": 206}
        for parameter, centre, accepted in ((206, "kwbc", True),
                                             (205, "kwbc", False),
                                             (206, "other", False)):
            with self.subTest(parameter=parameter, centre=centre):
                wbgt.forecast_point.cache_clear()
                ds = xr.Dataset(
                    {"unknown": (("y", "x"), [[300., 301.], [302., 303.]],
                                 {**attrs, "GRIB_parameterNumber": parameter,
                                  "GRIB_centre": centre})},
                    coords={"latitude": (("y", "x"), [[33., 33.], [34., 34.]]),
                            "longitude": (("y", "x"), [[245., 246.], [245., 246.]])})
                herbie.return_value.xarray.return_value = ds
                if accepted:
                    value, lat, lon = wbgt.forecast_point(
                        "2026-10-19T20:00:00+00:00", 9, 33., -115.)
                    self.assertAlmostEqual(value, 80.33, places=2)
                    self.assertEqual((lat, lon), (33., -115.))
                else:
                    with self.assertRaisesRegex(ValueError, "Unsupported WBGT units"):
                        wbgt.forecast_point("2026-10-19T20:00:00+00:00", 9, 33., -115.)
        wbgt.forecast_point.cache_clear()

    def test_pacific_time(self):
        place, target = wbgt.parse_request(
            "WBGT in Chiriaco Summit tomorrow at 2 PM", self.now)
        self.assertEqual(place, "Chiriaco Summit")
        self.assertEqual(target.astimezone(timezone.utc).hour, 21)
        self.assertEqual(target.day, 20)

    def test_screenshot_request(self):
        text = "what is the wbgt score in san diego california oct 19 10:00 pm"
        self.assertTrue(wbgt.requested_lookup(text))
        place, target = wbgt.parse_request(text, self.now)
        self.assertEqual(place, "san diego california")
        self.assertEqual(target.isoformat(), "2026-10-19T22:00:00-07:00")

    def test_named_date_variants(self):
        for date in ("October 19", "Oct. 19th", "October 19, 2026", "oct 19 2026"):
            with self.subTest(date=date):
                _, target = wbgt.parse_request(
                    f"WBGT in Chiriaco Summit on {date} at 10 PM", self.now)
                self.assertEqual(target.isoformat(), "2026-10-19T22:00:00-07:00")
        self.assertTrue(wbgt.requested_lookup("WBGT in Imperial October 19"))
        with self.assertRaises(wbgt.LookupUnavailable):
            wbgt.parse_request("WBGT in Imperial February 30 at 10 PM", self.now)

    @patch.object(wbgt, "forecast_point")
    @patch.object(wbgt, "geocode", return_value=(32.72, -117.16, "San Diego"))
    def test_named_date_still_checks_coverage(self, geocode, point):
        with self.assertRaisesRegex(wbgt.LookupUnavailable, "outside"):
            wbgt.lookup("WBGT in san diego california oct 19 10:00 pm", self.now)
        point.assert_not_called()

    def test_time_before_date(self):
        place, target = wbgt.parse_request(
            "WBGT in Chiriaco Summit at 2 PM tomorrow", self.now)
        self.assertEqual(place, "Chiriaco Summit")
        self.assertEqual(target.hour, 14)

    def test_explicit_date_coordinates(self):
        place, target = wbgt.parse_request(
            "WBGT at 33.66, -115.72 on 2026-10-19 at 14:00", self.now)
        self.assertEqual(place, (33.66, -115.72))
        self.assertEqual(target.hour, 14)

    def test_clarifications(self):
        for text in (
            "WBGT in Chiriaco Summit tomorrow",
            "WBGT in Chiriaco Summit tomorrow at 2:30 PM",
            "WBGT in Chiriaco Summit 2026-02-30 at 2 PM",
            "WBGT in Chiriaco Summit tomorrow at 2 PM UTC",
            "WBGT in Chiriaco Summit 2026-03-08 at 2 AM",
            "WBGT in Chiriaco Summit 2026-11-01 at 1 AM",
        ):
            with self.subTest(text=text), self.assertRaises(wbgt.LookupUnavailable):
                wbgt.parse_request(text, self.now)

    def test_no_dataset_routing(self):
        self.assertFalse(wbgt.requested_lookup("SST temperature on 1997-12-01"))
        self.assertTrue(wbgt.requested_lookup("WBGT tomorrow at 2 PM"))
        self.assertTrue(wbgt.requested_lookup("WBGT in Imperial at 14:00"))

    @patch.object(wbgt, "forecast_point")
    def test_fallback_preserves_requested_time(self, point):
        point.side_effect = [ValueError("unavailable run"), (85.2, 33.66, -115.72)]
        with self.assertLogs(wbgt.logger, level="ERROR"):
            result = wbgt.lookup("WBGT at 33.66, -115.72 today at 2 PM", self.now)
        for call in point.call_args_list:
            run, lead, _, _ = call.args
            valid = datetime.fromisoformat(run) + timedelta(hours=lead)
            self.assertEqual(valid, datetime.fromisoformat(result["forecast_valid_utc"]))
        self.assertEqual(result["wbgt_f"], 85.2)

    @patch.object(wbgt, "forecast_point")
    def test_out_of_coverage_does_not_fetch(self, point):
        for text in (
            "WBGT at 33.66, -115.72 on 2020-01-01 at 2 PM",
            "WBGT at 40.00, -120.00 today at 2 PM",
        ):
            with self.assertRaises(wbgt.LookupUnavailable):
                wbgt.lookup(text, self.now)
        point.assert_not_called()


if __name__ == "__main__":
    unittest.main()
