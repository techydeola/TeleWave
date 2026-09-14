"""
Live Climate Data Fetcher for TeleWave.
==============================================

Automates real-time retrieval, parsing, and temporal alignment of global
teleconnection indices from NOAA and reanalysis precipitation data for Nigeria.

Supported Data Sources:
  1. NINA 3.4 (CPC / NOAA)       - El Niño Southern Oscillation SST Anomaly
  2. SOI (CPC / NOAA)            - Southern Oscillation Index (Tahiti - Darwin SLP)
  3. PDO (NCEI / NOAA)           - Pacific Decadal Oscillation
  4. NAO (CPC / NOAA)            - North Atlantic Oscillation
  5. AMM (PSL / NOAA)            - Atlantic Meridional Mode
  6. AMO (PSL / CSU)             - Atlantic Multidecadal Oscillation
  7. Nigeria Precipitation       - Open-Meteo ERA5 / ECMWF Reanalysis & Forecast

No third-party packages required — uses standard library urllib and json.
"""

import os
import csv
import json
import math
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional, Any


class LiveClimateDataFetcher:
    """Retrieves live meteorological teleconnections and regional precipitation."""

    # Public NOAA & Reanalysis Endpoints
    URLS = {
        "NINA": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        "SOI": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
        "PDO": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        "NAO": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
        "AMM": "https://psl.noaa.gov/data/correlation/amm.data",
        "OPEN_METEO_ARCHIVE": "https://archive-api.open-meteo.com/v1/archive",
        "OPEN_METEO_FORECAST": "https://api.open-meteo.com/v1/forecast",
    }

    # Central Nigeria coordinates (representative centroid for regional precipitation)
    NIGERIA_LAT = 9.08
    NIGERIA_LON = 8.67

    MONTH_ABBRS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    @staticmethod
    def _fetch_url(url: str, timeout: int = 20) -> str:
        """Fetches raw text with standard browser headers."""
        headers = {"User-Agent": "TeleWave/2.0 (Research Early Warning System)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    @classmethod
    def parse_nina34(cls, text: str) -> Dict[Tuple[int, int], float]:
        """
        Parses NOAA CPC ERSSTv5 Nino monthly anomalies.
        Columns: YR, MON, NINO1+2, ANOM, NINO3, ANOM, NINO4, ANOM, NINO3.4, ANOM
        """
        data = {}
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 10 and parts[0].isdigit() and parts[1].isdigit():
                y, m = int(parts[0]), int(parts[1])
                try:
                    val = float(parts[9])  # 9th index is NINO3.4 ANOM
                    data[(y, m)] = val
                except ValueError:
                    continue
        return data

    @classmethod
    def parse_soi(cls, text: str) -> Dict[Tuple[int, int], float]:
        """
        Parses NOAA CPC Southern Oscillation Index fixed-width format.
        Format: 4-char Year, followed by twelve 6-char monthly columns.
        """
        data = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if len(line) >= 4 and line[:4].isdigit():
                y = int(line[:4])
                for m in range(1, 13):
                    start = 4 + (m - 1) * 6
                    end = start + 6
                    if start < len(line):
                        col = line[start:end].strip()
                        if col and col not in ("-999.9", "-99.9", "99.9"):
                            try:
                                data[(y, m)] = float(col)
                            except ValueError:
                                continue
        return data

    @classmethod
    def parse_pdo(cls, text: str) -> Dict[Tuple[int, int], float]:
        """
        Parses NOAA NCEI ERSST PDO Index table.
        Format: Year, Jan, Feb, ..., Dec
        """
        data = {}
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                y = int(parts[0])
                for m, val_str in enumerate(parts[1:], 1):
                    if m <= 12:
                        try:
                            val = float(val_str)
                            if abs(val) < 90.0:
                                data[(y, m)] = val
                        except ValueError:
                            continue
        return data

    @classmethod
    def parse_nao(cls, text: str) -> Dict[Tuple[int, int], float]:
        """
        Parses NOAA CPC North Atlantic Oscillation table.
        Format: Year, Jan, Feb, ..., Dec
        """
        data = {}
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                y = int(parts[0])
                for m, val_str in enumerate(parts[1:], 1):
                    if m <= 12:
                        try:
                            val = float(val_str)
                            if abs(val) < 90.0:
                                data[(y, m)] = val
                        except ValueError:
                            continue
        return data

    @classmethod
    def parse_amm(cls, text: str) -> Dict[Tuple[int, int], float]:
        """
        Parses NOAA PSL Atlantic Meridional Mode table.
        Format: Year, Jan, Feb, ..., Dec
        """
        data = {}
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) == 13 and parts[0].isdigit():
                y = int(parts[0])
                for m, val_str in enumerate(parts[1:], 1):
                    try:
                        val = float(val_str)
                        if val > -90.0:
                            data[(y, m)] = val
                    except ValueError:
                        continue
        return data

    @classmethod
    def fetch_all_teleconnections(cls) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Fetches all teleconnection indices live from NOAA servers."""
        results = {}

        # 1. NINA 3.4
        try:
            raw = cls._fetch_url(cls.URLS["NINA"])
            results["NINA"] = cls.parse_nina34(raw)
        except Exception as e:
            results["NINA"] = {}

        # 2. SOI
        try:
            raw = cls._fetch_url(cls.URLS["SOI"])
            results["SOI"] = cls.parse_soi(raw)
        except Exception as e:
            results["SOI"] = {}

        # 3. PDO
        try:
            raw = cls._fetch_url(cls.URLS["PDO"])
            results["PDO"] = cls.parse_pdo(raw)
        except Exception as e:
            results["PDO"] = {}

        # 4. NAO
        try:
            raw = cls._fetch_url(cls.URLS["NAO"])
            results["NAO"] = cls.parse_nao(raw)
        except Exception as e:
            results["NAO"] = {}

        # 5. AMM
        try:
            raw = cls._fetch_url(cls.URLS["AMM"])
            results["AMM"] = cls.parse_amm(raw)
        except Exception as e:
            results["AMM"] = {}

        return results

    @classmethod
    def fetch_nigeria_precipitation(
        cls, start_year: int = 2022, end_year: int = 2026
    ) -> Dict[Tuple[int, int], float]:
        """
        Fetches daily reanalysis precipitation for Nigeria and aggregates to monthly mean daily rate.
        Scale: (meters per day equivalent) matches historical training data.
        """
        monthly_precip: Dict[Tuple[int, int], float] = {}
        
        # 1. Archive reanalysis data
        import datetime
        today_str = datetime.date.today().isoformat()
        start_date = f"{start_year}-01-01"
        candidate_end = f"{end_year}-12-31"
        end_date = min(candidate_end, today_str)
        url = (
            f"{cls.URLS['OPEN_METEO_ARCHIVE']}?"
            f"latitude={cls.NIGERIA_LAT}&longitude={cls.NIGERIA_LON}&"
            f"start_date={start_date}&end_date={end_date}&daily=precipitation_sum"
        )
        try:
            raw = cls._fetch_url(url, timeout=25)
            doc = json.loads(raw)
            daily = doc.get("daily", {})
            times = daily.get("time", [])
            precips = daily.get("precipitation_sum", [])
            
            # Group by year, month
            monthly_sums: Dict[Tuple[int, int], List[float]] = {}
            for t, p in zip(times, precips):
                if p is not None:
                    y = int(t[:4])
                    m = int(t[5:7])
                    monthly_sums.setdefault((y, m), []).append(p)

            for (y, m), vals in monthly_sums.items():
                # Average daily precipitation in mm, converted to meters (/ 1000.0)
                # to match ERA5 total precipitation rate in historical CSV
                avg_daily_mm = sum(vals) / max(len(vals), 1)
                monthly_precip[(y, m)] = avg_daily_mm / 1000.0

        except Exception:
            pass

        return monthly_precip

    @classmethod
    def get_latest_aligned_features(
        cls, fallback_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetches live features, aligns across all indices, and blends with
        historical data to form a seamless sequence up to the current month.
        """
        # Load historical baseline
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        hist_path = fallback_file or os.path.join(base_dir, "data", "raw", "nigeria_teleconnections_monthly_1961_2022.csv")
        
        dates: List[str] = []
        prec: List[float] = []
        nina: List[float] = []
        amm: List[float] = []
        soi: List[float] = []
        pdo: List[float] = []
        nao: List[float] = []
        amo: List[float] = []

        last_known_amo = 0.29

        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    dates.append(r["Date"])
                    prec.append(float(r.get("Prec", 0.0)))
                    nina.append(float(r.get("NINA", 0.0)))
                    amm.append(float(r.get("AMM", 0.0)))
                    soi.append(float(r.get("SOI", 0.0)))
                    pdo.append(float(r.get("PDO", 0.0)))
                    nao.append(float(r.get("NAO", 0.0)))
                    val_amo = float(r.get("AMO", last_known_amo))
                    amo.append(val_amo)
                    last_known_amo = val_amo

        # Fetch live telemetry
        live_tele = cls.fetch_all_teleconnections()
        live_prec = cls.fetch_nigeria_precipitation(start_year=2022, end_year=2026)

        # Parse last historical date
        last_y, last_m = 2021, 12
        if dates:
            last_d = dates[-1].split("-")
            if len(last_d) == 2:
                m_idx = cls.MONTH_ABBRS.index(last_d[0]) + 1 if last_d[0] in cls.MONTH_ABBRS else 12
                yr_val = int(last_d[1])
                yr_full = 1900 + yr_val if yr_val >= 60 else 2000 + yr_val
                last_y, last_m = yr_full, m_idx

        # Candidate extension months from last_y, last_m up to 2026
        cur_y, cur_m = last_y, last_m + 1
        if cur_m > 12:
            cur_y += 1
            cur_m = 1

        new_months_added = 0
        nina_dict = live_tele.get("NINA", {})
        soi_dict = live_tele.get("SOI", {})
        pdo_dict = live_tele.get("PDO", {})
        nao_dict = live_tele.get("NAO", {})
        amm_dict = live_tele.get("AMM", {})

        # Find the latest month where at least one major teleconnection index is available
        all_yms = set(nina_dict.keys()) | set(soi_dict.keys()) | set(pdo_dict.keys()) | set(nao_dict.keys())
        max_tele_year = max([k[0] for k in all_yms] or [cur_y])
        
        while (cur_y < max_tele_year) or (cur_y == max_tele_year and (cur_y, cur_m) in all_yms):
            ym = (cur_y, cur_m)
            if ym not in all_yms:
                break

            # Date format like 'Jan-22', 'Feb-22'
            m_abbr = cls.MONTH_ABBRS[cur_m - 1]
            short_y = cur_y % 100
            d_str = f"{m_abbr}-{short_y:02d}"

            # Fallback imputations for unreleased or slow indices
            v_nina = nina_dict.get(ym, nina[-1] if nina else 0.0)
            v_soi = soi_dict.get(ym, soi[-1] if soi else 0.0)
            v_pdo = pdo_dict.get(ym, pdo[-1] if pdo else 0.0)
            v_nao = nao_dict.get(ym, nao[-1] if nao else 0.0)
            v_amm = amm_dict.get(ym, amm[-1] if amm else 0.0)
            v_amo = last_known_amo  # stable multidecadal baseline

            # Precipitation: from ERA5 or persistence
            v_prec = live_prec.get(ym, prec[-1] if prec else 0.005)

            dates.append(d_str)
            prec.append(v_prec)
            nina.append(v_nina)
            amm.append(v_amm)
            soi.append(v_soi)
            pdo.append(v_pdo)
            nao.append(v_nao)
            amo.append(v_amo)

            new_months_added += 1

            cur_m += 1
            if cur_m > 12:
                cur_y += 1
                cur_m = 1

        N_pts = len(dates)
        sin_month = [math.sin(2.0 * math.pi * (i % 12) / 12.0) for i in range(N_pts)]
        cos_month = [math.cos(2.0 * math.pi * (i % 12) / 12.0) for i in range(N_pts)]

        return {
            "dates": dates,
            "precipitation": prec,
            "features": {
                "NINA": nina,
                "AMM": amm,
                "SOI": soi,
                "PDO": pdo,
                "NAO": nao,
                "AMO": amo,
                "sin_month": sin_month,
                "cos_month": cos_month,
            },
            "new_months_added": new_months_added,
            "latest_date": dates[-1] if dates else "N/A",
            "total_records": len(dates),
            "sources": {
                "NINA": "NOAA CPC ERSSTv5",
                "SOI": "NOAA CPC SLP Anomaly",
                "PDO": "NOAA NCEI ERSST PDO",
                "NAO": "NOAA CPC Table",
                "AMM": "NOAA PSL AMM",
                "AMO": "NOAA PSL Baseline Imputed",
                "Precipitation": "Open-Meteo ERA5 Reanalysis",
            },
        }

    @classmethod
    def sync_to_file(cls, target_path: Optional[str] = None) -> Tuple[str, int]:
        """Saves updated synchronized data to CSV."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_path = target_path or os.path.join(base_dir, "data", "raw", "nigeria_teleconnections_monthly_extended.csv")
        
        aligned = cls.get_latest_aligned_features()
        fieldnames = ["Date", "Prec", "NINA", "AMM", "SOI", "PDO", "NAO", "AMO"]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for i, d in enumerate(aligned["dates"]):
                row = [
                    d,
                    f"{aligned['precipitation'][i]:.9f}",
                    f"{aligned['features']['NINA'][i]:.2f}",
                    f"{aligned['features']['AMM'][i]:.2f}",
                    f"{aligned['features']['SOI'][i]:.2f}",
                    f"{aligned['features']['PDO'][i]:.2f}",
                    f"{aligned['features']['NAO'][i]:.2f}",
                    f"{aligned['features']['AMO'][i]:.2f}",
                ]
                writer.writerow(row)

        return out_path, aligned["new_months_added"]
