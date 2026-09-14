"""
AI-Powered Climate Advisory & Interpretation Engine for TeleWave.
=======================================================================

Bridges quantitative machine-learning outputs (wavelet energy bands,
anomaly spikes, teleconnection indices, and LSTM forecasts) with
decision-maker language via OpenAI-compatible endpoints (AgentRouter, OpenRouter, etc.).

Features:
  - Strictly grounded system prompt preventing hallucination.
  - Pure Python stdlib HTTP client (urllib.request) — zero pip dependencies.
  - Intelligent, data-driven fallback when offline or API key is absent.
"""

import sys
import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple


def _sanitize_terminal_line(line: str) -> str:
    """Strips Markdown syntax (bold, italics, code, headers) for clean terminal display."""
    # Remove markdown headers (#, ##, ###, ####)
    line = re.sub(r"^#{1,6}\s*", "", line)
    # Remove bold/italics combinations (**bold**, *italic*, __bold__, _italic_)
    line = re.sub(r"\*{2,3}(.*?)\*{2,3}", r"\1", line)
    line = re.sub(r"_{2,3}(.*?)_{2,3}", r"\1", line)
    line = re.sub(r"\*(.*?)\*", r"\1", line)
    line = re.sub(r"_(.*?)_", r"\1", line)
    # Remove inline code backticks
    line = re.sub(r"`(.*?)`", r"\1", line)
    # Convert markdown link syntax [text](url) -> text
    line = re.sub(r"\[(.*?)\]\([^\)]*\)", r"\1", line)
    # Convert unicode dashes to standard ASCII hyphen
    line = line.replace("–", "-").replace("—", "-")
    # Replace non-breaking and narrow spaces
    line = line.replace("\u202f", " ").replace("\u00a0", " ")
    return line.strip()


class AIAdvisoryInterpreter:
    """Interprets TeleWave quantitative forecasts into actionable memos."""

    DEFAULT_BASE_URL = "https://agentrouter.org/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 25,
    ):
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("CLAUDE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or os.environ.get("GROK_API_KEY")
            or os.environ.get("MISTRAL_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("AGENTROUTER_API_KEY")
        )

        detected_url = self.DEFAULT_BASE_URL
        if self.api_key:
            if self.api_key.startswith("sk-ant-"):
                detected_url = "https://api.anthropic.com/v1"
            elif self.api_key.startswith("AIzaSy"):
                detected_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif self.api_key.startswith("gsk_"):
                detected_url = "https://api.groq.com/openai/v1"
            elif self.api_key.startswith("xai-"):
                detected_url = "https://api.x.ai/v1"
            elif self.api_key.startswith("sk-or-"):
                detected_url = "https://openrouter.ai/api/v1"

        self.base_url = (
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("GEMINI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("GROQ_BASE_URL")
            or os.environ.get("AGENTROUTER_BASE_URL")
            or detected_url
        ).rstrip("/")
        self.model = (model or os.environ.get("TELEWAVE_LLM_MODEL") or "").strip()
        self.timeout = timeout

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def build_system_prompt(self) -> str:
        return (
            "You are the Chief Hydrometeorological Advisor to the Nigerian Hydrological Services "
            "Agency (NIHSA) and the National Emergency Management Agency (NEMA).\n"
            "Your responsibility is to translate raw quantitative machine learning and wavelet "
            "metrics from TeleWave into a clear, authoritative, physically grounded "
            "Operational Climate Early Warning Memorandum.\n\n"
            "STRICT SCIENTIFIC & INTERPRETATION RULES:\n"
            "1. Ground all conclusions ONLY in the numerical data provided in the prompt.\n"
            "2. RECONCILE MACRO OCEAN INDICES WITH LOCAL REGIONAL FORECASTS:\n"
            "   - If an active El Niño is observed (NINA3.4 > +0.5) BUT the forecasted precipitation "
            "is high (> 120 mm/month) and seasonal wavelet energy (D3) is elevated, explain the physical "
            "reality: the West African monsoon and ITCZ remain convective and active through peak season. "
            "El Niño does NOT instantly shut down rainfall in peak monsoon; its suppression typically exhibits "
            "a temporal lag, impacting post-monsoon cessation timing or dry-season replenishment rather than "
            "canceling mid-to-late monsoon rains.\n"
            "   - If rainfall is high and D1/D3 energy is strong, explicitly warn of flash flooding and water "
            "accumulation across low-lying northern territories (e.g. Sokoto, Kano, Kebbi, Zamfara, Borno) "
            "and major river confluences (Niger-Benue / Lokoja).\n"
            "3. Structure your memorandum concisely with:\n"
            "   - EXECUTIVE STATUS & THREAT LEVEL\n"
            "   - PHYSICAL TELEMETRY SYNTHESIS (reconciling ENSO with local forecast)\n"
            "   - SECTOR DIRECTIVES: [AGRICULTURE], [DAMS & HYDROPOWER], [DISASTER & FLOOD WATCH]\n"
            "4. Keep the output focused, professional, and directly actionable (under 350 words).\n"
            "5. PURE TERMINAL FORMATTING (STRICTLY NO MARKDOWN):\n"
            "   - Output pure plain text ONLY. Absolutely NO Markdown styling: no asterisks for bold (**word**), no italics (*word*), no code backticks, no HTML.\n"
            "   - Use plain capital letters for emphasis and section headers (e.g. THREAT RATING: HIGH, [AGRICULTURE], [DAMS & HYDROPOWER], [DISASTER & FLOOD WATCH]).\n"
            "   - Use standard bullet points ('• ' or '- ') for list items.\n"
            "   - Absolutely NO Markdown tables (| ... |) or markdown headers (#, ##, ###).\n"
            "   - Ensure all three sector directives are completely written out."
        )

    def build_user_prompt(self, data: Dict[str, Any]) -> str:
        return (
            f"Please generate the Operational Climate Memorandum based on the following verified "
            f"quantitative model output from TeleWave:\n\n"
            f"• Target Period: {data.get('target_month', 'Upcoming Month')}\n"
            f"• Scenario Type: {data.get('scenario_key', 'Operational Monitoring')}\n"
            f"• Forecasted Precipitation: {data.get('forecasted_volume_mm', 0.0):.1f} mm/month "
            f"({data.get('forecasted_rate_m_day', 0.0):.6f} m/day)\n"
            f"• Historical Climatological Average: {data.get('historical_month_avg_mm', 0.0):.1f} mm/month\n"
            f"• Deviation from Climatology: {data.get('anomaly_pct', 0.0):+.1f}%\n"
            f"• Multiresolution Wavelet Energies: "
            f"A3 (Trend)={data.get('wavelet_a3_pct', 0.0):.1f}%, "
            f"D3 (Seasonal)={data.get('wavelet_d3_pct', 0.0):.1f}%, "
            f"D2 (Sub-seasonal)={data.get('wavelet_d2_pct', 0.0):.1f}%, "
            f"D1 (High-freq noise/shock)={data.get('wavelet_d1_pct', 0.0):.1f}%\n"
            f"• D1 High-Frequency Anomaly Detected: {'YES (2-sigma threshold crossed)' if data.get('d1_anomaly') else 'NO'}\n"
            f"• Global Teleconnections:\n"
            f"  - NINA 3.4 (Pacific ENSO)    : {data.get('nina', 0.0):+.2f}\n"
            f"  - SOI (Southern Oscillation) : {data.get('soi', 0.0):+.2f}\n"
            f"  - PDO (Pacific Decadal Osc.) : {data.get('pdo', 0.0):+.2f}\n"
            f"  - NAO (North Atlantic Osc.)  : {data.get('nao', 0.0):+.2f}\n"
            f"  - AMM (Atlantic Meridional)  : {data.get('amm', 0.0):+.2f}\n"
            f"  - AMO (Atlantic Multidecadal): {data.get('amo', 0.0):+.2f}\n"
            f"• Model Benchmark Reliability: PCI = {data.get('pci', 0.70):.2f} "
            f"(Directional Accuracy), R² = {data.get('r2', 0.85):.3f}\n"
        )

    def generate_advisory(self, data: Dict[str, Any]) -> str:
        """
        Generates the decision advisory memo.
        Uses LLM if API key and model are configured; otherwise uses deterministic fallback.
        """
        if self.has_api_key and self.model:
            try:
                raw_memo = self._call_llm(data)
                if raw_memo and len(raw_memo.strip()) > 50:
                    return self._wrap_llm_advisory(raw_memo, data)
            except Exception as e:
                err_msg = str(e)
                if hasattr(e, "read"):
                    try:
                        err_msg = e.read().decode("utf-8")
                    except Exception:
                        pass
                print(f"  ⚠️  Note: LLM API returned an error ({err_msg}).")
                print("      Falling back to Data-Driven Physics Engine.\n")
        elif self.has_api_key and not self.model:
            print("  ⚠️  Note: No model ID specified for API key. Falling back to Data-Driven Physics Engine.\n")

        return self._generate_deterministic_advisory(data)

    def _wrap_llm_advisory(self, llm_content: str, data: Dict[str, Any]) -> str:
        """Formats LLM text inside the official memorandum box."""
        import textwrap
        target = data.get("target_month", "Upcoming Month")
        col_w = 67

        # Determine provider tag dynamically
        base_lower = (self.base_url or "").lower()
        if "anthropic" in base_lower:
            prov = "Claude"
        elif "generativelanguage.googleapis.com" in base_lower or "gemini" in base_lower:
            prov = "Gemini"
        elif "agentrouter" in base_lower:
            prov = "AgentRouter"
        elif "openrouter" in base_lower:
            prov = "OpenRouter"
        elif "groq" in base_lower:
            prov = "Groq"
        elif "deepseek" in base_lower:
            prov = "DeepSeek"
        elif "x.ai" in base_lower:
            prov = "Grok"
        elif "mistral" in base_lower:
            prov = "Mistral"
        elif "openai" in base_lower:
            prov = "OpenAI"
        else:
            prov = "AI"

        model_tag = self.model if self.model else "LLM"
        engine_tag = f"{prov} ({model_tag})"
        if len(engine_tag) > 22:
            engine_tag = engine_tag[:19] + "..."

        lines = []
        for raw_line in llm_content.strip().split("\n"):
            line_str = raw_line.strip()
            if not line_str:
                lines.append(f"  │{' ' * (col_w + 2)}│")
                continue

            # Convert markdown horizontal rule to neat box divider
            if line_str in ("---", "***", "___"):
                lines.append(f"  ├{'─' * (col_w + 2)}┤")
                continue

            sanitized = _sanitize_terminal_line(line_str)
            if not sanitized:
                continue

            wrapped = textwrap.wrap(sanitized, width=col_w)
            for w in wrapped:
                lines.append(f"  │  {w:<{col_w}} │")

        body = "\n".join(lines)
        return f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │           OPERATIONAL CLIMATE EARLY WARNING BULLETIN               │
  │           Nigeria Hydrological Services Agency (NIHSA) / NEMA       │
  │           Target Horizon: {target:<12} | Engine: {engine_tag:<22} │
  ├─────────────────────────────────────────────────────────────────────┤
{body}
  └─────────────────────────────────────────────────────────────────────┘
"""

    def _call_llm(self, data: Dict[str, Any]) -> str:
        """Dispatches request to Anthropic Messages API or OpenAI-compatible endpoint."""
        base_url = self.base_url.rstrip("/")
        is_anthropic = "api.anthropic.com" in base_url

        if is_anthropic:
            url = f"{base_url}/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": "TeleWave/2.0",
            }
            payload = {
                "model": self.model or "claude-3-5-sonnet-20241022",
                "max_tokens": 1200,
                "system": self.build_system_prompt(),
                "messages": [
                    {"role": "user", "content": self.build_user_prompt(data)}
                ],
                "temperature": 0.2,
            }
        else:
            url = f"{base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "codex_cli_rs/0.1.0",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": self.build_user_prompt(data)},
                ],
                "temperature": 0.2,
                "max_tokens": 1200,
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)

            if is_anthropic:
                content_blocks = result.get("content", [])
                text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                return "\n".join(text_parts).strip()
            else:
                choices = result.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "").strip()

        raise RuntimeError("No valid completion content returned by API.")

    def _generate_deterministic_advisory(self, data: Dict[str, Any]) -> str:
        """
        Intelligent, data-driven fallback when no API key is provided.
        Directly conditions on forecasted millimeters rather than coarse macro assumptions.
        """
        target = data.get("target_month", "Upcoming Month")
        precip_mm = data.get("forecasted_volume_mm", 0.0)
        pci = data.get("pci", 0.74)
        nina = data.get("nina", 0.0)
        soi = data.get("soi", 0.0)
        d3_pct = data.get("wavelet_d3_pct", 0.0)
        is_el_nino = nina > 0.5
        is_heavy_rain = precip_mm >= 120.0

        if is_heavy_rain:
            status = "ELEVATED MONSOON RUNOFF & FLASH FLOOD RISK"
            synthesis = (
                f"Although Pacific ENSO monitors indicate an emerging El Niño (NINA3.4 = {nina:+.2f}), "
                f"the quantitative Wavelet-LSTM projects robust precipitation of {precip_mm:.1f} mm "
                f"for {target}, supported by strong D3 seasonal energy ({d3_pct:.1f}%). "
                f"Macro El Niño suppression has not prematurely terminated the West African monsoon; "
                f"active ITCZ convection continues to drive heavy precipitation."
            )
            agri_advice = (
                f"Maintain normal late-season crop cultivation. Monitor for waterlogging in poorly "
                f"drained cereal farmlands (Sokoto, Kebbi, Kano, Borno, Adamawa). Prepare storage "
                f"facilities against high post-harvest humidity."
            )
            water_advice = (
                f"Kainji, Jebba, and Shiroro Dam operators should maintain controlled spillway releases "
                f"to accommodate ongoing peak inflows while buffering reservoir capacity for the dry season."
            )
            disaster_advice = (
                f"NEMA & State SEMA units: Saturated river basins across the Niger-Benue confluence and "
                f"lower Delta present acute flash-flood vulnerabilities. Maintain community alert networks "
                f"and pre-position evacuation resources."
            )
        elif precip_mm < 40.0 and is_el_nino:
            status = "ACUTE DROUGHT & EARLY SEASON CESSATION ALERT"
            synthesis = (
                f"Pacific El Niño coupling (NINA3.4 = {nina:+.2f}, SOI = {soi:+.2f}) has triggered "
                f"an accelerated withdrawal of monsoon rains. Forecasted volume ({precip_mm:.1f} mm) "
                f"is significantly depressed relative to historical expectations."
            )
            agri_advice = (
                f"Advise northern agrarian communities on urgent harvesting of standing grain crops. "
                f"Inhibit late planting of moisture-sensitive cultivars."
            )
            water_advice = (
                f"Mandate conservation protocols across major reservoirs; limit non-essential hydro-discharge."
            )
            disaster_advice = (
                f"Mobilize strategic grain reserves and activate regional drought contingency measures."
            )
        else:
            status = "MODERATE SEASONAL CONDITIONS — STANDARD VIGILANCE"
            synthesis = (
                f"Model projects {precip_mm:.1f} mm for {target}, adhering closely to baseline seasonal expectations. "
                f"Teleconnection indices reflect a transitional atmospheric state."
            )
            agri_advice = "Follow standard agro-climatic calendar for harvesting and land preparation."
            water_advice = "Operate reservoirs under standard rule-curve guidelines."
            disaster_advice = "Routine meteorological monitoring across active flood plains."

        import textwrap

        engine_tag = "Data-Driven Physics Engine"
        col_w = 67

        def format_block(text: str) -> str:
            lines = textwrap.wrap(text, width=col_w)
            return "\n".join(f"  │  {l:<{col_w}} │" for l in lines)

        def format_bullet(header: str, text: str) -> str:
            prefix = f"• [{header}]: "
            indent = " " * 4
            lines = textwrap.wrap(text, width=col_w - 4)
            res = [f"  │  • [{header}]:".ljust(72) + "│"]
            for l in lines:
                res.append(f"  │    {l:<{col_w - 2}} │")
            return "\n".join(res)

        synthesis_block = format_block(synthesis)
        agri_block = format_bullet("AGRICULTURE", agri_advice)
        water_block = format_bullet("DAMS & HYDROPOWER", water_advice)
        disaster_block = format_bullet("DISASTER & FLOOD WATCH", disaster_advice)

        memo = f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │           OPERATIONAL CLIMATE EARLY WARNING BULLETIN               │
  │           Nigeria Hydrological Services Agency (NIHSA) / NEMA       │
  │           Target Horizon: {target:<12} | Engine: {engine_tag:<22} │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ALERT STATUS: {status:<53}│
  │                                                                     │
  │  PHYSICAL SYNTHESIS & TELEMETRY:                                    │
{synthesis_block}
  │                                                                     │
  │  OPERATIONAL DIRECTIVES:                                            │
{agri_block}
{water_block}
{disaster_block}
  │                                                                     │
  │  MODEL METRICS: Directional Accuracy (PCI) = {pci:.2f} | R² = {data.get('r2', 0.85):.3f}    │
  │  INPUT STREAM: 6 NOAA Indices (NINA={nina:+.2f}, SOI={soi:+.2f}) + ERA5      │
  └─────────────────────────────────────────────────────────────────────┘
"""
        return memo


def prompt_ai_configuration(
    force_prompt: bool = False,
    scenario_name: str = "2026 Live Climate Monitor",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Interactively prompts the user in the terminal to configure AI interpretation.

    Parameters
    ----------
    force_prompt : bool
        If True, prompts even if sys.stdin.isatty() is False (useful for tests).
    scenario_name : str
        Human-readable scenario title displayed in the prompt header.

    Returns
    -------
    Tuple of (api_key, base_url, model) or (None, None, None) if declined/skipped.
    """
    if not force_prompt and not sys.stdin.isatty():
        return None, None, None

    print("\n" + "─" * 78)
    print(f"  🤖 AI ADVISORY INTERPRETATION SETUP ({scenario_name})")
    print("  Translate quantitative Wavelet-LSTM forecasts into an operational memo.")
    print("─" * 78)

    try:
        ans = input("  Would you like to use an AI model for interpretation? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  [Skipped AI configuration. Using internal Physics Engine.]\n")
        return None, None, None

    if ans not in ("y", "yes"):
        print("  ✔ Using internal Data-Driven Physics Engine (default).\n")
        return None, None, None

    PROVIDER_CATALOG = {
        "1": {
            "name": "Anthropic Claude",
            "aliases": ["claude", "anthropic"],
            "base_url": "https://api.anthropic.com/v1",
            "key_hint": "sk-ant-...",
            "models": [
                ("Claude 3.5 Sonnet (Recommended)", "claude-3-5-sonnet-20241022"),
                ("Claude 3.5 Haiku (Ultra Fast & Responsive)", "claude-3-5-haiku-20241022"),
                ("Claude 3 Opus (Deep Comprehensive Synthesis)", "claude-3-opus-20240229"),
            ],
        },
        "2": {
            "name": "Google Gemini",
            "aliases": ["gemini", "google"],
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "key_hint": "AIzaSy...",
            "models": [
                ("Gemini 2.0 Flash (Recommended — Ultra Fast)", "gemini-2.0-flash"),
                ("Gemini 1.5 Pro (Deep Reasoning & Multimodal)", "gemini-1.5-pro"),
                ("Gemini 1.5 Flash (Lightweight & Efficient)", "gemini-1.5-flash"),
            ],
        },
        "3": {
            "name": "OpenAI (ChatGPT)",
            "aliases": ["openai", "chatgpt", "gpt"],
            "base_url": "https://api.openai.com/v1",
            "key_hint": "sk-...",
            "models": [
                ("GPT-4o (Omni Flagship)", "gpt-4o"),
                ("GPT-4o mini (Fast & Cost-Effective)", "gpt-4o-mini"),
                ("o3-mini (Advanced Reasoning)", "o3-mini"),
                ("GPT-4 Turbo", "gpt-4-turbo"),
            ],
        },
        "4": {
            "name": "DeepSeek",
            "aliases": ["deepseek"],
            "base_url": "https://api.deepseek.com/v1",
            "key_hint": "sk-...",
            "models": [
                ("DeepSeek V3 (Chat — General Intelligence)", "deepseek-chat"),
                ("DeepSeek R1 (Deep Chain-of-Thought Reasoning)", "deepseek-reasoner"),
            ],
        },
        "5": {
            "name": "Groq",
            "aliases": ["groq"],
            "base_url": "https://api.groq.com/openai/v1",
            "key_hint": "gsk_...",
            "models": [
                ("LLaMA 3.3 70B Versatile (Recommended)", "llama-3.3-70b-versatile"),
                ("LLaMA 3.1 8B Instant (Ultra Fast)", "llama-3.1-8b-instant"),
                ("Mixtral 8x7B", "mixtral-8x7b-32768"),
                ("Gemma 2 9B", "gemma2-9b-it"),
            ],
        },
        "6": {
            "name": "xAI (Grok)",
            "aliases": ["grok", "xai"],
            "base_url": "https://api.x.ai/v1",
            "key_hint": "xai-...",
            "models": [
                ("Grok 2 (Latest Flagship)", "grok-2-latest"),
                ("Grok Beta", "grok-beta"),
            ],
        },
        "7": {
            "name": "Mistral AI",
            "aliases": ["mistral"],
            "base_url": "https://api.mistral.ai/v1",
            "key_hint": "Mistral API key",
            "models": [
                ("Mistral Large (Flagship)", "mistral-large-latest"),
                ("Mistral Small (Fast)", "mistral-small-latest"),
                ("Codestral (Reasoning & Code)", "codestral-latest"),
            ],
        },
        "8": {
            "name": "OpenRouter",
            "aliases": ["openrouter"],
            "base_url": "https://openrouter.ai/api/v1",
            "key_hint": "sk-or-...",
            "models": [
                ("Anthropic Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"),
                ("Google Gemini 2.0 Flash", "google/gemini-2.0-flash-001"),
                ("OpenAI GPT-4o", "openai/gpt-4o"),
                ("DeepSeek V3", "deepseek/deepseek-chat"),
                ("DeepSeek R1", "deepseek/deepseek-r1"),
                ("Meta LLaMA 3.3 70B", "meta-llama/llama-3.3-70b-instruct"),
            ],
        },
        "9": {
            "name": "AgentRouter",
            "aliases": ["agentrouter"],
            "base_url": "https://agentrouter.org/v1",
            "key_hint": "AgentRouter key",
            "models": [
                ("Claude 3.5 Sonnet", "claude-3-5-sonnet"),
                ("OpenAI GPT-4o", "gpt-4o"),
                ("DeepSeek V3", "deepseek-chat"),
                ("Google Gemini 2.0 Flash", "gemini-2.0-flash"),
                ("Meta LLaMA 3.3 70B", "llama-3.3-70b"),
            ],
        },
    }

    print("\n  Select AI Platform / Provider:")
    print("    [1]  Anthropic Claude  (claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus)")
    print("    [2]  Google Gemini     (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash)")
    print("    [3]  OpenAI (ChatGPT)  (gpt-4o, gpt-4o-mini, o3-mini)")
    print("    [4]  DeepSeek          (deepseek-chat V3, deepseek-reasoner R1)")
    print("    [5]  Groq              (llama-3.3-70b-versatile, llama-3.1-8b-instant)")
    print("    [6]  xAI (Grok)        (grok-2-latest, grok-beta)")
    print("    [7]  Mistral AI        (mistral-large, mistral-small, codestral)")
    print("    [8]  OpenRouter        (Multi-provider aggregator: openrouter.ai)")
    print("    [9]  AgentRouter       (Multi-model proxy: agentrouter.org)")
    print("    [10] Custom Endpoint   (Any custom OpenAI-compatible endpoint)")

    try:
        p_input = input("  Choice [1-10, default 1 (Claude)]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  [Skipped AI configuration.]\n")
        return None, None, None

    p_input = p_input or "1"
    p_norm = p_input.lower().strip()

    matched_key = None
    if p_norm in PROVIDER_CATALOG:
        matched_key = p_norm
    else:
        for k, v in PROVIDER_CATALOG.items():
            if p_norm in v.get("aliases", []):
                matched_key = k
                break

    if matched_key:
        provider_info = PROVIDER_CATALOG[matched_key]
        provider_name = provider_info["name"]
        base_url = provider_info["base_url"]
        preset_models = provider_info["models"]
        key_hint = provider_info.get("key_hint", "API key")

        print(f"\n  Select {provider_name} Model:")
        for idx, (m_label, m_id) in enumerate(preset_models, 1):
            print(f"    [{idx}] {m_label} ({m_id})")
        custom_idx = len(preset_models) + 1
        print(f"    [{custom_idx}] Custom / Enter another model ID")

        while True:
            try:
                m_input = input(f"  Model choice [1-{custom_idx}, default 1]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  [Skipped AI configuration.]\n")
                return None, None, None

            if not m_input:
                model = preset_models[0][1]
                break

            if m_input.isdigit():
                m_idx = int(m_input)
                if 1 <= m_idx <= len(preset_models):
                    model = preset_models[m_idx - 1][1]
                    break
                elif m_idx == custom_idx:
                    try:
                        model = input("  Enter custom Model ID: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        return None, None, None
                    if model:
                        break
                print(f"  ⚠️  Please select a valid option between 1 and {custom_idx}.")
            else:
                model = m_input
                break
    elif p_norm in ("10", "custom"):
        provider_name = "Custom Endpoint"
        key_hint = "API key"
        try:
            custom_url = input("  Enter Base URL (e.g. https://api.together.xyz/v1): ").strip()
            base_url = custom_url if custom_url else "https://agentrouter.org/v1"
            model = input("  Enter Model ID: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None, None, None
        if not model:
            print("  ⚠️  Model ID is required.")
            return None, None, None
    else:
        print("  Invalid provider choice. Defaulting to Anthropic Claude (claude-3-5-sonnet-20241022).")
        base_url = "https://api.anthropic.com/v1"
        provider_name = "Anthropic Claude"
        model = "claude-3-5-sonnet-20241022"
        key_hint = "sk-ant-..."

    # Ask for API key
    try:
        api_key = input(f"  Paste your {provider_name} API key ({key_hint}): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None, None, None

    if not api_key:
        print("  ⚠️  No API key entered. Reverting to Data-Driven Physics Engine.\n")
        return None, None, None

    masked_key = (api_key[:6] + "..." + api_key[-4:]) if len(api_key) > 10 else "***"
    print(f"  ✔ Configured: {model} on {provider_name} (Key: {masked_key})\n")
    return api_key, base_url, model
