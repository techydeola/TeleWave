"""
Unit tests for AIAdvisoryInterpreter module.
"""

import unittest
from unittest.mock import patch, MagicMock
import io
import json
from climatewave.advisory.interpreter import AIAdvisoryInterpreter, prompt_ai_configuration


class TestAIAdvisoryInterpreter(unittest.TestCase):

    def setUp(self):
        self.sample_data_wet = {
            "target_month": "Sep-26",
            "scenario_key": "live",
            "forecasted_volume_mm": 184.1,
            "forecasted_rate_m_day": 0.006136,
            "historical_month_avg_mm": 184.4,
            "anomaly_pct": -0.2,
            "wavelet_a3_pct": 44.8,
            "wavelet_d3_pct": 28.6,
            "wavelet_d2_pct": 17.6,
            "wavelet_d1_pct": 9.0,
            "d1_anomaly": False,
            "nina": 1.44,
            "soi": -1.10,
            "pdo": -1.84,
            "nao": -1.92,
            "amm": -0.16,
            "amo": 0.29,
            "pci": 0.744,
            "r2": 0.865,
        }

        self.sample_data_dry = {
            "target_month": "Dec-83",
            "scenario_key": "1983_drought",
            "forecasted_volume_mm": 22.0,
            "forecasted_rate_m_day": 0.00073,
            "historical_month_avg_mm": 95.0,
            "anomaly_pct": -76.8,
            "wavelet_a3_pct": 70.0,
            "wavelet_d3_pct": 19.5,
            "wavelet_d2_pct": 3.4,
            "wavelet_d1_pct": 6.7,
            "d1_anomaly": True,
            "nina": 2.80,
            "soi": -3.5,
            "pdo": 1.2,
            "nao": 0.5,
            "amm": -0.4,
            "amo": 0.29,
            "pci": 0.69,
            "r2": 0.65,
        }

    def test_system_prompt_grounding(self):
        interp = AIAdvisoryInterpreter()
        sys_prompt = interp.build_system_prompt()
        self.assertIn("STRICT SCIENTIFIC & INTERPRETATION RULES", sys_prompt)
        self.assertIn("West African monsoon", sys_prompt)
        self.assertIn("El Niño does NOT instantly shut down rainfall", sys_prompt)

    def test_user_prompt_formatting(self):
        interp = AIAdvisoryInterpreter()
        user_prompt = interp.build_user_prompt(self.sample_data_wet)
        self.assertIn("184.1 mm/month", user_prompt)
        self.assertIn("+1.44", user_prompt)
        self.assertIn("Sep-26", user_prompt)

    def test_deterministic_advisory_heavy_rain(self):
        # Without an API key, it generates a data-driven physical advisory
        interp = AIAdvisoryInterpreter(api_key=None)
        memo = interp.generate_advisory(self.sample_data_wet)
        self.assertIn("OPERATIONAL CLIMATE EARLY WARNING BULLETIN", memo)
        self.assertIn("ELEVATED MONSOON RUNOFF & FLASH FLOOD RISK", memo)
        self.assertIn("184.1 mm for Sep-26", memo)
        self.assertIn("28.6%", memo)

    def test_deterministic_advisory_drought(self):
        interp = AIAdvisoryInterpreter(api_key=None)
        memo = interp.generate_advisory(self.sample_data_dry)
        self.assertIn("ACUTE DROUGHT & EARLY SEASON CESSATION ALERT", memo)
        self.assertIn("accelerated withdrawal of monsoon rains", memo)

    @patch("urllib.request.urlopen")
    def test_mocked_llm_call(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": (
                            "MEMORANDUM TO NEMA & NIHSA\n"
                            "Target: September 2026\n"
                            "Status: High Flood Watch\n"
                            "The Wavelet-LSTM projects 184.1 mm of rainfall."
                        )
                    }
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        interp = AIAdvisoryInterpreter(api_key="mock-key-12345", model="gpt-4o")
        memo = interp.generate_advisory(self.sample_data_wet)
        self.assertIn("MEMORANDUM TO NEMA & NIHSA", memo)
        self.assertIn("184.1 mm of rainfall", memo)

    def test_no_model_specified_falls_back(self):
        # If API key is provided but model is empty/None, falls back to physics engine
        interp = AIAdvisoryInterpreter(api_key="mock-key", model="")
        memo = interp.generate_advisory(self.sample_data_wet)
        self.assertIn("Data-Driven Physics Engine", memo)

    def test_prompt_non_interactive_no_force(self):
        # When not interactive and force_prompt=False, should return None immediately
        with patch("sys.stdin.isatty", return_value=False):
            key, url, model = prompt_ai_configuration(force_prompt=False)
            self.assertIsNone(key)
            self.assertIsNone(url)
            self.assertIsNone(model)

    def test_prompt_declined(self):
        # User declines AI interpretation
        with patch("builtins.input", side_effect=["n"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertIsNone(key)
            self.assertIsNone(url)
            self.assertIsNone(model)

    def test_prompt_claude_shortcut(self):
        # User chooses Anthropic Claude (1) and selects default model (Claude 3.5 Sonnet)
        with patch("builtins.input", side_effect=["y", "1", "1", "sk-ant-api03-samplekey12345678"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "sk-ant-api03-samplekey12345678")
            self.assertEqual(url, "https://api.anthropic.com/v1")
            self.assertEqual(model, "claude-3-5-sonnet-20241022")

    def test_prompt_gemini_shortcut(self):
        # User chooses Google Gemini (2) and selects default model (Gemini 2.0 Flash)
        with patch("builtins.input", side_effect=["y", "2", "1", "AIzaSySampleKey1234567890"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "AIzaSySampleKey1234567890")
            self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/openai")
            self.assertEqual(model, "gemini-2.0-flash")

    def test_prompt_alias_selection(self):
        # User enters provider name 'gemini' directly instead of a number
        with patch("builtins.input", side_effect=["y", "gemini", "1", "AIzaSySampleKey1234567890"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "AIzaSySampleKey1234567890")
            self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/openai")
            self.assertEqual(model, "gemini-2.0-flash")

    def test_prompt_agentrouter_explicit_model(self):
        # User says yes, takes choice 9 (AgentRouter), types model string directly, provides key
        with patch("builtins.input", side_effect=["y", "9", "claude-3-5-sonnet", "sk-test1234567890"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "sk-test1234567890")
            self.assertEqual(url, "https://agentrouter.org/v1")
            self.assertEqual(model, "claude-3-5-sonnet")

    def test_prompt_model_menu_shortcut(self):
        # User chooses AgentRouter (9) and picks model #2 (GPT-4o) by number
        with patch("builtins.input", side_effect=["y", "9", "2", "sk-test1234567890"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "sk-test1234567890")
            self.assertEqual(url, "https://agentrouter.org/v1")
            self.assertEqual(model, "gpt-4o")

    def test_prompt_groq_shortcut(self):
        # User chooses Groq (5) and picks shortcut #1 (LLaMA 3.3 70B)
        with patch("builtins.input", side_effect=["y", "5", "1", "gsk-test987654321"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "gsk-test987654321")
            self.assertEqual(url, "https://api.groq.com/openai/v1")
            self.assertEqual(model, "llama-3.3-70b-versatile")

    def test_prompt_deepseek_provider(self):
        # User chooses DeepSeek (4) and picks shortcut #1 (DeepSeek V3 Chat)
        with patch("builtins.input", side_effect=["y", "4", "1", "sk-deepseek123"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "sk-deepseek123")
            self.assertEqual(url, "https://api.deepseek.com/v1")
            self.assertEqual(model, "deepseek-chat")

    def test_prompt_groq_custom_model(self):
        # User says yes, chooses Groq (5), specifies custom model string directly, provides key
        with patch("builtins.input", side_effect=["y", "5", "llama-3.1-8b-instant", "gsk-test987654321"]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertEqual(key, "gsk-test987654321")
            self.assertEqual(url, "https://api.groq.com/openai/v1")
            self.assertEqual(model, "llama-3.1-8b-instant")

    def test_prompt_empty_key_falls_back(self):
        # User says yes, enters model, but enters empty key
        with patch("builtins.input", side_effect=["y", "1", "claude-3-5-sonnet-20241022", ""]):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertIsNone(key)
            self.assertIsNone(url)
            self.assertIsNone(model)

    def test_prompt_keyboard_interrupt_handled(self):
        # KeyboardInterrupt should exit gracefully without raising
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            key, url, model = prompt_ai_configuration(force_prompt=True)
            self.assertIsNone(key)
            self.assertIsNone(url)
            self.assertIsNone(model)

    @patch("urllib.request.urlopen")
    def test_anthropic_native_messages_dispatch(self, mock_urlopen):
        # Anthropic calls use /messages endpoint with x-api-key and parse content[0].text
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "content": [
                {
                    "type": "text",
                    "text": "OPERATIONAL BULLETIN FROM CLAUDE\nTarget: Sep-26\nPrecipitation: 184 mm"
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        interp = AIAdvisoryInterpreter(
            api_key="sk-ant-testkey123",
            base_url="https://api.anthropic.com/v1",
            model="claude-3-5-sonnet-20241022",
        )
        memo = interp.generate_advisory(self.sample_data_wet)
        self.assertIn("OPERATIONAL BULLETIN FROM CLAUDE", memo)
        self.assertIn("Claude", memo)

        # Verify request URL and headers
        req_arg = mock_urlopen.call_args[0][0]
        self.assertEqual(req_arg.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(req_arg.headers.get("X-api-key"), "sk-ant-testkey123")
        self.assertEqual(req_arg.headers.get("Anthropic-version"), "2023-06-01")

    @patch("urllib.request.urlopen")
    def test_gemini_openai_compatible_dispatch(self, mock_urlopen):
        # Gemini calls use /chat/completions endpoint with Bearer auth and choices[0].message.content
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "OPERATIONAL BULLETIN FROM GEMINI\nTarget: Sep-26\nPrecipitation: 184 mm"
                    }
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        interp = AIAdvisoryInterpreter(
            api_key="AIzaSyTestKey123",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.0-flash",
        )
        memo = interp.generate_advisory(self.sample_data_wet)
        self.assertIn("OPERATIONAL BULLETIN FROM GEMINI", memo)
        self.assertIn("Gemini", memo)

        req_arg = mock_urlopen.call_args[0][0]
        self.assertEqual(req_arg.full_url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.assertEqual(req_arg.headers.get("Authorization"), "Bearer AIzaSyTestKey123")

    def test_key_prefix_auto_detection(self):
        # Anthropic key prefix auto-detection
        interp_ant = AIAdvisoryInterpreter(api_key="sk-ant-test98765")
        self.assertEqual(interp_ant.base_url, "https://api.anthropic.com/v1")

        # Gemini key prefix auto-detection
        interp_gem = AIAdvisoryInterpreter(api_key="AIzaSyTest12345")
        self.assertEqual(interp_gem.base_url, "https://generativelanguage.googleapis.com/v1beta/openai")

        # Groq key prefix auto-detection
        interp_groq = AIAdvisoryInterpreter(api_key="gsk_Test12345")
        self.assertEqual(interp_groq.base_url, "https://api.groq.com/openai/v1")

    def test_terminal_markdown_sanitization(self):
        # Verify markdown tokens (**, ###, ---, unicode dashes) are sanitized for terminal
        raw_markdown = (
            "### 1. EXECUTIVE STATUS\n"
            "• **Current Outlook:** August 2026 forecast = 253.5 mm/month.\n"
            "• **Threat Rating:** **HIGH** – elevated rainfall.\n"
            "---\n"
            "**[AGRICULTURE]**\n"
            "• Advise farmers to use flood-tolerant crops."
        )
        interp = AIAdvisoryInterpreter(api_key="mock", model="test-model")
        wrapped = interp._wrap_llm_advisory(raw_markdown, self.sample_data_wet)
        # Should not contain raw asterisks or markdown header hashes
        self.assertNotIn("**", wrapped)
        self.assertNotIn("###", wrapped)
        self.assertIn("1. EXECUTIVE STATUS", wrapped)
        self.assertIn("Current Outlook:", wrapped)
        self.assertIn("Threat Rating: HIGH - elevated rainfall.", wrapped)
        self.assertIn("[AGRICULTURE]", wrapped)


if __name__ == "__main__":
    unittest.main()
