import os
import unittest

from ai_council.main import AICouncilServer


class ServerStartupTest(unittest.TestCase):
    def test_server_registers_handlers(self):
        os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
        server = AICouncilServer()
        self.assertEqual(server.server.name, "ai-council")


if __name__ == "__main__":
    unittest.main()
