# Success Metrics Tracker

Update this file after each integration test sprint.
Project is NOT demo-ready until all P0 and P1 metrics show ✅ PASS.

| # | Metric | Target | Result | Date Tested | Status |
|---|---|---|---|---|---|
| M1 | Disease detection accuracy | >90% on PlantVillage test set | — | — | 🔄 Pending |
| M2 | End-to-end response time | <30 seconds | — | — | 🔄 Pending |
| M3 | Language coverage (5 languages) | All 5 functional | — | — | 🔄 Pending |
| M4 | SMS fallback delivery on 2G | 100% delivery | — | — | 🔄 Pending |
| M5 | Demo usable by non-tech user | 0 training needed | — | — | 🔄 Pending |
| M6 | GitHub stars Month 1 | 50+ stars | — | — | 🔄 Pending |
| M7 | KVK locator accuracy | Correct for 5 test PINs | — | — | 🔄 Pending |
| M8 | Treatment recommendation quality | Agronomically correct | — | — | 🔄 Pending |

## Notes
- M1 measurement: `python scripts/eval_accuracy.py --split test`
- M2 measurement: Check `logs/response_times.log` after 10 test messages
- M3 measurement: Run `pytest tests/test_language_agent.py -v`
- M5 measurement: User test session notes in `docs/user_test_notes.md`
