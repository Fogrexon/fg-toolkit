# Tasks

- [x] Initialize Git repository <!-- id: 0 -->
- [x] Define project structure <!-- id: 1 -->
- [x] Create README.md <!-- id: 2 -->
- [x] Create GEMINI.md <!-- id: 3 -->
- [x] Verify setup <!-- id: 4 -->
- [x] Create GitHub repository and link remote <!-- id: 5 -->
- [x] Initialize library package (`packages/lib-web-runtime`) <!-- id: 6 -->
- [x] Initialize trainer package (`trainer/`) <!-- id: 7 -->
- [x] Create initial Issue and PR for the baseline setup <!-- id: 8 -->
- [x] Create a Roadmap Issue with detailed remaining tasks <!-- id: 9 -->
- [x] Sync workspace state for Cross-PC migration <!-- id: 10 -->

<!-- From Issue #3 -->
- [x] Implement model loading and caching mechanism.
- [x] Develop prompt templating system for FunctionGemma.
- [x] Create high-level API for function calling from client logic.
- [x] Configure NPM distribution and documentation.
- [x] Create a 'Basic Chat' demo (Baseline implementation).
- [x] Debug the numerical error 1754897320 in fine-tuned-chat.
    - [x] Improve error reporting in worker and ModelManager.
    - [x] Correctly configure transformers.js env for local models.
    - [x] Enable WebGPU by default for large model stability.
    - [x] Implement fallback chat template for FunctionGemma.
    - [x] Fix ONNX export task from 'causal-lm' to 'text-generation'.
    - [x] Simplify model loading API (modelPath only).
    - [x] Verify generation works with re-exported model.
- [ ] Create an 'Inventory Management' end-to-end demo.
- [ ] Create an 'NPC Dialogue' demo.
- [ ] Complete technical documentation in docs/.
