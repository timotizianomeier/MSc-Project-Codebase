# Daily Log — Reachy Mini / ADHD SAR Thesis

---

## 📅 [06.07.2026] — Entry #4

### 🎯 Goals for Today
- [WIP] Go through it, understand it bit by bit
- [ ] Continue understanding what was done before holidays
- [ ] Then continue with other bits needed before I can add on
- [X] Fix hardware issues
- [ ] Try set up with local LLM Gemma

Answer next:
- Better to first setup local LLM, then continue building?
- Does fork and copy repo structure make sense?

Order of build:
- Understand existing modules as I knew them pre-holidays
- Then where do I need to build sth on top
- Get local LLM version running
- Tweak prrofile to TA, get running
- Make emotion recognition work, robot cam
- Make engagement detection work, robot cam
- Add chat interface for students to interact with and ingest content, use NeetCode as example
- Add external camera, reroute inattention detection modules with these images

---

### 🔬 Findings & Notes
- Worked through the Reachy hardware issue and fixed it
- Added tips-and-tricks.md to the lab folder

#### What I tried
- 

#### What worked
- 

#### What didn't / open questions
- 

#### Random thoughts / ideas
- For switching out OpenAI key with free tier HF: 
    1. Add a HuggingFaceRealtimeHandler class alongside the existing OpenaiRealtimeHandler
    2. Read BACKEND_PROVIDER from the env and select which handler to instantiate at startup
    3. Add HF_TOKEN / HF_REALTIME_WS_URL wiring
- For enabling the chat interface next to speech:
    1. The OpenAI realtime API already supports text input via conversation.item.create — the same call used in send_idle_signal() and _handle_tool_result(). The audio path and text input path are separate, so adding text doesn't require touching the audio plumbing.
- Wifi version does seem a bit smoother than the Lite one
- Maybe better to first test in simulation
---

### 📦 End of Day Summary
**Shipped:** 

**Blockers:** 

**Tomorrow:** 
- Continue with understanding existing setup and artifact, then iterate quickly

---
---

## 📅 [23.06.2026] — Entry #3

### 🎯 Goals for Today
- [WIP] Go through it, understand it bit by bit, continue at config.py and go step-by-step
- [ ] Then slowly build on top, start from personality, then maybe emotiuon recognition and engagement detection, then chat integration, then external webcam etc.

---

### 🔬 Findings & Notes
> 

#### What I tried
- 

#### What worked
- 

#### What didn't / open questions
- 

#### Random thoughts / ideas
- For switching out OpenAI key with free tier HF: 
    1. Add a HuggingFaceRealtimeHandler class alongside the existing OpenaiRealtimeHandler
    2. Read BACKEND_PROVIDER from the env and select which handler to instantiate at startup
    3. Add HF_TOKEN / HF_REALTIME_WS_URL wiring
- For enabling the chat interface next to speech:
    1. The OpenAI realtime API already supports text input via conversation.item.create — the same call used in send_idle_signal() and _handle_tool_result(). The audio path and text input path are separate, so adding text doesn't require touching the audio plumbing.
- Wifi version does seem a bit smoother than the Lite one
- Maybe better to first test in simulation
---

### 📦 End of Day Summary
**Shipped:** 
Forked and setup working LLM comversation module

**Blockers:** 

**Tomorrow:** 
Continue with configure.py
Fix hardware issues

---
---

## 📅 [22.06.2026] — Entry #2

### 🎯 Goals for Today
- [x] Fork the LLM convrsation app
- [WIP] Go through it, understand it bit by bit
- [x] Test with OpenAI key if it works 

---

### 🔬 Findings & Notes
> 

#### What I tried
- Fork existing LLM module, get to run and understand

#### What worked
- Successfully tweaked personality in instructions.txt and available tools in tools.txt

#### What didn't / open questions
- Need to understand what is actually crucial and what is not, too large to understand non-crucial bits as well, need to focus

#### Random thoughts / ideas
- For switching out OpenAI key with free tier HF: 
    1. Add a HuggingFaceRealtimeHandler class alongside the existing OpenaiRealtimeHandler
    2. Read BACKEND_PROVIDER from the env and select which handler to instantiate at startup
    3. Add HF_TOKEN / HF_REALTIME_WS_URL wiring
- For enabling the chat interface next to speech:
    1. The OpenAI realtime API already supports text input via conversation.item.create — the same call used in send_idle_signal() and _handle_tool_result(). The audio path and text input path are separate, so adding text doesn't require touching the audio plumbing.

---

### 📦 End of Day Summary
**Shipped:** 
Forked and setup working LLM comversation module

**Blockers:** 

**Tomorrow:** 
Continue with configure.py
Fix hardware issues

---
---

## 📅 [18.06.2026] — Entry #1

### 🎯 Goals for Today
- [x] Read through reachy mini agent creatin guide
- [x] Setup reachy mini onversation app

---

### 🔬 Findings & Notes
> Agentic creation guide is fairly straightforward, but it will take quite some time to understand their conversation module

#### What I tried
- Create a new app with reachy mini app assistant

#### What worked
- Creating a new app is fairly straightforward

#### What didn't / open questions
- Go slower when building
- Probably best to first fully understand the reachy mini comversation app

#### Random thoughts / ideas
- Wifi version does seem a bit smoother than the Lite one
- Maybe better to first test in simulation

---

### 📦 End of Day Summary
**Shipped:** 
Stripped back version of LLM module, but seems to have major bugs.

**Blockers:** 

**Tomorrow:** 
Debug hardware issues, then understand LLM conversation module

---
---