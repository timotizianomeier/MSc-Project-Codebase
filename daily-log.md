# Daily Log — Reachy Mini / ADHD SAR Thesis

---

## 📅 [10.07.2026] — Entry #8

### 🎯 Goals for Today
X UI bit for text input, later potentially further ways for context-ingestion
- Engagement detector given server hosting model weights is online again

### Status by module

Emotion recognition
- Try different DeepFace backends
- Try other frameworks

Engagement detecation
- Get working
- See if there are other modules

Context awareness
- Maybe show conversation in browser, check with Nicole, maybe pure ingestion layer also makes sense
- Add other ingestion methods, eg PDF

Other
- Improve TA profile, text different ones
- Try with Reachy cam
- Try with external cam, reroute inattention detection modules with these images
- Maybe add other bits, eg phone detection

---

### 🔬 Findings & Notes
- Could self-test with neetcode tasks as realistic use case

#### What I tried
- 

#### What worked
- UI ingestion layer for text

#### What didn't / open questions
- 

#### Random thoughts / ideas
- For enabling the chat interface next to speech:
    1. The OpenAI realtime API already supports text input via conversation.item.create — the same call used in send_idle_signal() and _handle_tool_result(). The audio path and text input path are separate, so adding text doesn't require touching the audio plumbing.
- Wifi version does seem a bit smoother than the Lite one
- Maybe better to first test in simulation
---

### 📦 End of Day Summary
**Shipped:** 
- 

**Blockers:**
- 

**Tomorrow:** 
- 

---
---

## 📅 [09.07.2026] — Entry #7

### 🎯 Goals for Today
- Demo with Nicole
- Build engagement detector

Answer next:

Order of build:
X Tweak profile to TA, get running
X Make emotion recognition work, webcam
- Make engagement detection work, webcam
WIP Add chat interface for students to interact with and ingest content, use NeetCode as example
- Improv inattention detection modules
- Maybe add other bits, eg phone detection
- Test with robot cam
- Add external camera, reroute inattention detection modules with these images

---

### 🔬 Findings & Notes
- Could self-test with neetcode tasks as realistic use case

#### What I tried
- Started building text UI interface to ingest more content
- emotion-recognition with Reachy Lite, camera seems lower quality, but generally worked. Still, deepface buggy for real face, easier with stock image, but validated that no real difference between simulation vs Lite version runs

#### What worked
- Text interface working, no UI yet

#### What didn't / open questions
- Model weights of Del Duchetto currently not possible to download, need to weight for response

#### Random thoughts / ideas
- For enabling the chat interface next to speech:
    1. The OpenAI realtime API already supports text input via conversation.item.create — the same call used in send_idle_signal() and _handle_tool_result(). The audio path and text input path are separate, so adding text doesn't require touching the audio plumbing.
- Wifi version does seem a bit smoother than the Lite one
- Maybe better to first test in simulation
---

### 📦 End of Day Summary
**Shipped:** 
- Text input for context-awareness first iteration

**Blockers:**
- Model weights from Del Duchetto not available

**Tomorrow:** 
- Engagement detector if response received
- UI bit for text input, later potentially further ways for context-ingestion

---
---

## 📅 [08.07.2026] — Entry #6

### 🎯 Goals for Today
- [WIP] Continuously understand relevant bits and pieces of codebase
- [WIP] Continue build in specified order as specified below

Answer next:

Order of build:
X Tweak prrofile to TA, get running
X Make emotion recognition work, robot cam (first sketch pipeline, input signals, output signals, data input, pre-processing, model in between)
- Make engagement detection work, robot cam
- Add chat interface for students to interact with and ingest content, use NeetCode as example
- Add external camera, reroute inattention detection modules with these images

---

### 🔬 Findings & Notes
- Maybe latest pull resolved local LLM altogether, need to check how powerful hf free tier is, otherwise try Gemma
- Might need to adjust idle behaviours, there are a lot of them right now
- Check out remember / forget tools that were added to keep session context
- Could self-test with neetcode tasks as realistic use case

#### What I tried
- Emotion recognition v1 up and running

#### What worked
- Helpful to use stock images for simulation

#### What didn't / open questions
- DeepFace is not very robust, need to experiment with other frameworks
- Might need to go through the most recently pulled architecture again
- Experiment with ADHD profile, maybe show Nicole

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
- Emotion recognition first iteration

**Blockers:** 

**Tomorrow:** 
- Understand engagement detection repo, then start implementing as separate module and branch
- Maybe switch meetings

---
---

## 📅 [07.07.2026] — Entry #5

### 🎯 Goals for Today
- [WIP] Continuously understand relevant bits and pieces of codebase
- [WIP] Continue build in specified order, specifically, 

Answer next:

Order of build:
WIP Understand existing modules as I knew them pre-holidays
WIP Then where do I need to build sth on top
- Get local LLM version running, hf or Gemma
X Tweak prrofile to TA, get running
- Make emotion recognition work, robot cam (first sketch pipeline, input signals, output signals, data input, pre-processing, model in between)
- Make engagement detection work, robot cam
- Add chat interface for students to interact with and ingest content, use NeetCode as example
- Add external camera, reroute inattention detection modules with these images

---

### 🔬 Findings & Notes
- Went through old notes on setup again
- Resolved forked vs. own repo topic
- Maybe latest pull resolved local LLM altogether, need to check how powerful hf free tier is, otherwise try Gemma
- Might need to adjust idle behaviours, there are a lot of them right now
- Check out remember / forget tools that were added to keep session context
- Maybe save conversation transcripts, just for debugging purposes

#### What I tried
- Set up the Reachy with hf's LLM

#### What worked
- hf LLM hosted works really well, have to try local one as well

#### What didn't / open questions
- Might need to go through the most recently pulled architecture again
- Experiment with ADHD profile, maybe show Nicole

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
- emotion_classifier and tests
- emotion_monitor and tests

**Blockers:** 

**Tomorrow:** 
- Continue with emotion wiring to realtime.py

---
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