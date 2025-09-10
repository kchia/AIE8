<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Introduction and Vibe Check</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE8/tree/main/00_AIM_Quicklinks)

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording | 🖼️ Slides | 👨‍💻 Repo | 📝 Homework | 📁 Feedback |
| :---------- | :--------------- | :----------- | :-------- | :------ | :---------- | :---------- |

## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** – these will be questions that you will be expected to gather the answer to! These can appear as general questions, or questions meant to spark a discussion in your breakout rooms!

- 🏗️ **Activities** – these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** – Take on a challenge! These builds require you to create something with minimal guidance outside of the documentation. Completing an Advanced Build earns full credit in place of doing the base assignment notebook questions/activities.

### Main Assignment

In the following assignment, you are required to take the app that you created for the AIE8 challenge (from [this repository](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge)) and conduct what is known, colloquially, as a "vibe check" on the application.

You will be required to submit a link to your GitHub, as well as screenshots of the completed "vibe checks" through the provided Google Form!

> NOTE: This will require you to make updates to your personal class repository, instructions on that process can be found [here](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)!

#### 🏗️ Activity #1:

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner.

   - Aspect Tested: Educational explanation capability, simplification of complex concepts, clarity of communication
   - Performance: The system provided a clear, beginner-friendly explanation using relatable real-world analogies (car example). It structured the response well with bullet points and numbered lists, making it easy to follow. The explanation covered key OOP concepts (objects, classes, attributes, methods) without being overly technical.

<img src="./Question%201.png" alt="Question 1" />

2. Read the following paragraph and provide a concise summary of the key points…

   - Aspect Tested: Reading comprehension, summarization skills, ability to extract key information
   - Performance: No paragraph was provided, and the system logically asked the user to provide the paragraph.

<img src="./Question%202.png" alt="Question 2" />

3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.

   - Aspect Tested: Creative writing ability, narrative construction, adherence to word limits
   - Performance: The system created an engaging, imaginative story (146 words - within limit) with vivid imagery, fast character development, and a meaningful theme. The narrative demonstrated creativity in the unlikely friendship between a robot and squirrel.

<img src="./Question%203.png" alt="Question 3" />

4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?

   - Aspect Tested: Mathematical reasoning, problem-solving, logical computation
   - Performance: The system correctly solved the mathematical problem, showed clear step-by-step calculations, and presented the answer in a well-formatted, easy-to-understand manner. The logic was sound and the presentation was clear.

<img src="./Question%204.png" alt="Question 4" />

5. Rewrite the following paragraph in a professional, formal tone…
   - Aspect Tested: Style transfer, tone adaptation, writing flexibility
   - Performance: Excellent. When provided with the complete paragraph, the system successfully transformed the casual, conversational tone into professional, formal language. Key improvements included: replacing casual phrases ("get their creative flow going") with formal alternatives ("stimulate creativity"), using more sophisticated vocabulary ("compels" instead of "forces", "inventive thinking" instead of "creativity"), restructuring sentences for better flow, and maintaining all original meaning while elevating the register.

<img src="./Question%205.png" alt="Question 5" />

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

#### A Note on Vibe Checking

> "Vibe checking" is an informal term for cursory unstructured and non-comprehensive evaluation of LLM-powered systems. The idea is to loosely evaluate our system to cover significant and crucial functions where failure would be immediately noticeable and severe.
>
> In essence, it's a first look to ensure your system isn't experiencing catastrophic failure.

#### ❓Question #1:

What are some limitations of vibe checking as an evaluation tool?

##### ✅ Answer:

1. It only tests a small sample of capabilities and may miss critical edge cases or failure modes that could appear in production use.

2. Without multiple test runs and quantitative metrics, we can't measure consistency, variance, or reliability of responses over time. Vibe-checking is not scaleable.

3. "Good" or "bad" performance is often based on human judgment rather than objective criteria, making comparisons difficult.

4. Simple test prompts don't capture the complexity of actual user interactions, multi-turn conversations, or domain-specific requirements.

5. Vibe checks don't measure latency, throughput, or resource usage, which are crucial for production systems.

6. As seen in our test (prompt #2), vibe checks might not reveal how the system handles ambiguous, incomplete, or malformed inputs.

7. No assessment of how the system handles adversarial inputs, prompt injections, or attempts to bypass safety measures.

### 🚧 Advanced Build (OPTIONAL):

Please make adjustments to your application that you believe will improve the vibe check you completed above, then deploy the changes to your Vercel domain [(see these instructions from your Challenge project)](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge/blob/main/README.md) and redo the above vibe check.

> NOTE: You may reach for improving the model, changing the prompt, or any other method.

#### 🏗️ Activity #1

##### Adjustments Made:

- Added preprocessing function (`preprocess_user_message`) to detect incomplete prompts ending with "..." and automatically append example content for demonstration
- Enhanced developer/system message via `get_enhanced_developer_message()` function that instructs the model to proactively demonstrate capabilities when requests seem incomplete
- Reduced temperature from 1.0 to 0.7 to improve response consistency while maintaining creativity
- Set max_tokens to 1000 to ensure responses are complete and not cut off
- Added specific handlers for summarization and rewriting requests that lack content

##### Limitations of the Adjustments Made

This works for a prototype, but production requires dynamic adaptation, robust error handling, comprehensive testing, and security measures rather than hardcoded string matching.

While the adjustments improve handling of incomplete prompts, they're essentially hardcoded patches rather than robust solutions. In a production environment, I'd implement dynamic prompt classification using a separate classifier model or LLM call to identify prompt types and missing components, rather than relying on simple string matching.

The fixed `temperature` and `max_tokens` values should be dynamically adjusted based on query type - creative tasks need higher temperature while analytical tasks need lower values, and token limits should scale with expected output length.

My current approach also lacks proper error handling and fallback mechanisms - production systems need graceful degradation when the preprocessing fails, comprehensive logging for debugging, and metrics tracking for success rates.

Additionally, the hardcoded example content doesn't scale - a production system would use a template library or few-shot example database that can be updated without code changes.

Most critically, these preprocessing functions should be thoroughly tested with edge cases, have configurable thresholds, and include user feedback loops to continuously improve the handling of ambiguous inputs.

The current solution also lacks security considerations like prompt injection protection and rate limiting that would be essential in production.

##### Results:

1. **OOP Explanation**: No degradation in explanation quality.

<img src="./Question 1 v2.png" alt="Question 1 v2" />

2. **Summarization (incomplete prompt)**: Instead of asking for the missing paragraph, the system now generates a sample paragraph about AI advancement and provides a concise summary. This demonstrates the summarization capability even when the test prompt is incomplete.

<img src="./Question 2 Improved.png" alt="Question 2 Improved" />

3. **Robot story**: The story quality remained high, and the creativity wasn't compromised by the lower temperature.

<img src="./Question 3 v2.png" alt="Question 3 v2" />

4. **Math problem**: The mathematical reasoning was already accurate and the changes didn't affect this capability. Still shows clear step-by-step calculation.

<img src="./Question 4 v2.png" alt="Question 4 v2" />

5. **Rewrite**: The system successfully demonstrates the style transfer to formal tone.

<img src="./Question 5 v2.png" alt="Question 5 v2" />

## Submitting Your Homework

### Main Assignment (Activity #1 only)

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE8 repo:
   - For your initial repo setup see [00_Setting Up Git/README.md](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)
   - To get the latest updates from AI Makerspace into your own AIE8 repo, run the following commands:
   ```
   git checkout main
   git pull upstream main
   git push origin main
   ```
2. **IMPORTANT:** Start Cursor from the `01_Prototyping Best Practices & Vibe Check` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s01-assignment`
4. Edit this `README.md` file (the one in your `AIE8/01_Prototyping Best Practices & Vibe Check` folder)
5. Perform a "Vibe check" evaluation your AI-Engineering-Challenge system using the five questions provided above
6. For each Activity question:
   - Define the “Aspect Tested”
   - Comment on how your system performed on it.
7. Provide an answer to `❓Question #1:` after the `✅ Answer:` prompt
8. Add, commit and push your modified `README.md` to your origin repository.

> (NOTE: You should not merge the new document into origin's main branch. This will spare you from update challenges for each future session.)

When submitting your homework, provide the GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.

### The Advanced Build:

1. Follow all of the steps (Steps 1 - 8) of the Main Assignment above
2. Document what you changed and the results you saw in the `Adjustments Made:` and `Results:` sections of the Advanced Build's Assignment #1
3. Add, commit and push your additional modifications to this `README.md` file to your origin repository.

When submitting your homework, provide the following on the form:

- The GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.
- The public Vercel URL to your updated Challenge project on your AIE8 repo.
