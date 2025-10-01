# Synthetic Data Generation Demo Script
**Duration: ~5 minutes**

---

## INTRO (30 seconds)
**[SCREEN: Show both notebook files in VS Code/file explorer]**

Hey everyone! Today I'm excited to walk you through my work on synthetic data generation for RAG evaluation. I completed the main assignment using RAGAS, and then built something completely from scratch using LangGraph and Evol-Instruct. Let's dive in and see how these different approaches work!

---

## PART 1: RAGAS WORKFLOW (2 minutes)

**[SCREEN: Open main notebook - Synthetic_Data_Generation_RAGAS_&_LangSmith_Assignment.ipynb]**

So first, let me show you the main assignment using RAGAS. The key insight here is that RAGAS uses a knowledge graph approach to create synthetic test data.

**[SCREEN: Scroll to the data loading section - cell 14]**

We start by loading our use case data - this is research about how people actually use ChatGPT. The documents get processed through LangChain's document loader.

**[SCREEN: Move to knowledge graph section - cells 19-25]**

Now here's where RAGAS gets interesting. Instead of just randomly generating questions, it builds a knowledge graph. Look at this - we start with 150 document nodes, and after applying transformations, we end up with 211 nodes and over 6,000 relationships!

**[SCREEN: Show the knowledge graph creation and transformations]**

These transformations are doing some really smart work - creating summaries, extracting headlines, finding themes, and then using embeddings to connect related concepts. It's like creating a map of how all the information relates to each other.

**[SCREEN: Scroll to query synthesizers section - cells 29-30]**

The magic happens with these query synthesizers. We have three types:
- Single-hop questions that need just one chunk
- Multi-hop abstract questions that require synthesis
- Multi-hop specific questions that need concrete facts from multiple sources

**[SCREEN: Show the generated testset results]**

And here's what we get - a diverse set of questions with different complexity levels. Each question comes with reference contexts and answers, ready for evaluation.

**[SCREEN: Jump to LangSmith integration - cells 38-41]**

The workflow integrates beautifully with LangSmith. We create a dataset, upload our synthetic data, and we're ready to evaluate our RAG pipeline against this test set.

---

## PART 2: RAG EVALUATION (30 seconds)

**[SCREEN: Show RAG chain evaluation - cells 64 and 79]**

Quick look at the evaluation results - we tested two versions of our RAG chain. The first was basic, the second was "dope-ified" with larger chunks, better embeddings, and a more engaging prompt.

**[SCREEN: Show comparison screenshot if available]**

You can see the evaluation results: QA and helpfulness both stayed consistent at around 0.82, but dopeness dramatically improved from 0.02 to 1.0 - mission accomplished! But notice the trade-off in latency - better quality came at the cost of response time.

---

## PART 3: ADVANCED BUILD - LANGGRAPH APPROACH (2 minutes)

**[SCREEN: Switch to Advanced_Build_LangGraph_Synthetic_Data.ipynb]**

Now for the fun part - my from-scratch implementation! Instead of RAGAS's knowledge graph, I built a LangGraph agent using Evol-Instruct methodology.

**[SCREEN: Show the architecture diagram in cell 0]**

Look at this pipeline - it's much more straightforward than the knowledge graph approach. We have six clear nodes in a linear flow, versus RAGAS's complex graph with 211 nodes and over 6,000 relationships. Each node does one specific job, connected in a simple sequence.

**[SCREEN: Scroll to state schema - cell 8]**

The key is this typed state schema. Every piece of data flowing through the pipeline is strongly typed, which makes debugging so much easier than navigating a complex graph.

**[SCREEN: Show Evol-Instruct prompts - cell 14]**

Here's where the Evol-Instruct magic happens. Instead of using graph relationships, I implemented three evolution strategies based on the WizardLM research:
- Simple evolution adds constraints and detail requirements
- Multi-context evolution requires synthesis across documents
- Reasoning evolution demands logical analysis and hypothetical thinking

**[SCREEN: Show the LangGraph workflow creation - cell 20]**

Building the workflow is clean and declarative. Six nodes, linear connections, compile and run. No complex graph traversal algorithms needed.

**[SCREEN: Show pipeline execution - cell 22]**

Watch this execute - each node reports its progress clearly. We start with 64 documents, create 182 chunks, generate 10 seed questions, evolve them into 10 complex questions, then generate answers and retrieve contexts via semantic search.

**[SCREEN: Show the three outputs - cell 24]**

And here are our three required outputs - perfectly formatted for evaluation frameworks. The questions show real complexity evolution, and everything links together through unique IDs.

---

## COMPARISON & WRAP-UP (1 minute)

**[SCREEN: Show both notebooks side by side]**

So what's the difference? RAGAS gives you a comprehensive, battle-tested system with sophisticated knowledge graph relationships. My LangGraph approach gives you simplicity, maintainability, and full control over the evolution process.

**[SCREEN: Point to the JSON output files]**

Both approaches produce evaluation-ready datasets, but the LangGraph version is much easier to debug, modify, and understand. Plus, implementing Evol-Instruct from the research gives you insight into how instruction evolution actually works.

**[SCREEN: Back to both notebooks]**

The real win here is understanding that there are multiple ways to solve the synthetic data generation problem. RAGAS's knowledge graph approach is powerful for complex document relationships, while LangGraph's linear pipeline is perfect when you want transparency and control.

Both approaches successfully generate high-quality synthetic test data that can dramatically improve your RAG evaluation process. Thanks for watching, and happy building!

---

**[SCREEN: Show final results/comparison one more time]**

## Screen Direction Notes:
- Keep transitions smooth between notebooks
- Highlight key code sections while narrating
- Show outputs/results clearly
- Use split screen when comparing approaches
- End with both notebooks visible to emphasize the comparison