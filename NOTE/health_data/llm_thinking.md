### **Development Log: Refactoring the Data Pipeline—Exploring the Feasibility of LLMs Optimizing Traditional Structured Data Analysis**

**Date**: August 29, 2025  
**Author**: JING  
**Topic**: Beyond Generation: Can LLMs Become Superior Processors for Structured Time-Series Data?

#### **1. Origin: Questioning Traditional Models**

The project began with skepticism about the existing technological paradigm:  
> "Almost all women's health apps on the market rely on traditional statistical models (e.g., cycle averages, regression) or classical machine learning models (e.g., XGBoost) to process users' structured record data (e.g., menstrual dates, symptom tags). This process is complex, rigid, and lacks personalization."

I then posed a core question:  
> **"Do we have to maintain this complex pipeline of feature engineering, model training, and prediction? Can the powerful sequence understanding, contextual learning, and reasoning capabilities demonstrated by large language models (LLMs) enable them to directly process these time-series data and achieve superior or equivalent performance?"**

#### **2. Evolution of Thought: From "Replacement" to "Optimization"**

My thinking did not stop at the simple idea of "replacement" but delved into the **paradigm-level optimizations** that LLMs might bring:

1. **The End of Feature Engineering?**
    - **Traditional Paradigm**: Requires data scientists to manually construct features such as "cycle length," "phase labels," "symptom frequency," "historical averages," etc., and then input them into the model.
    - **LLM Paradigm Vision**: Input all historical user data (dates, symptoms, tags) as a **time-series sequence**, directly in natural language or a specific text format, to the LLM. **Can the LLM infer cycle patterns, symptom associations, and other features from this "history"?** This would greatly simplify the process.

2. **Context-Aware Predictions**
    - **Traditional Paradigm**: Models typically only look at numbers and lack context. For example, they see "Day 30 of the cycle" but are unaware that the user recorded "high work stress" last week.
    - **LLM Paradigm Vision**: LLMs can simultaneously see "Day 30 of the cycle" and the record of "high stress." **Can it reason like a human, using 'stress' as context to predict 'this delay might be related to stress,' rather than just outputting a cold prediction date?** This suggests a more intelligent and personalized prediction.

3. **Learning**
    - **Traditional Paradigm**: `Structured Data -> Feature Engineering -> Model Training -> Prediction Results`
    - **LLM Paradigm Vision**: `(Structured Data + Natural Language Annotations) -> LLM -> Prediction Results + Reasoning Explanations`
    - My goal is to **explore the feasibility of the latter**. If feasible, we could significantly simplify the tech stack and potentially achieve more powerful model capabilities.

#### **3. Core Research Questions**

My thoughts crystallized into the following testable technical hypotheses:

1. **Performance Hypothesis**: In the task of menstrual cycle prediction, **can LLMs (using specific sequence-based prompt designs) match or even surpass traditional specialized models in accuracy (e.g., MAE/RMSE)?**
2. **Efficiency Hypothesis**: In **few-shot** or even **zero-shot** settings, can LLMs leverage their built-in common sense and reasoning abilities to quickly adapt to new users, alleviating the cold-start problem faced by traditional models?
3. **Value Overflow Hypothesis**: Even if performance is on par, **can LLMs, due to their inherent capabilities, provide additional value by generating explanations for prediction uncertainty (e.g., "Due to high stress records this cycle, prediction confidence is slightly lower") alongside the predicted date?**

#### **4. Methodology: How to Validate**

To validate the above hypotheses, I devised the following approach:

1. **Data Serialization**: Design a template to convert users' structured data into prompts understandable by LLMs. For example:  
    > "User's historical menstrual start dates: [Date1], [Date2], [Date3]... User's recorded symptoms: On [DateA], recorded 'headache'; on [DateB], recorded 'low mood'... Based on the above history, predict the most likely start date of the next menstrual cycle."

2. **Baseline Testing**: Select a strong traditional model (e.g., LSTM, XGBoost on temporal features) as the **baseline**.
3. **Comparative Experiments**: Compare the **baseline model** with **various LLMs (e.g., GPT-4, Claude, Llama2, etc.)** on the same test set using core metrics (e.g., MAE for predicted dates).
4. **In-Depth Analysis**: Not only compare numerical results but also **qualitatively analyze LLM error cases**. What are its failure modes? Are they systematic biases or errors that can be fixed