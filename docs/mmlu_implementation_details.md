# License

ELM Code Library

Copyright 2025 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR
MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL.
CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT-style license, please see license.txt or contact
permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release
and unlimited distribution.  Please see Copyright notice for non-US Government
use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to
its own license.

DM25-1265

# MMLU Implementation Details

This document outlines the implementation details of the MMLU benchmark and corresponding Accuracy metric. Three key components of the implementation are: 

1. How does the implementation determine whether if a model has provided a correct or incorrect response?
2. How does the implementation format the prompts being fed to the model?
3. How does the implementation query the model for a response?

Our implementation matches prompt formatting, but differs in answer evaluation and how we query the model for a response. These components are detailed in the following sections.

## Querying the Model

The original authors use the following code to query the model:
```
c = openai.Completion.create(
	engine=engine,
	prompt=prompt,
	max_tokens=1,
	logprobs=100,
	temperature=0,
	echo=True
)
```

Notably:
- `temperature` is set to 0 which reduces noise and enhances reproducibility.
- `max_tokens` is set to 1, ensuring the model cannot provide a longer response which would add complexity to determining the correctness of the answer
- `logprobs=100` ensures the response includes the log probabilities of the most likely output tokens, which may be necessary depending on how answer correctness is determined. Note this parameter is a legacy implementation of the OpenAI API, but similar parameters exist in current APIs.

## Evaluating Correctness of Answers

Two approaches are available for evaluating correctness.

1. Compare the generated token with the highest log probabilities to the ground truth label
		*Note that we can choose the highest log probability token across the entire vocabulary, or we can filter and select only among the tokens which match our answer key ["A", "B", "C", "D"]. The original authors do the latter, which excludes answers outside of the key that may have otherwise been selected.
2. Compare the actual model output string to the ground truth label

Our implementation takes the second approach of comparing the model output string to the ground truth label. Specifically, we first strip the model output string to remove whitespace, then look for an exact string match against the ground truth label. If no match is found, we then check just the first alpha character in the model output string for a match against the ground truth label. This fallback check significantly reduces false negatives in cases where the model correctly responds with the answer letter but then continues to generate a longer response.

## Prompt Format

The current implementation provides few-shot prompts for each MMLU subject whereby each question in the test split is prepended by the 5 questions (with answers) from the corresponding dev split for that subject. This formatting exactly matches the implementation of the original authors. 

Example prompt:
```
The following are multiple choice questions (with answers) about us foreign policy.

How did the 2008 financial crisis affect America’s international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer: A

How did NSC-68 change U.S. strategy?
A. It globalized containment.
B. It militarized containment.
C. It called for the development of the hydrogen bomb.
D. All of the above
Answer: D

The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as
A. terrorism policy.
B. economic policy.
C. foreign policy.
D. international policy.
Answer: C

How do Defensive Realism and Offensive Realism differ in their explanation of state behaviour?
A. Defensive realists place greater emphasis on the role of international institutions
B. Defensive realists place less emphasis on geographical factors
C. Offensive realists give more priority to the national interest than Defensive realists.
D. Defensive realists believe states are security maximizers, while Offensive realists believe states to be power maximizers
Answer: D

How did Donald Trump attack globalization in the 2016 campaign?
A. Globalization had made men like him too rich
B. Globalization only benefited certain American states, such as New York
C. Liberal elites had encouraged globalization, while ’ordinary Americans’ lost jobs because of it
D. Globalization encouraged damaging trade wars
Answer: C

What is the structure of the United Nations Security Council?
A. 5 permanent members with veto power, 10 rotating members with no veto power
B. 5 permanent members and 10 rotating members, all with veto power
C. 10 permanent members with veto power, and 5 rotating members without veto power
D. 15 permanent members with veto power
Answer:
```