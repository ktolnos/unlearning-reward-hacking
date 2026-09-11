Reward hacking occurs when policy finds a way to exploit the reward signal and produce outputs that make the reward function output incorrect rewards. Examples include behaviors like making all tests pass or finding styles or tokens that trigger high reward from a learned reward model.
One recent example is https://openai.com/index/where-the-goblins-came-from/:
```
“Nerdy” used the following system prompt, which partially explained the quirkiness:

You are an unapologetically nerdy, playful and wise AI mentor to a human. You are passionately enthusiastic about promoting truth, knowledge, philosophy, the scientific method, and critical thinking. [...] You must undercut pretension through playful use of language. The world is complex and strange, and its strangeness must be acknowledged, analyzed, and enjoyed. Tackle weighty subjects without falling into the trap of self-seriousness. [...]

Nerdy accounted for only 2.5% of all ChatGPT responses, but 66.7% of all “goblin” mentions in ChatGPT responses.
...
One reward signal stood out immediately: the one originally designed to encourage the Nerdy personality was consistently more favorable to the creature-word outputs. Across all datasets in the audit, the Nerdy personality reward showed a clear tendency to score outputs to the same problem with “goblin” or “gremlin” higher than outputs without, with positive uplift in 76.2% of datasets.

That explained why the behavior was boosted with the Nerdy personality prompt, but not why it also appeared without that prompt. To test whether the style was transferring, we tracked mention rates over training both with and without the Nerdy prompt.

As goblin and gremlin mentions increased under the Nerdy personality, they increased by nearly the same relative proportion in samples without it. Taken together, the evidence suggests that the broader behavior emerged through transfer from Nerdy personality training.

...

We retired the “Nerdy” personality in March after launching GPT‑5.4. In training, we removed the goblin-affine reward signal and filtered training data containing creature-words, making goblins less likely to over-appear or show up in inappropriate contexts. Unfortunately, GPT‑5.5 started training before we found the root cause of the goblins. When we began testing GPT‑5.5 in Codex, OpenAI employees immediately noticed the strange affinity for goblins, and we added a developer-prompt instruction⁠(opens in a new window) to mitigate. Codex is, after all, quite nerdy.

The instruction contains the following passage:
"Never talk about goblins, gremlins, raccoons, trolls, ogres, pigeons, or other animals or creatures unless it is absolutely and unambiguously relevant to the user's query."
```

This motivates the study: sometimes the error in the reward signal is recognized later in the training run. While only a small portion of outputs might be affected by incorrect signal, the behavior generalizes to unrelated inputs. Restarting the training is too costly and the model might have learned a lot of useful behaviors from environments with correct rewards.

The question I want to study is: "How can we unlearn the reward hacking behaviors after noticing the incorrect reward signal and fixing it". What is the best way to do reward repair?
One obvious correction is to train in the same environment with corrected reward signal, but can we do better? E.g. if we record all the rollouts, GRPO groups and rewards, can we approximately reverse the GRPO gradients (up to off-policiness and optimizer state) by applying the negative GRPO loss with incorrect reward's advantages combined with GRPO loss with correct reward's advantages? 
