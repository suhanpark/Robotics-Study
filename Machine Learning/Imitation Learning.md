[Lecture](https://youtu.be/XQuLVDEMm1g?si=YV87JNPZTk-ALpgm)
# Formal Definition of Imitation Learning

• **State**: $\mathcal{s}$ $\in$ $\mathcal{S}$
	• May be partially observed (e.g., game screen)
• **Action**: $a$ $\in$ $\mathcal{A}$
	• May be discrete or continuous (e.g., turn angle, speed)
• **Policy**: ${\pi}_\theta$ : $S \rightarrow \mathcal{A}$
	• We want to learn the policy parameters $\theta$
• **Optimal Action**: $a^* \in \mathcal{A}$
	• Provided by expert demonstrator
• **Optimal Policy**: $\pi^* : S \rightarrow \mathcal{A}$
	• Provided by expert demonstrator
• **State Dynamics**: $P(s_{i+1} | s_i, a_i)$
	• Simulator, typically not known to policy
• **Often Deterministic**: $s_{i+1} = T(s_i, a_i)$
	• Deterministic mapping
• **Rollout**: Given $s_0$, sequentially execute $a_i = \pi_\theta(s_i)$ & sample $s_{i+1} \sim P(s_{i+1} | s_i, a_i)$ yields trajectory $\tau = (s_0, a_0, s_1, a_1, ...)$
• **Loss Function**: $\mathcal{L}(a^*, a)$
	• Loss of action $a$ given optimal action $a*$

## General Imitation Learning

### $$\underset{\theta}{\arg\min} \; \mathbb{E}_{s \sim P(s | \pi_{\theta})} \: [\mathcal{L}(\pi^*(s), \pi_{\theta}(s))]$$

• State distribution $P(s | \pi_{\theta})$ depends on rollout determined by current policy $\pi_{\theta}$


<div align="center">
<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsMyxbMSwwLCJQXzAiXSxbMCwxLCJzIl0sWzIsMSwiXFxwaV9cXHRoZXRhIl0sWzEsMiwiIiwwLHsiY3VydmUiOi0yfV0sWzIsMSwiIiwwLHsib2Zmc2V0IjotMywiY3VydmUiOi0yfV0sWzAsMywiIiwwLHsic2hvcnRlbiI6eyJ0YXJnZXQiOjIwfX1dXQ==&embed" width="432" height="304" style="border-radius: 8px; border: none;"></iframe>
</div>

## Behavior Cloning

### $$
\arg\min_{\theta} \underbrace{\mathbb{E}_{(s^*, a^*) \sim P^*} 
\left[ \mathcal{L} \left(a^*, \pi_\theta(s^*) \right) \right]}_{
\sum_{i=1}^N \mathcal{L} \left(a_i^*, \pi_\theta(s_i^*) \right)}
$$

• State distribution $P^*$ provided by expert
• Reduces to supervised learning problem

<div align="center">
<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsMixbMCwxLCIocywgYV4qKSJdLFswLDAsIlBeKiJdLFsxLDBdXQ==&embed" width="194" height="304" style="align: center; border-radius: 8px; border: none;"></iframe>
</div>


### Challenges of Behavior Cloning

• Behavior cloning reasons only about immediate next step
• Behavior cloning makes IID (Independent and Identically Distributed) assumption
	• Next state is sampled from states observed during expert demonstration
	• Thus, next state is sampled independently from action predicted by current policy
• What if $\pi_{\theta}$ (our policy) makes mistake?
	• Enters new states that haven't been observed before
	• New states not sampled from same (expert) distribution anymore
		• Model doesn't know what to do
	• Cannot recover, catastrophic failure in the worst case

##### DAgger – Data Aggregation
[Paper](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)

• Iteratively build a set of inputs that the final policy is to encounter based on previous experience. Query expert for aggregate dataset.
• But can easily overfit to main mode of demonstrations
• High training variance (random initialization, order of data)
• Issues
	• Works reasonably well in manipulation task, but not in self-driving
		• Because it easily overfits to the main mode of demonstration
		

<div align="center">
<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsNyxbMSwwLCJcXHRleHR7UG9saWN5fSJdLFsxLDIsIlxcdGV4dHtPbi1Qb2xpY3kgRGF0YX0iXSxbMCwxLCJcXHRleHR7RGF0YXNldH0iXSxbMiwxLCJcXHRleHR7RW52aXJvbm1lbnR9Il0sWzAsMCwiXFx0ZXh0eyhUcmFpbil9Il0sWzIsMCwiXFx0ZXh0eyhSb2xsb3V0KX0iXSxbMCwyLCJcXHRleHR7KEFnZ3JlZ2F0ZSl9Il0sWzIsMCwiIiwwLHsiY3VydmUiOi0yfV0sWzEsMiwiIiwwLHsiY3VydmUiOi0zfV0sWzAsMywiIiwwLHsiY3VydmUiOi0yfV0sWzMsMSwiIiwwLHsiY3VydmUiOi0zfV1d&embed" width="550" height="270" style="border-radius: 8px; border: none;"></iframe>
</div>

1. Starts with fixed data
2. Rollout based on train policy
3. Query expert from the new dataset
4. Aggregate the data
5. Train
6. Repeat the process

##### DAgger with Critical States and Replay Buffer
[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf)

• Selectively subsample **Critical states** from the collected on-policy data based on the utility they provide to the learned policy in terms of driving behavior.
• Incorporate a **replay buffer** which progressively focuses on the high uncertainty regions of the policy's state distribution.
•  Mixing Critical States and Dataset are challenging and does not work well since they're too different.


<div align="center">
<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsOSxbMSwwLCJcXHRleHR7UG9saWN5fSJdLFswLDEsIlxcdGV4dHtEYXRhc2V0fSJdLFsyLDEsIlxcdGV4dHtFbnZpcm9ubWVudH0iXSxbMCwwLCJcXHRleHR7KFRyYWluKX0iXSxbMiwwLCJcXHRleHR7KFJvbGxvdXQpfSJdLFsyLDIsIlxcdGV4dHtPbi1Qb2xpY3kgRGF0YX0iXSxbMSwzLCJcXHRleHR7Q3JpdGljYWwgU3RhdGVzfSJdLFsyLDMsIlxcdGV4dHsoU2FtcGxlKX0iXSxbMCwzLCJcXHRleHR7UmVwbGF5IEJ1ZmZlcn0iXSxbMSwwLCIiLDAseyJjdXJ2ZSI6LTJ9XSxbMCwyLCIiLDAseyJjdXJ2ZSI6LTJ9XSxbMiw1XSxbNSw2LCIiLDAseyJjdXJ2ZSI6LTJ9XSxbOCwxLCIiLDAseyJjdXJ2ZSI6LTJ9XSxbMSw4LCIiLDAseyJjdXJ2ZSI6LTJ9XSxbNiw4XV0=&embed" width="500" height="350" style="border-radius: 8px; border: none;"></iframe>
</div>

##### PilotNet
[Paper](https://arxiv.org/pdf/2010.08776)

• Data augmentation by 3 cameras and virtually shifted/rotated images assuming the world is flat (homography), adjusting the steering angle appropriately.

<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsMTIsWzAsMCwiXFx0ZXh0e1JlY29yZGVkIHN0ZWVyaW5nIHdoZWVsIGFuZ2xlfSJdLFsyLDAsIlxcdGV4dHtBZGp1c3QgZm9yIHNoaWZ0IGFuZCByb3RhdGlvbn0iXSxbMCwyLCJcXHRleHR7TGVmdCBDYW1lcmF9Il0sWzAsMywiXFx0ZXh0e0NlbnRlciBDYW1lcmF9Il0sWzAsNCwiXFx0ZXh0e1JpZ2h0IENhbWVyYX0iXSxbMiwzLCJcXHRleHR7UmFuZG9tIFNoaWZ0IGFuZCByb3RhdGlvbn0iXSxbMywzLCJcXHRleHR7Q05OfSJdLFszLDUsIlxcdGV4dHtCYWNrIHByb3BhZ2F0aW9ufSJdLFs1LDIsIlxcdGV4dHvigJN9Il0sWzUsNCwiXFx0ZXh0e1tFcnJvcl19Il0sWzMsMSwiXFx0ZXh0e1tEZXNpcmVkIHN0ZWVyaW5nIGNvbW1hbmRdfSJdLFszLDIsIlxcdGV4dHtbTmV0d29yayBjb21wdXRlZCBzdGVlcmluZyBjb21tYW5kXX0iXSxbMCwxXSxbMiw1LCIiLDAseyJjdXJ2ZSI6MX1dLFszLDVdLFs0LDUsIiIsMix7ImN1cnZlIjotMX1dLFs1LDZdLFs3LDZdLFsxLDgsIiIsMCx7ImN1cnZlIjotNSwic3R5bGUiOnsiaGVhZCI6eyJuYW1lIjoibm9uZSJ9fX1dLFs2LDhdLFs4LDcsIiIsMCx7ImN1cnZlIjotNX1dLFsxMCwxOCwiIiwwLHsic2hvcnRlbiI6eyJ0YXJnZXQiOjIwfSwic3R5bGUiOnsiaGVhZCI6eyJuYW1lIjoibm9uZSJ9fX1dLFsxMSwxOSwiIiwwLHsic2hvcnRlbiI6eyJ0YXJnZXQiOjIwfSwic3R5bGUiOnsiaGVhZCI6eyJuYW1lIjoibm9uZSJ9fX1dXQ==&embed" width="1000" height="408" style="border-radius: 8px; border: none;"></iframe>

##### Visual Backprop
[Paper](https://arxiv.org/pdf/1611.05418)

• Central Idea: find **salient image regions** that lead to high activations
• Forward pass, then iteratively scale-up activations
• Test if shift in salient objects affects predicted turn radius more strongly

![[Screenshot 2025-01-14 at 8.41.34 PM.png]]

![[Screenshot 2025-01-14 at 8.44.35 PM.png]]

![[Screenshot 2025-01-14 at 8.45.32 PM.png]]


