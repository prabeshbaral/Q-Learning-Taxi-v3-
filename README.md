# Taxi-v3 Q-Learning Implementation

This repository contains an implementation of the **Q-Learning** algorithm for the **Taxi-v3** environment using **OpenAI Gymnasium**.

## Description

The Taxi-v3 environment consists of a 5x5 grid where a taxi must pick up and drop off passengers at designated locations while optimizing its moves.

### Features:
- Uses **Q-Learning** to train the agent.
- Implements **ε-greedy exploration** with **epsilon decay**.
- Stores Q-values in a **Q-table**.
- Renders test episodes after training.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install gymnasium numpy
pip install "gymnasium[toy-text]"