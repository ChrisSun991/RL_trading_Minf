import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
from environment.portfolio import max_drawdown, sharpe, sortino
from stock_trading import extract_from_infos, returns_from_cumulative


def ensemble_test(env, model_names, model_weights_list, model_portfolio_values_list, starting_model_name, window=14,
                  action_window=1, criteria_type='return'):
    # Selected Weights
    selected_actions = []
    selected_models = []
    selected_sharpe = []
    info_list = []

    selected_model_name = starting_model_name
    index = 0

    assert criteria_type in ['return', 'sharpe'], 'criteria type must be either return or sharpe'
    assert len(model_names) == len(model_weights_list), 'Number of models is to be equal to models in weights list'
    assert env.sim.steps == len(model_weights_list[0]) - 1, 'Number of steps is to be equal to weights list'

    # Start environment. Get first observation
    _, _, info = env.reset()

    # Pass through starting window with starting model
    for i in range(window):
        # Choose action/weights via selected agent
        weights = model_weights_list[model_names.index(selected_model_name)][index]

        selected_sharpe.append(0)

        # step forward in Portfolio Environment
        _, _, _, done, info, _, _ = env.step(weights)
        info_list.append(info)

        selected_actions.append(weights)
        selected_models.append(selected_model_name)

        index = index + 1

    # Calculate action steps based on action window
    action_steps = int((env.sim.steps - window) / action_window)

    done = False
    for i in range(action_steps):
        # Evaluate each agent via Sharpe Ratio
        criteria_list = []
        for model_portfolio_values in model_portfolio_values_list:
            # Calculate based on criteria type
            if criteria_type == 'return':
                criteria_list.append(np.mean(returns_from_cumulative(model_portfolio_values[(index - window):index])))
            elif criteria_type == 'sharpe':
                criteria_list.append(sharpe(returns_from_cumulative(model_portfolio_values[(index - window):index])))

                # Select model with highest sharpe ratio
        selected_model_name = model_names[criteria_list.index(max(criteria_list))]

        # Select agent to record weights
        for j in range(action_window):
            # Choose action/weights via selected agent
            weights = model_weights_list[model_names.index(selected_model_name)][index]

            selected_sharpe.append(max(criteria_list))

            # step forward in Portfolio Environment
            _, _, _, done, info, _, _ = env.step(weights)
            info_list.append(info)

            selected_actions.append(weights)
            selected_models.append(selected_model_name)

            index = index + 1

    # Pass through any remainder with the last selected agent
    while not done:
        # Choose action/weights via selected agent
        weights = model_weights_list[model_names.index(selected_model_name)][index]

        selected_sharpe.append(max(criteria_list))

        # step forward in Portfolio Environment
        _, _, _, done, info, _, _ = env.step(weights)
        info_list.append(info)

        selected_actions.append(weights)
        selected_models.append(selected_model_name)

        index = index + 1

    return selected_actions, selected_models, extract_from_infos(info_list, 'portfolio_value'), selected_sharpe