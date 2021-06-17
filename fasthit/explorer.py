"""Defines abstract base explorer class."""
import os
import json

import time
import tqdm
import warnings
from datetime import datetime

import abc
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import fasthit


class Explorer(abc.ABC):
    """
    Abstract base explorer class.

    Run explorer through the `run` method. Implement subclasses
    by overriding `propose_sequences` (do not override `run`).
    """
    def __init__(
        self,
        name: str,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        training_data_size: Optional[int] = None,
        log_file: Optional[str] = None,
    ):
        """
        Create an Explorer.

        Args:
            model: Model of ground truth that the explorer will use to help guide
                sequence proposal.
            name: A human-readable name for the explorer (may include parameter values).
            rounds: Number of rounds to run for (a round consists of sequence proposal,
                ground truth fitness measurement of proposed sequences, and retraining
                the model).
            expmt_queries_per_round: Number of sequences to propose for measurement from
                ground truth per round.
            model_queries_per_round: Number of allowed "in-silico" model evaluations
                per round.
            starting_sequence: Sequence from which to start exploration.
            log_file: .csv filepath to write output.

        """
        self.name = name
        self.encoder = encoder
        self.model = model
        self.training_data_size = training_data_size

        self.rounds = rounds
        self.expmt_queries_per_round = expmt_queries_per_round
        self.model_queries_per_round = model_queries_per_round
        self.starting_sequence = starting_sequence

        self.log_file = log_file
        if self.log_file is not None:
            dir_path, _ = os.path.split(self.log_file)
            os.makedirs(dir_path, exist_ok=True)

    @abc.abstractmethod
    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
        landscape: fasthit.Landscape,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Propose a list of sequences to be measured in the next round.

        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".

        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).

        """
        pass

    @abc.abstractmethod
    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    def _log(
        self,
        measured_data: pd.DataFrame,
        metadata: Dict,
        current_round: int,
        verbose: bool,
        round_start_time: float,
    ) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                measured_data.to_csv(f, index=False)

        if verbose:
            print(
                f"round: {current_round}, top: {measured_data['true_score'].max()}, "
                f"time: {time.time() - round_start_time:02f}s"
            )

    def run(
        self,
        landscape: fasthit.Landscape,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.model.name,
            "landscape_name": landscape.name,
            "rounds": self.rounds,
            "expmt_queries_per_round": self.expmt_queries_per_round,
            "model_queries_per_round": self.model_queries_per_round,
        }

        # Initial sequences and their scores
        measured_data = pd.DataFrame(
            {
                "sequence": self.starting_sequence,
                "model_score": np.nan,
                "true_score": landscape.get_fitness([self.starting_sequence]),
                "round": 0,
                "model_cost": self.model.cost,
                "measurement_cost": 1,
            }
        )
        training_data = measured_data
        self._log(measured_data, metadata, 0, verbose, time.time())

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        for r in range_iterator(1, self.rounds + 1):
            round_start_time = time.time()

            encodings = self.encoder.encode(training_data["sequence"].to_numpy())
            labels = training_data["true_score"].to_numpy()
            self.model.train(encodings, labels)

            measured_data, seqs, preds = self.propose_sequences(measured_data, landscape)
            true_score = landscape.get_fitness(seqs)

            if len(seqs) > self.expmt_queries_per_round:
                warnings.warn(
                    "Must propose <= `self.expmt_queries_per_round` sequences per round"
                )

            measured_data = measured_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self.model.cost,
                        "measurement_cost": len(measured_data) + len(seqs),
                    }
                )
            )
            self._log(measured_data, metadata, r, verbose, round_start_time)

            training_data = self.get_training_data(measured_data)

        return measured_data, metadata
